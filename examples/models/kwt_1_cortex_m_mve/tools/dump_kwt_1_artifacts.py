# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).
"""Run a KWT-1 model through PT2E + edge lower + CortexMPassManager,
extract Layer records, and emit the generated/ headers the standalone
runner needs.

Phase 7 scope: a single hand-built encoder block (OneBlockKWT).
Phase 8+ extends to the real 12-layer KWT-1 once trained weights
are in scope.
"""

from __future__ import annotations

import argparse
import pathlib
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export, ExportedProgram
from torch.utils import _pytree as pytree
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.exir.memory_planning import MemoryPlanningAlgorithmSuite, apply_algo, greedy
from executorch.exir.passes.spec_prop_pass import make_spec
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
import executorch.backends.cortex_m.ops.operators  # noqa

from .extractors import extract_program
from .emitters import emit_arena, emit_inference_body, emit_params, emit_weights


def _populate_specs(gm: torch.fx.GraphModule) -> None:
    """Populate node.meta['spec'] for every node in the graph (in-place
    SpecPropPass equivalent that preserves node identities)."""
    for module in gm.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            meta_val = node.meta.get("val", None)
            if node.op == "output":
                node.meta["spec"] = pytree.tree_map(
                    lambda x: x.meta.get("spec", None) if hasattr(x, "meta") else None,
                    node.args[0],
                )
            else:
                node.meta["spec"] = pytree.tree_map(make_spec, meta_val)


def _plan_memory(prog: ExportedProgram, alignment: int = 16) -> tuple[dict, dict, int]:
    """Run exir.memory_planning greedy and return
    ({node_name: offset}, {node_name: nbytes}, arena_bytes)."""
    gm = prog.graph_module
    _populate_specs(gm)
    suite = MemoryPlanningAlgorithmSuite([greedy])
    bufsizes = apply_algo(
        suite, gm, alignment, prog.graph_signature,
        alloc_graph_input=False, alloc_graph_output=False,
    )
    offsets: dict[str, int] = {}
    sizes: dict[str, int] = {}
    for node in gm.graph.nodes:
        spec = node.meta.get("spec", None)
        if spec is None or not hasattr(spec, "mem_offset") or spec.mem_offset is None:
            continue
        offsets[node.name] = int(spec.mem_offset)
        sizes[node.name] = int(spec.allocated_memory)
    arena = int(bufsizes[1]) if len(bufsizes) > 1 else 0
    return offsets, sizes, arena


class OneBlockKWT(nn.Module):
    """Single transformer encoder block at KWT-1 dims (d=64, d_ff=256)."""
    def __init__(self, d=64, d_ff=256):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.lq = nn.Linear(d, d); self.lk = nn.Linear(d, d); self.lv = nn.Linear(d, d)
        self.lo = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, d_ff); self.ff2 = nn.Linear(d_ff, d)

    def forward(self, x):
        h = self.ln1(x)
        q = self.lq(h); k = self.lk(h); v = self.lv(h)
        scores = torch.bmm(q, k.transpose(-1, -2))
        probs = F.softmax(scores, dim=-1)
        a = torch.bmm(probs, v)
        a = self.lo(a)
        x = x + a
        h = self.ln2(x)
        h = self.ff2(F.gelu(self.ff1(h)))
        x = x + h
        return x


class KWT1Encoder(nn.Module):
    """Stack of N transformer encoder blocks.  Each block matches
    OneBlockKWT's structure (KWT-1 single-head, d_model=64, d_ff=256
    are the defaults).  N=12 gives the canonical KWT-1 encoder."""
    def __init__(self, num_blocks: int = 12, d: int = 64, d_ff: int = 256):
        super().__init__()
        self.blocks = nn.ModuleList(
            [OneBlockKWT(d=d, d_ff=d_ff) for _ in range(num_blocks)]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class KWT1(nn.Module):
    """Full KWT-1 / Keyword Transformer.

    Input is (B, S, mfcc_dim) post-MFCC features (e.g. (1, 98, 40) for
    Speech Commands v2 at 1s/16kHz with 25ms windows / 10ms hop / 40
    mel bands).  Architecture:

      x = mfcc_input
      e = Linear(mfcc_dim -> d_model) per timestep  (the "patch embed")
      e = e + learned positional encoding
      h = N encoder blocks
      pooled = mean over the sequence dimension      (no class token; mean
                                                       pooling sidesteps the
                                                       slice op that the
                                                       cortex_m backend does
                                                       not currently lower)
      logits = Linear(d_model -> num_classes)

    The canonical KWT-1 paper uses a learned class token + slice; we
    substitute mean pooling because (a) the cortex_m backend does not
    currently lower a 1-d slice in this PT2E shape and (b) the
    architecture is otherwise unchanged.  When real KWT-1 checkpoints
    land we can fold the [CLS] token + slice into a learned-weighted
    mean instead — same matmul, no slice.
    """
    def __init__(
        self,
        mfcc_dim: int = 40,
        seq_len: int = 98,
        num_blocks: int = 12,
        d: int = 64,
        d_ff: int = 256,
        num_classes: int = 35,
        use_pos_enc: bool = False,
    ):
        super().__init__()
        self.embed = nn.Linear(mfcc_dim, d)
        if use_pos_enc:
            self.pos = nn.Parameter(torch.zeros(1, seq_len, d))
            nn.init.normal_(self.pos, std=0.02)
        else:
            self.pos = None
        self.encoder = KWT1Encoder(num_blocks=num_blocks, d=d, d_ff=d_ff)
        self.head = nn.Linear(d, num_classes)

    def forward(self, x):
        # x: (B, S, mfcc_dim)
        h = self.embed(x)                            # (B, S, d)
        if self.pos is not None:
            h = h + self.pos
        h = self.encoder(h)                          # (B, S, d)
        pooled = h.mean(dim=1)                       # (B, d)
        return self.head(pooled)                     # (B, num_classes)


def lower_to_cortex_m(model: nn.Module, sample_inputs: tuple[torch.Tensor, ...]):
    torch.manual_seed(0)
    captured = export(model, sample_inputs).module()
    q = CortexMQuantizer()
    q.transform_for_annotation(captured)
    prepared = prepare_pt2e(captured, q)
    for _ in range(32):
        prepared(torch.randn_like(sample_inputs[0]))
    quantized = convert_pt2e(prepared)
    exported = export(quantized, sample_inputs)
    edge = to_edge_transform_and_lower(
        exported,
        compile_config=EdgeCompileConfig(
            preserve_ops=[torch.ops.aten.linear.default],
            _check_ir_validity=False,
        ),
    )
    return CortexMPassManager(edge.exported_program()).transform()


def emit_input_fixture(out_dir: pathlib.Path, ref_input: torch.Tensor) -> None:
    flat = ref_input.flatten().tolist()
    lines = [
        "#ifndef KWT_1_INPUT_FIXTURE_H_",
        "#define KWT_1_INPUT_FIXTURE_H_",
        "",
        "#include <stdint.h>",
        "",
        f"const float kwt_1_fixture_input[{len(flat)}] = {{",
    ]
    def _lit(v):
        s = f"{v:.9g}"
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s + "f"
    for i in range(0, len(flat), 6):
        chunk = flat[i : i + 6]
        lines.append("  " + ", ".join(_lit(v) for v in chunk) + ",")
    lines.append("};")
    lines.append("")
    lines.append("#endif")
    lines.append("")
    (out_dir / "input_fixture.h").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=8,
                        help="Sequence length S.  Smaller than KWT-1's 99 by "
                             "default to keep host-build smoke tests fast.")
    parser.add_argument("--num-blocks", type=int, default=1,
                        help="Number of stacked transformer encoder blocks. "
                             "1 = OneBlockKWT (Phase 0-8 validation unit); "
                             "12 = canonical KWT-1 encoder.")
    parser.add_argument("--full-kwt1", action="store_true",
                        help="Use the full KWT-1 model (input embed + pos enc + "
                             "12 encoder blocks + mean pool + classification "
                             "head).  Input shape becomes (1, --seq-len, --mfcc-dim); "
                             "output is (1, --num-classes).")
    parser.add_argument("--mfcc-dim", type=int, default=40,
                        help="MFCC feature dimension (only used with --full-kwt1).")
    parser.add_argument("--num-classes", type=int, default=35,
                        help="Classification output dimension (only used with --full-kwt1).")
    parser.add_argument("--save-ref", action="store_true",
                        help="Also save the python-reference int8 output for "
                             "bit-exact validation.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.full_kwt1:
        model = KWT1(
            mfcc_dim=args.mfcc_dim,
            seq_len=args.seq_len,
            num_blocks=args.num_blocks if args.num_blocks > 1 else 12,
            d=args.d_model,
            d_ff=args.d_ff,
            num_classes=args.num_classes,
        ).eval()
        torch.manual_seed(1)
        sample = (torch.randn(1, args.seq_len, args.mfcc_dim),)
    elif args.num_blocks == 1:
        model = OneBlockKWT(args.d_model, args.d_ff).eval()
        torch.manual_seed(1)
        sample = (torch.randn(1, args.seq_len, args.d_model),)
    else:
        model = KWT1Encoder(args.num_blocks, args.d_model, args.d_ff).eval()
        torch.manual_seed(1)
        sample = (torch.randn(1, args.seq_len, args.d_model),)
    prog = lower_to_cortex_m(model, sample)

    offsets, sizes, arena_bytes = _plan_memory(prog)
    layers, arena_bytes = extract_program(prog, offsets, sizes, arena_bytes)

    # I/O slot info from first/last layers.
    in_layer = layers[0]
    out_layer = layers[-1]
    input_elements = in_layer.num_elements   # float input size
    output_elements = out_layer.num_elements  # int8 output size

    emit_arena(args.out_dir, layers, arena_bytes, input_elements, output_elements)
    emit_weights(args.out_dir, layers)
    emit_params(args.out_dir, layers)
    emit_inference_body(args.out_dir, layers)
    emit_input_fixture(args.out_dir, sample[0])

    if args.save_ref:
        # The python-ref int8 output is everything just before the final
        # dequantize.  Find the dequantize's input.
        torch.manual_seed(7)
        test_input = torch.randn_like(sample[0])
        # Save fixture matching test_input for the runner to consume.
        emit_input_fixture(args.out_dir, test_input)
        # Run the lowered program through python.
        ref_out_fp = prog.module()(test_input)
        # The graph's final node is the dequantize, which produces float.
        # The standalone runner writes the int8 just before that.  Re-run
        # the lowered IR but stop one node short: easiest path is to look
        # up the dequantize's input via the IR.
        from executorch.exir.dialects._ops import ops as exir_ops
        deq_node = None
        for n in prog.graph_module.graph.nodes:
            if (n.op == "call_function"
                and n.target == exir_ops.edge.cortex_m.dequantize_per_tensor.default):
                deq_node = n
        if deq_node is None:
            print("  (no dequantize found; saving float output)")
            import torch as _t
            ref_q = _t.round(ref_out_fp.flatten() / out_layer.scale + out_layer.zero_point).clamp(-128, 127).to(torch.int8)
        else:
            # Re-run by interpreting the graph up to the dequantize input.
            # Cheaper hack: requantize ref_out_fp using the dequant's params.
            ref_q = (torch.round(ref_out_fp / out_layer.scale)
                     + out_layer.zero_point).clamp(-128, 127).to(torch.int8)
        ref_bytes = struct.pack(f"<{ref_q.numel()}b", *ref_q.flatten().tolist())
        (args.out_dir / "_ref_int8.bin").write_bytes(ref_bytes)

    print(f"Emitted to {args.out_dir}")
    print(f"  arena bytes: {arena_bytes}")
    print(f"  layers: {len(layers)}")
    print(f"  input elements: {input_elements}")
    print(f"  output elements: {output_elements}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
