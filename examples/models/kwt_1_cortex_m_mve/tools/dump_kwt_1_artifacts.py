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
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
import executorch.backends.cortex_m.ops.operators  # noqa

from .extractors import extract_program
from .emitters import emit_arena, emit_inference_body, emit_params, emit_weights


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
    parser.add_argument("--save-ref", action="store_true",
                        help="Also save the python-reference int8 output for "
                             "bit-exact validation.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = OneBlockKWT(args.d_model, args.d_ff).eval()
    torch.manual_seed(1)
    sample = (torch.randn(1, args.seq_len, args.d_model),)
    prog = lower_to_cortex_m(model, sample)

    layers, arena_bytes = extract_program(prog)

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
