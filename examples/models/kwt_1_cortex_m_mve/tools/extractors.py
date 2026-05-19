# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).
"""Per-op extractors that walk the post-CortexMPassManager FX graph
and produce structured Layer records for KWT-1.  Phased per
docs/PLAN.md; this module covers the ops needed for a single
transformer encoder block (Phase 6 surface).

Memory layout is a simple sequential allocation: each layer's output
tensor gets a fresh arena slot.  No greedy planner yet — TODO once
multi-block models land.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
from executorch.backends.transforms.utils import get_param_tensor
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram
from torch.fx import Node


# ---------------------------------------------------------------------------
# Tensor slot + layer dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TensorSlot:
    """One activation tensor placed in the arena."""
    name: str
    offset: int
    nbytes: int
    shape: tuple[int, ...]


@dataclass
class QuantInputLayer:
    kernel: str = "quantize_input"
    output: Optional[TensorSlot] = None
    scale: float = 0.0
    zero_point: int = 0
    qmin: int = -128
    qmax: int = 127
    num_elements: int = 0


@dataclass
class DequantOutputLayer:
    kernel: str = "dequantize_output"
    input: Optional[TensorSlot] = None
    scale: float = 0.0
    zero_point: int = 0
    num_elements: int = 0


@dataclass
class LayerNormLayer:
    kernel: str = "layer_norm_s8"
    input: Optional[TensorSlot] = None
    output: Optional[TensorSlot] = None
    num_rows: int = 0
    embed_dim: int = 0
    input_zp: int = 0
    input_scale: float = 0.0
    output_zp: int = 0
    output_scale: float = 0.0
    eps: float = 1e-5
    gamma: Optional[torch.Tensor] = None
    beta: Optional[torch.Tensor] = None


@dataclass
class GELULayer:
    kernel: str = "gelu_lut_s8"
    input: Optional[TensorSlot] = None
    output: Optional[TensorSlot] = None
    num_elements: int = 0
    lut: Optional[torch.Tensor] = None  # 256 int8


@dataclass
class LinearLayer:
    kernel: str = "linear_s8"
    input: Optional[TensorSlot] = None
    output: Optional[TensorSlot] = None
    num_rows: int = 0
    in_features: int = 0
    out_features: int = 0
    input_offset: int = 0
    filter_offset: int = 0
    output_offset: int = 0
    output_multiplier: int = 0
    output_shift: int = 0
    activation_min: int = -128
    activation_max: int = 127
    weight: Optional[torch.Tensor] = None  # int8 (N, K)
    kernel_sum: Optional[torch.Tensor] = None  # int32 (N,)


@dataclass
class AddLayer:
    kernel: str = "add_s8"
    self_in: Optional[TensorSlot] = None
    other_in: Optional[TensorSlot] = None
    output: Optional[TensorSlot] = None
    # When self_in or other_in is None, the corresponding *_const holds
    # the flash-resident int8 tensor that the kernel should read in
    # place of the arena slot.  Used for the positional-encoding add in
    # KWT-1, where one operand is a graph-signature BUFFER.
    self_const: Optional[torch.Tensor] = None
    other_const: Optional[torch.Tensor] = None
    num_elements: int = 0
    self_zp: int = 0
    self_multiplier: int = 0
    self_shift: int = 0
    other_zp: int = 0
    other_multiplier: int = 0
    other_shift: int = 0
    output_zp: int = 0
    output_multiplier: int = 0
    output_shift: int = 0
    activation_min: int = -128
    activation_max: int = 127


@dataclass
class FusedAttentionLayer:
    kernel: str = "attention_fused_s8"
    q_in: Optional[TensorSlot] = None
    k_in: Optional[TensorSlot] = None
    v_in: Optional[TensorSlot] = None
    output: Optional[TensorSlot] = None
    batch: int = 0
    seq_len: int = 0
    embed_dim: int = 0
    q_offset: int = 0
    k_offset: int = 0
    qk_output_zp: int = 0
    qk_output_multiplier: int = 0
    qk_output_shift: int = 0
    softmax_input_multiplier: int = 0
    softmax_input_shift: int = 0
    v_offset: int = 0
    av_output_zp: int = 0
    av_output_multiplier: int = 0
    av_output_shift: int = 0


@dataclass
class TransposeLayer:
    kernel: str = "transpose_s8"
    input: Optional[TensorSlot] = None
    output: Optional[TensorSlot] = None
    perm: tuple[int, ...] = ()
    # shape[r] is the size of axis r in the OUTPUT tensor;
    # in_stride[r] is the stride in the INPUT tensor (elements).
    shape: tuple[int, ...] = ()
    in_stride: tuple[int, ...] = ()


@dataclass
class MeanDimLayer:
    """int8 mean across one axis with fused dequant + requant.  Used
    for KWT-1's sequence-mean pooling (h.mean(dim=1) on (B, S, D));
    subsumes the dequant→aten.mean.dim→quant trio the lowering pass
    inserts around the float mean op."""
    kernel: str = "mean_dim_s8"
    input: Optional[TensorSlot] = None
    output: Optional[TensorSlot] = None
    outer: int = 0       # product of dims before the reduced one (B)
    reduce: int = 0      # size of the reduced dim (S)
    inner: int = 0       # product of dims after the reduced one (D)
    input_zp: int = 0
    input_scale: float = 0.0
    output_zp: int = 0
    output_scale: float = 0.0


Layer = Union[
    QuantInputLayer, DequantOutputLayer,
    LayerNormLayer, GELULayer, LinearLayer, AddLayer,
    FusedAttentionLayer, TransposeLayer, MeanDimLayer,
]


# ---------------------------------------------------------------------------
# Memory planning (simple sequential allocator for now)
# ---------------------------------------------------------------------------


def _node_shape(node: Node) -> tuple[int, ...]:
    return tuple(node.meta["val"].shape)


def _shape_nbytes(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


class _ArenaAllocator:
    """Arena slot lookup backed by exir.memory_planning.greedy results.

    The caller plans memory once (see _plan_memory in
    dump_kwt_1_artifacts.py), then constructs this with the planner's
    {node_name -> offset} / {node_name -> nbytes} dicts and the total
    arena bytes.  assign() just materializes a TensorSlot from those
    dicts; nodes outside the planner's output (placeholders, view-like
    ops that share storage with another node) get a None offset and
    are stitched up via _stitch_aliases() before extractors run.
    """
    def __init__(self, offsets: dict, sizes: dict, total: int) -> None:
        self.offsets = dict(offsets)
        self.sizes = dict(sizes)
        self._total = int(total)
        self.by_node: dict[str, TensorSlot] = {}

    def assign(self, node: Node) -> TensorSlot:
        if node.name in self.by_node:
            return self.by_node[node.name]
        shape = _node_shape(node)
        nbytes = self.sizes.get(node.name, _shape_nbytes(shape))
        offset = self.offsets.get(node.name)
        if offset is None:
            raise KeyError(
                f"node {node.name} has no planned mem_offset; either it's a "
                f"placeholder / view alias not captured by the planner, or "
                f"the memory plan didn't run."
            )
        slot = TensorSlot(name=node.name, offset=int(offset), nbytes=int(nbytes), shape=shape)
        self.by_node[node.name] = slot
        return slot

    @property
    def total(self) -> int:
        return self._total


# ---------------------------------------------------------------------------
# Per-op extractors
# ---------------------------------------------------------------------------


def _resolve_tensor(prog: ExportedProgram, arg) -> torch.Tensor:
    if hasattr(arg, "op"):
        return get_param_tensor(prog, arg)
    return arg


def _build_gelu_lut(in_zp: int, in_scale: float, out_zp: int, out_scale: float,
                    approximate_tanh: bool) -> torch.Tensor:
    xs = torch.arange(-128, 128, dtype=torch.int32).to(torch.int8)
    deq = (xs.to(torch.int32) - int(in_zp)).float() * float(in_scale)
    approx = "tanh" if approximate_tanh else "none"
    gelu = torch.nn.functional.gelu(deq, approximate=approx)
    q = (torch.round(gelu / float(out_scale)) + int(out_zp)).clamp(-128, 127)
    return q.to(torch.int8)


def _extract_layer_norm(node: Node, prog: ExportedProgram, alloc: _ArenaAllocator) -> LayerNormLayer:
    input_arg, in_zp, in_scale, w_arg, b_arg, eps, out_zp, out_scale = node.args
    gamma = _resolve_tensor(prog, w_arg).detach().float().contiguous()
    beta = _resolve_tensor(prog, b_arg).detach().float().contiguous() if b_arg is not None else None
    in_slot = alloc.assign(input_arg)
    out_slot = alloc.assign(node)
    embed_dim = int(gamma.numel())
    num_rows = _shape_nbytes(_node_shape(node)) // embed_dim
    return LayerNormLayer(
        input=in_slot, output=out_slot,
        num_rows=num_rows, embed_dim=embed_dim,
        input_zp=int(in_zp), input_scale=float(in_scale),
        output_zp=int(out_zp), output_scale=float(out_scale),
        eps=float(eps), gamma=gamma, beta=beta,
    )


def _extract_gelu(node: Node, prog: ExportedProgram, alloc: _ArenaAllocator) -> GELULayer:
    input_arg, in_zp, in_scale, out_zp, out_scale, approximate_tanh = node.args
    in_slot = alloc.assign(input_arg)
    out_slot = alloc.assign(node)
    lut = _build_gelu_lut(int(in_zp), float(in_scale),
                          int(out_zp), float(out_scale),
                          bool(approximate_tanh))
    return GELULayer(
        input=in_slot, output=out_slot,
        num_elements=_shape_nbytes(_node_shape(node)),
        lut=lut,
    )


def _extract_linear(node: Node, prog: ExportedProgram, alloc: _ArenaAllocator) -> LinearLayer:
    (in_arg, w_arg, b_arg, ks_arg, in_off, fil_off, out_off,
     mults, shifts, act_max, act_min) = node.args
    w_t = _resolve_tensor(prog, w_arg).detach().to(torch.int8).contiguous()
    if ks_arg is not None:
        ks_t = _resolve_tensor(prog, ks_arg).detach().to(torch.int32).flatten().contiguous()
    else:
        ks_t = torch.zeros(w_t.shape[0], dtype=torch.int32)
    in_slot = alloc.assign(in_arg)
    out_slot = alloc.assign(node)
    K = w_t.shape[1]
    N = w_t.shape[0]
    num_rows = _shape_nbytes(_node_shape(node)) // N
    return LinearLayer(
        input=in_slot, output=out_slot,
        num_rows=num_rows, in_features=K, out_features=N,
        input_offset=int(in_off), filter_offset=int(fil_off),
        output_offset=int(out_off),
        output_multiplier=int(mults[0]), output_shift=int(shifts[0]),
        activation_min=int(act_min), activation_max=int(act_max),
        weight=w_t, kernel_sum=ks_t,
    )


def _slot_or_const(node: Node, prog: ExportedProgram, alloc: _ArenaAllocator):
    """For an Add input: if the node is a placeholder/buffer, return
    (None, tensor) so the emitter can stamp it as a flash const; if it's
    a regular activation, return (slot, None) and let the planner own
    the arena slot."""
    if node.op == "placeholder":
        return None, _resolve_tensor(prog, node).detach().contiguous()
    return alloc.assign(node), None


def _extract_add(node: Node, prog: ExportedProgram, alloc: _ArenaAllocator) -> AddLayer:
    (a_node, s_zp, s_m, s_s, b_node, o_zp, o_m, o_s,
     out_zp, out_m, out_s, amin, amax) = node.args
    a_slot, a_const = _slot_or_const(a_node, prog, alloc)
    b_slot, b_const = _slot_or_const(b_node, prog, alloc)
    out_slot = alloc.assign(node)
    return AddLayer(
        self_in=a_slot, other_in=b_slot, output=out_slot,
        self_const=a_const, other_const=b_const,
        num_elements=_shape_nbytes(_node_shape(node)),
        self_zp=int(s_zp), self_multiplier=int(s_m), self_shift=int(s_s),
        other_zp=int(o_zp), other_multiplier=int(o_m), other_shift=int(o_s),
        output_zp=int(out_zp), output_multiplier=int(out_m), output_shift=int(out_s),
        activation_min=int(amin), activation_max=int(amax),
    )


def _extract_fused_attention(node: Node, prog: ExportedProgram, alloc: _ArenaAllocator) -> FusedAttentionLayer:
    (q_node, k_node, v_node, q_off, k_off,
     qk_zp, qk_mult, qk_shift, sm_mult, sm_shift, sm_diff_min,
     v_off, av_zp, av_mult, av_shift) = node.args
    q_slot = alloc.assign(q_node); k_slot = alloc.assign(k_node); v_slot = alloc.assign(v_node)
    out_slot = alloc.assign(node)
    B, S, D = _node_shape(q_node)
    return FusedAttentionLayer(
        q_in=q_slot, k_in=k_slot, v_in=v_slot, output=out_slot,
        batch=int(B), seq_len=int(S), embed_dim=int(D),
        q_offset=int(q_off), k_offset=int(k_off),
        qk_output_zp=int(qk_zp), qk_output_multiplier=int(qk_mult), qk_output_shift=int(qk_shift),
        softmax_input_multiplier=int(sm_mult), softmax_input_shift=int(sm_shift),
        v_offset=int(v_off),
        av_output_zp=int(av_zp), av_output_multiplier=int(av_mult), av_output_shift=int(av_shift),
    )


def _extract_transpose(node: Node, prog: ExportedProgram, alloc: _ArenaAllocator) -> TransposeLayer:
    in_arg, perm = node.args
    in_slot = alloc.assign(in_arg); out_slot = alloc.assign(node)
    in_shape = _node_shape(in_arg)
    out_shape = _node_shape(node)
    # in_stride[r] = stride of input axis = perm[r] in the input's row-major layout.
    in_row_major_strides = []
    acc = 1
    for d in reversed(in_shape):
        in_row_major_strides.append(acc); acc *= d
    in_row_major_strides = list(reversed(in_row_major_strides))
    in_stride = tuple(in_row_major_strides[int(perm[r])] for r in range(len(perm)))
    return TransposeLayer(
        input=in_slot, output=out_slot,
        perm=tuple(int(x) for x in perm),
        shape=tuple(int(x) for x in out_shape),
        in_stride=in_stride,
    )


def _extract_quantize_input(node: Node, alloc: _ArenaAllocator) -> QuantInputLayer:
    # quantize_per_tensor args: (input, scale, zp, qmin, qmax, dtype)
    _, scale, zp, qmin, qmax, _ = node.args
    out_slot = alloc.assign(node)
    return QuantInputLayer(
        output=out_slot, scale=float(scale), zero_point=int(zp),
        qmin=int(qmin), qmax=int(qmax),
        num_elements=_shape_nbytes(_node_shape(node)),
    )


def _extract_dequantize_output(node: Node, alloc: _ArenaAllocator) -> DequantOutputLayer:
    in_arg, scale, zp, _qmin, _qmax, _dtype = node.args
    in_slot = alloc.assign(in_arg)
    return DequantOutputLayer(
        input=in_slot, scale=float(scale), zero_point=int(zp),
        num_elements=_shape_nbytes(_node_shape(node)),
    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


_EXTRACTORS = {
    "cortex_m.quantized_layer_norm": _extract_layer_norm,
    "cortex_m.quantized_gelu": _extract_gelu,
    "cortex_m.quantized_linear": _extract_linear,
    "cortex_m.quantized_add": _extract_add,
    "cortex_m.quantized_fused_attention": _extract_fused_attention,
    "cortex_m.transpose": _extract_transpose,
}


def _op_key(node: Node) -> str:
    t = node.target
    if hasattr(t, "_schema"):
        return f"{t.namespace}.{t._schema.name.split('::')[-1]}"
    return str(t)


def _find_mean_patterns(gm) -> tuple[set, dict]:
    """Scan for dequant → aten.mean.dim → quant chains.  For each, return
    (absorbed_nodes, {mean_node: (dequant_node, quant_node)}).  Other
    dequant / quant nodes stay as model-boundary ops.

    _op_key drops overload names (returns 'aten.mean' for both
    'mean.dim' and 'mean.default') so we filter on the EdgeOpOverload
    string representation instead."""
    absorbed = set()
    patterns = {}
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if "aten.mean.dim" not in str(node.target):
            continue
        deq = node.args[0]
        if not (hasattr(deq, "op") and deq.op == "call_function"
                and _op_key(deq) == "cortex_m.dequantize_per_tensor"):
            continue
        users = list(node.users)
        if len(users) != 1:
            continue
        q = users[0]
        if not (hasattr(q, "op") and q.op == "call_function"
                and _op_key(q) == "cortex_m.quantize_per_tensor"):
            continue
        patterns[node] = (deq, q)
        absorbed.add(deq)
        absorbed.add(q)
    return absorbed, patterns


def _extract_mean_dim(node, deq, q, prog, alloc):
    """Build a MeanDimLayer from a dequant → mean(dim=[r]) → quant trio.
    The quant is the layer's output slot; the dequant's input is the
    int8 source.  Caller is responsible for assigning offsets via alloc."""
    deq_in, in_scale, in_zp, _, _, _ = deq.args
    _, out_scale, out_zp, _, _, _ = q.args
    mean_args = node.args
    if len(mean_args) < 2:
        raise ValueError(f"unexpected aten.mean.dim args: {mean_args}")
    dim_arg = mean_args[1]
    if not isinstance(dim_arg, (list, tuple)) or len(dim_arg) != 1:
        raise NotImplementedError(
            f"mean over multiple/no dims not supported: dim={dim_arg}")
    dim = int(dim_arg[0])
    in_shape = _node_shape(deq_in)
    if dim < 0:
        dim += len(in_shape)
    outer = 1
    for d in in_shape[:dim]:
        outer *= int(d)
    reduce = int(in_shape[dim])
    inner = 1
    for d in in_shape[dim + 1:]:
        inner *= int(d)
    in_slot = alloc.assign(deq_in)
    out_slot = alloc.assign(q)
    return MeanDimLayer(
        input=in_slot, output=out_slot,
        outer=outer, reduce=reduce, inner=inner,
        input_zp=int(in_zp), input_scale=float(in_scale),
        output_zp=int(out_zp), output_scale=float(out_scale),
    )


def extract_program(
    prog: ExportedProgram,
    offsets: dict,
    sizes: dict,
    arena_bytes: int,
) -> tuple[list[Layer], int]:
    """Walk topologically and produce (layers, arena_bytes).

    `offsets`, `sizes`, `arena_bytes` come from exir.memory_planning.greedy
    (see dump_kwt_1_artifacts._plan_memory).  Every cortex_m call-function
    node referenced by the schedule must have an entry in `offsets`.
    """
    alloc = _ArenaAllocator(offsets, sizes, arena_bytes)
    layers: list[Layer] = []
    absorbed, mean_patterns = _find_mean_patterns(prog.graph_module)
    for node in prog.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        if node in absorbed:
            continue
        target = _op_key(node)
        if "aten.mean.dim" in str(node.target) and node in mean_patterns:
            deq, q = mean_patterns[node]
            layers.append(_extract_mean_dim(node, deq, q, prog, alloc))
            continue
        if target == "cortex_m.quantize_per_tensor":
            layers.append(_extract_quantize_input(node, alloc))
        elif target == "cortex_m.dequantize_per_tensor":
            layers.append(_extract_dequantize_output(node, alloc))
        elif target in _EXTRACTORS:
            ex = _EXTRACTORS[target]
            if ex in (_extract_quantize_input, _extract_dequantize_output):
                layers.append(ex(node, alloc))
            elif ex is _extract_transpose:
                layers.append(ex(node, prog, alloc))
            else:
                layers.append(ex(node, prog, alloc))
        elif target == "aten.permute_copy":
            # The convert_to_cortex_m_pass inserts back-to-back
            # permute_copy + cortex_m.transpose to materialize BMM
            # rhs_transposed.  We treat both as a single transpose.
            in_arg, perm = node.args
            in_slot = alloc.assign(in_arg); out_slot = alloc.assign(node)
            in_shape = _node_shape(in_arg)
            in_rms = []
            acc = 1
            for d in reversed(in_shape):
                in_rms.append(acc); acc *= d
            in_rms = list(reversed(in_rms))
            in_stride = tuple(in_rms[int(perm[r])] for r in range(len(perm)))
            layers.append(TransposeLayer(
                input=in_slot, output=out_slot,
                perm=tuple(int(x) for x in perm),
                shape=tuple(int(x) for x in _node_shape(node)),
                in_stride=in_stride,
            ))
        # else: skip ops we don't handle (view_copy, etc.)
    return layers, alloc.total
