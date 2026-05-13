# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).
"""
Per-op extractors that pull weights, biases, and quantization parameters out of
the post-CortexMPassManager fx graph into structured layer records.

Phase A supports: quantize_per_tensor, dequantize_per_tensor, quantized_linear.
Phase B/C/D add: conv2d variants, depthwise, add, avgpool.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from executorch.backends.transforms.utils import get_param_tensor
from torch.export import ExportedProgram
from torch.fx import Node


@dataclass
class TensorSlot:
    """One activation tensor in the planned arena."""
    name: str
    offset: int
    size: int
    shape: tuple[int, ...]


@dataclass
class QuantInputLayer:
    kernel: str = "quantize_input"
    input_name: str = ""
    output: TensorSlot = None
    scale: float = 0.0
    zero_point: int = 0
    qmin: int = -128
    qmax: int = 127


@dataclass
class DequantOutputLayer:
    kernel: str = "dequantize_output"
    input: TensorSlot = None
    output_name: str = ""
    scale: float = 0.0
    zero_point: int = 0


@dataclass
class MemcpyLayer:
    """Buffer copy with no data transformation: view_copy, identity pad, etc."""
    kernel: str = "memcpy"
    input: TensorSlot = None
    output: TensorSlot = None
    nbytes: int = 0


@dataclass
class Conv2dLayer:
    kernel: str = "conv2d_s8"
    input: TensorSlot = None
    output: TensorSlot = None
    weight: torch.Tensor = None       # int8 [out_C, kH, kW, in_C]  (OHWI)
    bias: Optional[torch.Tensor] = None  # int32 [out_C]
    requantize_multipliers: torch.Tensor = None  # int32 [out_C]
    requantize_shifts: torch.Tensor = None       # int32 [out_C]
    stride: tuple = (1, 1)
    padding: tuple = (0, 0)
    dilation: tuple = (1, 1)
    input_offset: int = 0
    output_offset: int = 0
    activation_min: int = -128
    activation_max: int = 127
    # Optional packed-32 weight blob for the first 3x3 stride-2 in_c=3 conv.
    # See Conv2dParams.weight_packed_32 in mv2_layer_params.h for the layout.
    weight_packed_32: Optional[torch.Tensor] = None  # int8 [out_C, 32]


@dataclass
class DepthwiseConv2dLayer:
    kernel: str = "dwconv2d_s8"
    input: TensorSlot = None
    output: TensorSlot = None
    weight: torch.Tensor = None       # int8 [1, kH, kW, C]  (IHWO, depth_mul=1)
    bias: Optional[torch.Tensor] = None  # int32 [C]
    requantize_multipliers: torch.Tensor = None
    requantize_shifts: torch.Tensor = None
    stride: tuple = (1, 1)
    padding: tuple = (0, 0)
    dilation: tuple = (1, 1)
    depth_multiplier: int = 1
    input_offset: int = 0
    output_offset: int = 0
    activation_min: int = -128
    activation_max: int = 127
    # Optional bias-with-offset: bias + input_offset * sum(weight[c]) over
    # all kernel positions.  Used by the fast 4-pixel tile for pixels where
    # all kH×kW kernel taps are valid (i.e., away from the padding ring).
    # Lets the runtime skip the per-tap vaddq_s32(x, v_in_off) for those
    # pixels; the boundary path keeps the original bias and runtime offset.
    bias_with_offset_full: Optional[torch.Tensor] = None  # int32 [C]


@dataclass
class QuantizedAddLayer:
    kernel: str = "add_s8"
    self_in: TensorSlot = None
    other_in: TensorSlot = None
    output: TensorSlot = None
    self_zero_point: int = 0
    self_multiplier: int = 0
    self_shift: int = 0
    other_zero_point: int = 0
    other_multiplier: int = 0
    other_shift: int = 0
    output_zero_point: int = 0
    output_multiplier: int = 0
    output_shift: int = 0
    activation_min: int = -128
    activation_max: int = 127


@dataclass
class AvgPool2dLayer:
    kernel: str = "avgpool_s8"
    input: TensorSlot = None
    output: TensorSlot = None
    kernel_size: tuple = (1, 1)
    stride: tuple = (1, 1)
    padding: tuple = (0, 0)
    zero_point: int = 0
    multiplier: int = 0
    shift: int = 0


@dataclass
class LinearLayer:
    kernel: str = "gemv_s8"
    input: TensorSlot = None
    output: TensorSlot = None
    weight: torch.Tensor = None       # int8 [out_features, in_features]
    bias: Optional[torch.Tensor] = None  # int32 [out_features]
    kernel_sum: Optional[torch.Tensor] = None  # int32 [out_features]
    input_offset: int = 0
    filter_offset: int = 0
    output_offset: int = 0
    multiplier: int = 0
    shift: int = 0
    activation_min: int = -128
    activation_max: int = 127


@dataclass
class ProgramSchedule:
    layers: list = field(default_factory=list)
    arena_bytes: int = 0
    input_slot: Optional[TensorSlot] = None
    output_slot: Optional[TensorSlot] = None
    # Final dequant scale/zp (for the test side to convert int8 logits to float).
    output_scale: float = 1.0
    output_zero_point: int = 0


def _resolve_tensor(program: ExportedProgram, node: Node) -> torch.Tensor:
    """Resolve a parameter/buffer/lifted-constant node to its concrete tensor."""
    tensor = get_param_tensor(program, node)
    if tensor is None:
        raise ValueError(f"Cannot resolve tensor from node {node} (op={node.op}, target={node.target})")
    return tensor


def _slot_from_node(node: Node, offsets: dict, sizes: dict) -> TensorSlot:
    """Build a TensorSlot from a node using its TensorSpec metadata."""
    spec = node.meta["spec"]
    shape = tuple(spec.shape)
    return TensorSlot(
        name=str(node.name),
        offset=offsets[node.name],
        size=sizes[node.name],
        shape=shape,
    )


def _node_args_map(node: Node, names: list[str]) -> dict:
    """Map positional args to named slots, falling back to kwargs."""
    out = {}
    for i, name in enumerate(names):
        if i < len(node.args):
            out[name] = node.args[i]
        elif name in node.kwargs:
            out[name] = node.kwargs[name]
        else:
            out[name] = None
    return out


def extract_quantize_per_tensor(node: Node, program: ExportedProgram, offsets, sizes) -> QuantInputLayer:
    # Schema: (Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype)
    a = _node_args_map(node, [
        "input", "scale", "zero_point", "quant_min", "quant_max", "dtype",
    ])
    input_node = a["input"]
    return QuantInputLayer(
        input_name=str(input_node.name),
        output=_slot_from_node(node, offsets, sizes),
        scale=float(a["scale"]),
        zero_point=int(a["zero_point"]),
        qmin=int(a["quant_min"]),
        qmax=int(a["quant_max"]),
    )


def extract_dequantize_per_tensor(node: Node, program: ExportedProgram, offsets, sizes) -> DequantOutputLayer:
    a = _node_args_map(node, [
        "input", "scale", "zero_point", "quant_min", "quant_max", "dtype",
    ])
    input_node = a["input"]
    return DequantOutputLayer(
        input=_slot_from_node(input_node, offsets, sizes),
        output_name=str(node.name),
        scale=float(a["scale"]),
        zero_point=int(a["zero_point"]),
    )


def extract_quantized_linear(node: Node, program: ExportedProgram, offsets, sizes) -> LinearLayer:
    # Schema mirrors operators.py:376
    a = _node_args_map(node, [
        "input", "weights", "bias", "kernel_sum",
        "input_offset", "filter_offset", "output_offset",
        "requantize_multipliers", "requantize_shifts",
        "activation_max", "activation_min",
    ])
    weight_t = _resolve_tensor(program, a["weights"])
    bias_t = _resolve_tensor(program, a["bias"]) if a["bias"] is not None else None
    kernel_sum_t = _resolve_tensor(program, a["kernel_sum"]) if a["kernel_sum"] is not None else None
    mults = a["requantize_multipliers"]
    shifts = a["requantize_shifts"]
    # multipliers/shifts come through as Python lists or get_attr; flatten to int.
    mult = int(mults[0]) if isinstance(mults, (list, tuple)) else int(mults)
    shift = int(shifts[0]) if isinstance(shifts, (list, tuple)) else int(shifts)
    return LinearLayer(
        input=_slot_from_node(a["input"], offsets, sizes),
        output=_slot_from_node(node, offsets, sizes),
        weight=weight_t.detach().to(torch.int8).contiguous(),
        bias=bias_t.detach().to(torch.int32).flatten().contiguous() if bias_t is not None else None,
        kernel_sum=kernel_sum_t.detach().to(torch.int32).flatten().contiguous() if kernel_sum_t is not None else None,
        input_offset=int(a["input_offset"]),
        filter_offset=int(a["filter_offset"]),
        output_offset=int(a["output_offset"]),
        multiplier=mult,
        shift=shift,
        activation_min=int(a["activation_min"]),
        activation_max=int(a["activation_max"]),
    )


def _coerce_int_pair(v) -> tuple:
    if isinstance(v, (list, tuple)):
        if len(v) == 1:
            return (int(v[0]), int(v[0]))
        return (int(v[0]), int(v[1]))
    return (int(v), int(v))


def extract_quantized_conv2d(node: Node, program: ExportedProgram, offsets, sizes) -> Conv2dLayer:
    a = _node_args_map(node, [
        "input", "weight", "bias",
        "stride", "padding", "dilation",
        "input_offset", "output_offset",
        "requantize_multipliers", "requantize_shifts",
        "activation_min", "activation_max",
    ])
    weight_t = _resolve_tensor(program, a["weight"])
    bias_t = _resolve_tensor(program, a["bias"]) if a["bias"] is not None else None
    mults_t = _resolve_tensor(program, a["requantize_multipliers"])
    shifts_t = _resolve_tensor(program, a["requantize_shifts"])
    # When safe (1x1 conv with no padding, or any conv with padding=0 *and* an
    # output region fully inside the input — which 1x1 always satisfies), fold
    # the input_offset contribution into a per-channel bias once AOT.
    #
    # Math: cortex_m conv2d has no filter_offset (weights are symmetric with
    # zero_point == 0), so
    #   acc[oc] = sum_kh_kw_ic((x + input_offset) * w[oc][...])
    #          = sum(x * w[oc]) + input_offset * sum_kh_kw_ic(w[oc])
    # The second term is a per-OC constant when all kernel taps contribute
    # uniformly across output pixels.  For 1x1 convs that holds trivially.
    # For non-1x1 convs with padding, edge output pixels use only a subset of
    # the kernel taps, so the second term varies per pixel — we keep
    # input_offset as a runtime arg in that case.
    weight_int32 = weight_t.detach().to(torch.int32).contiguous()
    out_c = weight_int32.shape[0]
    kernel_h, kernel_w = weight_int32.shape[1], weight_int32.shape[2]  # OHWI
    in_c = weight_int32.shape[3]
    pad_h, pad_w = _coerce_int_pair(a["padding"])
    stride_h, stride_w = _coerce_int_pair(a["stride"])

    # 3x3 stride-2 pad-1 in_c=3 special case (the MV2 first conv).  The
    # 16-wide vmladavaq_s8 inner loop can't trigger on in_c=3 alone, but we
    # can pack each OC's 27 weights + 5 zero pads into a 32-byte vector and
    # use 2 vmladavaq_s8 per output channel.  Boundary pixels are handled by
    # putting -input_offset in the cropped patch positions so the AOT
    # offset-fold still produces the correct result (the zero-padded weight
    # positions contribute nothing regardless).
    pack_first_3x3 = (
        kernel_h == 3 and kernel_w == 3 and in_c == 3
        and stride_h == 2 and stride_w == 2
        and pad_h == 1 and pad_w == 1
    )

    safe_to_fold_1x1 = (
        kernel_h == 1 and kernel_w == 1 and pad_h == 0 and pad_w == 0
    )

    weight_packed_32 = None
    if pack_first_3x3:
        # OHWI flatten = (kh, kw, ic) order = exactly the 27-element packing.
        flat = weight_int32.reshape(out_c, kernel_h * kernel_w * in_c)  # [out_c, 27]
        pad = torch.zeros((out_c, 32 - flat.shape[1]), dtype=torch.int32)
        packed = torch.cat([flat, pad], dim=1)
        weight_packed_32 = packed.to(torch.int8).contiguous()

    if safe_to_fold_1x1 or pack_first_3x3:
        sum_w_per_oc = weight_int32.flatten(1).sum(dim=1)  # [out_c]
        offset_term = sum_w_per_oc * int(a["input_offset"])
        if bias_t is not None:
            bias_with_offset = bias_t.detach().to(torch.int32).flatten() + offset_term
        else:
            bias_with_offset = offset_term
        bias_with_offset = bias_with_offset.to(torch.int32).contiguous()
        emitted_bias = bias_with_offset
        emitted_input_offset = 0
    else:
        emitted_bias = (
            bias_t.detach().to(torch.int32).flatten().contiguous()
            if bias_t is not None else None
        )
        emitted_input_offset = int(a["input_offset"])

    # For the packed first conv, the runtime needs the original input_offset
    # to put -input_offset in cropped patch positions.  Stash it in the
    # padding=p->pad_h fields?  No: we just keep input_offset != 0 alongside
    # the packed marker.  Override emitted_input_offset to keep the offset
    # available to the kernel while still signalling "offset is folded into
    # bias_with_offset".  We use the packed pointer non-null as the signal,
    # not input_offset == 0.
    if pack_first_3x3:
        emitted_input_offset = int(a["input_offset"])

    return Conv2dLayer(
        input=_slot_from_node(a["input"], offsets, sizes),
        output=_slot_from_node(node, offsets, sizes),
        weight=weight_t.detach().to(torch.int8).contiguous(),
        bias=emitted_bias,
        requantize_multipliers=mults_t.detach().to(torch.int32).flatten().contiguous(),
        # frexp exponent — always in roughly [-30, +5], packs into int8.
        requantize_shifts=shifts_t.detach().to(torch.int8).flatten().contiguous(),
        stride=_coerce_int_pair(a["stride"]),
        padding=_coerce_int_pair(a["padding"]),
        dilation=_coerce_int_pair(a["dilation"]),
        # input_offset == 0 signals to the 1x1-fast-path kernel that the
        # offset is already folded into the bias.  For the packed first-conv
        # path, input_offset stays non-zero and the kernel reads it to fill
        # cropped patch positions; the packed pointer is the signal.
        input_offset=emitted_input_offset,
        output_offset=int(a["output_offset"]),
        activation_min=int(a["activation_min"]),
        activation_max=int(a["activation_max"]),
        weight_packed_32=weight_packed_32,
    )


def extract_quantized_depthwise_conv2d(node: Node, program: ExportedProgram, offsets, sizes) -> DepthwiseConv2dLayer:
    a = _node_args_map(node, [
        "input", "weight", "bias",
        "stride", "padding", "dilation",
        "depth_multiplier",
        "input_offset", "output_offset",
        "requantize_multipliers", "requantize_shifts",
        "activation_min", "activation_max",
    ])
    weight_t = _resolve_tensor(program, a["weight"])
    bias_t = _resolve_tensor(program, a["bias"]) if a["bias"] is not None else None
    mults_t = _resolve_tensor(program, a["requantize_multipliers"])
    shifts_t = _resolve_tensor(program, a["requantize_shifts"])
    # Pre-compute bias_with_offset_full for the fast 4-pixel tile.  Layout
    # is IHWO [1, kH, kW, C]; sum over (kH, kW) axes gives per-channel total.
    bias_with_offset_full = None
    if bias_t is not None and int(a["input_offset"]) != 0:
        w_int32 = weight_t.detach().to(torch.int32)
        sum_w_per_c = w_int32.sum(dim=(0, 1, 2))  # [C]
        offset_term = sum_w_per_c * int(a["input_offset"])
        bias_with_offset_full = (
            bias_t.detach().to(torch.int32).flatten() + offset_term
        ).to(torch.int32).contiguous()
    return DepthwiseConv2dLayer(
        input=_slot_from_node(a["input"], offsets, sizes),
        output=_slot_from_node(node, offsets, sizes),
        weight=weight_t.detach().to(torch.int8).contiguous(),
        bias=bias_t.detach().to(torch.int32).flatten().contiguous() if bias_t is not None else None,
        bias_with_offset_full=bias_with_offset_full,
        requantize_multipliers=mults_t.detach().to(torch.int32).flatten().contiguous(),
        requantize_shifts=shifts_t.detach().to(torch.int8).flatten().contiguous(),
        stride=_coerce_int_pair(a["stride"]),
        padding=_coerce_int_pair(a["padding"]),
        dilation=_coerce_int_pair(a["dilation"]),
        depth_multiplier=int(a["depth_multiplier"]),
        input_offset=int(a["input_offset"]),
        output_offset=int(a["output_offset"]),
        activation_min=int(a["activation_min"]),
        activation_max=int(a["activation_max"]),
    )


def extract_quantized_add(node: Node, program: ExportedProgram, offsets, sizes) -> QuantizedAddLayer:
    a = _node_args_map(node, [
        "self", "self_zero_point", "self_multiplier", "self_shift",
        "other", "other_zero_point", "other_multiplier", "other_shift",
        "output_zero_point", "output_multiplier", "output_shift",
        "activation_min", "activation_max",
    ])
    return QuantizedAddLayer(
        self_in=_slot_from_node(a["self"], offsets, sizes),
        other_in=_slot_from_node(a["other"], offsets, sizes),
        output=_slot_from_node(node, offsets, sizes),
        self_zero_point=int(a["self_zero_point"]),
        self_multiplier=int(a["self_multiplier"]),
        self_shift=int(a["self_shift"]),
        other_zero_point=int(a["other_zero_point"]),
        other_multiplier=int(a["other_multiplier"]),
        other_shift=int(a["other_shift"]),
        output_zero_point=int(a["output_zero_point"]),
        output_multiplier=int(a["output_multiplier"]),
        output_shift=int(a["output_shift"]),
        activation_min=int(a["activation_min"]),
        activation_max=int(a["activation_max"]),
    )


def extract_quantized_avg_pool2d(node: Node, program: ExportedProgram, offsets, sizes) -> AvgPool2dLayer:
    a = _node_args_map(node, [
        "input", "kernel_size", "stride", "padding",
        "zero_point", "multiplier", "shift",
    ])
    return AvgPool2dLayer(
        input=_slot_from_node(a["input"], offsets, sizes),
        output=_slot_from_node(node, offsets, sizes),
        kernel_size=_coerce_int_pair(a["kernel_size"]),
        stride=_coerce_int_pair(a["stride"]),
        padding=_coerce_int_pair(a["padding"]),
        zero_point=int(a["zero_point"]),
        multiplier=int(a["multiplier"]),
        shift=int(a["shift"]),
    )


def _extract_memcpy(node: Node, program: ExportedProgram, offsets, sizes) -> MemcpyLayer:
    """Used for view_copy / identity pad / dim-order clone — the input data
    and output data are the same bytes; only the planned offset differs."""
    src_node = node.args[0]
    in_slot = _slot_from_node(src_node, offsets, sizes)
    out_slot = _slot_from_node(node, offsets, sizes)
    return MemcpyLayer(input=in_slot, output=out_slot, nbytes=out_slot.size)


def extract_cortex_m_pad(node: Node, program: ExportedProgram, offsets, sizes) -> MemcpyLayer:
    """cortex_m.pad with all-zero pre/post pads is a no-op rewrite the
    cortex_m passes insert.  We assert the no-op shape here; if non-zero
    pads ever appear the codegen will need a real pad kernel."""
    a = _node_args_map(node, ["input", "pre_pad", "post_pad", "pad_value"])
    pre = list(a["pre_pad"]) if a["pre_pad"] is not None else []
    post = list(a["post_pad"]) if a["post_pad"] is not None else []
    if any(int(v) for v in pre) or any(int(v) for v in post):
        raise NotImplementedError(
            f"cortex_m.pad with non-zero pads not supported yet: pre={pre} post={post}"
        )
    return _extract_memcpy(node, program, offsets, sizes)


EXTRACTORS = {
    "cortex_m.quantize_per_tensor.default": extract_quantize_per_tensor,
    "cortex_m.dequantize_per_tensor.default": extract_dequantize_per_tensor,
    "cortex_m.quantized_linear.default": extract_quantized_linear,
    "cortex_m.quantized_conv2d.default": extract_quantized_conv2d,
    "cortex_m.quantized_avg_pool2d.default": extract_quantized_avg_pool2d,
    "cortex_m.quantized_depthwise_conv2d.default": extract_quantized_depthwise_conv2d,
    "cortex_m.quantized_add.default": extract_quantized_add,
    "cortex_m.pad.default": extract_cortex_m_pad,
    "aten.view_copy.default": _extract_memcpy,
    "dim_order_ops._clone_dim_order.default": _extract_memcpy,
}

NOOP_OPS: set[str] = set()  # kept for back-compat with dump_mv2_artifacts.py


def op_target_string(node: Node) -> str:
    """Stable string identifier for a call_function op."""
    t = node.target
    if hasattr(t, "_schema"):
        # OpOverload — render as 'namespace.opname.overload'
        ns = t.namespace
        op = t._schema.name.split("::")[-1]
        overload = t._schema.overload_name or "default"
        return f"{ns}.{op}.{overload}"
    return str(t)
