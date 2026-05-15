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
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
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
class FusedConv2dDwconv2dLayer:
    """MV2 expand+dwconv fusion: a 1x1 conv feeding a 3x3 depthwise conv.

    Both layers' params travel together in a single Layer record so the
    emitter produces one FusedConv2dDwconv2dParams struct and one
    mv2_conv2d_dwconv2d_fused_s8 call.  The runtime kernel streams the
    expand output through a 3-row rolling buffer that the dwconv consumes
    immediately, avoiding materialization of the full HxWxC_expand tensor.
    """
    kernel: str = "conv2d_dwconv2d_fused_s8"
    input: TensorSlot = None
    output: TensorSlot = None
    # Expand conv1x1 (OHWI [expand_out_c, 1, 1, in_c])
    expand_weight: torch.Tensor = None
    expand_bias: Optional[torch.Tensor] = None  # int32 [expand_out_c], offset-folded
    expand_requantize_multipliers: torch.Tensor = None
    expand_requantize_shifts: torch.Tensor = None
    expand_input_offset: int = 0   # typically 0 (folded into expand_bias)
    expand_output_offset: int = 0
    expand_activation_min: int = -128
    expand_activation_max: int = 127
    # Dwconv 3x3 (IHWO [1, kH, kW, expand_out_c])
    dw_weight: torch.Tensor = None
    dw_bias: Optional[torch.Tensor] = None  # int32 [expand_out_c]
    dw_bias_with_offset_full: Optional[torch.Tensor] = None  # int32 [expand_out_c]
    dw_requantize_multipliers: torch.Tensor = None
    dw_requantize_shifts: torch.Tensor = None
    dw_stride: tuple = (1, 1)
    dw_padding: tuple = (1, 1)
    dw_input_offset: int = 0
    dw_output_offset: int = 0
    dw_activation_min: int = -128
    dw_activation_max: int = 127


@dataclass
class FusedInvertedResidualLayer:
    """Full MV2 inverted-residual block fused into one runtime call:
    expand 1x1 + dwconv 3x3 + project 1x1 [+ optional residual add].

    The runtime kernel streams two rolling buffers — one for expand output
    rows feeding the dwconv (3 rows of in_w x expand_out_c), one for the
    dwconv output row feeding the project conv (1 row of out_w x
    expand_out_c).  When residual is present, the block input is read a
    second time during the per-row add epilogue (the planner keeps it in
    its same arena slot throughout this op).
    """
    kernel: str = "inverted_residual_fused_s8"
    input: TensorSlot = None
    output: TensorSlot = None
    # Expand 1x1
    expand_weight: torch.Tensor = None  # OHWI [expand_out_c, 1, 1, in_c]
    expand_bias: Optional[torch.Tensor] = None
    expand_requantize_multipliers: torch.Tensor = None
    expand_requantize_shifts: torch.Tensor = None
    expand_input_offset: int = 0
    expand_output_offset: int = 0
    expand_activation_min: int = -128
    expand_activation_max: int = 127
    # Dwconv 3x3
    dw_weight: torch.Tensor = None  # IHWO [1, kH, kW, expand_out_c]
    dw_bias: Optional[torch.Tensor] = None
    dw_bias_with_offset_full: Optional[torch.Tensor] = None
    dw_requantize_multipliers: torch.Tensor = None
    dw_requantize_shifts: torch.Tensor = None
    dw_stride: tuple = (1, 1)
    dw_padding: tuple = (1, 1)
    dw_input_offset: int = 0
    dw_output_offset: int = 0
    dw_activation_min: int = -128
    dw_activation_max: int = 127
    # Project 1x1
    project_weight: torch.Tensor = None  # OHWI [project_out_c, 1, 1, expand_out_c]
    project_bias: Optional[torch.Tensor] = None  # offset-folded
    project_requantize_multipliers: torch.Tensor = None
    project_requantize_shifts: torch.Tensor = None
    project_input_offset: int = 0  # folded into bias
    project_output_offset: int = 0
    project_activation_min: int = -128
    project_activation_max: int = 127
    # Residual add (None when this block has no skip path).
    residual_input: Optional[TensorSlot] = None
    residual_self_zero_point: int = 0
    residual_self_multiplier: int = 0
    residual_self_shift: int = 0
    residual_other_zero_point: int = 0
    residual_other_multiplier: int = 0
    residual_other_shift: int = 0
    residual_output_zero_point: int = 0
    residual_output_multiplier: int = 0
    residual_output_shift: int = 0
    residual_activation_min: int = -128
    residual_activation_max: int = 127


@dataclass
class FusedQuantizeStemDwconv2dConv2dLayer:
    """quantize_per_tensor + stem 3x3 stride-2 + B0 dwconv 3x3 + B0 project
    1x1 fused into one runtime call.  Reads raw float input; the per-tensor
    quantize happens row-by-row inside the kernel into a small int8 rolling
    buffer that the stem consumes, so the H*W*C int8 quantized input never
    materializes in arena.
    """
    kernel: str = "quantize_stem_dwconv2d_conv2d_fused_s8"
    # Float input comes from the entry point parameter, not arena.  Track
    # the input node name only for documentation / debugging.
    input_name: str = ""
    output: TensorSlot = None
    num_input_elements: int = 0  # for safety / loop bounds
    quant_scale: float = 1.0
    quant_zero_point: int = 0
    quant_qmin: int = -128
    quant_qmax: int = 127
    stem_weight: torch.Tensor = None
    stem_bias: Optional[torch.Tensor] = None
    stem_weight_packed_32: Optional[torch.Tensor] = None
    stem_requantize_multipliers: torch.Tensor = None
    stem_requantize_shifts: torch.Tensor = None
    stem_input_offset: int = 0
    stem_output_offset: int = 0
    stem_activation_min: int = -128
    stem_activation_max: int = 127
    dw_weight: torch.Tensor = None
    dw_bias: Optional[torch.Tensor] = None
    dw_bias_with_offset_full: Optional[torch.Tensor] = None
    dw_requantize_multipliers: torch.Tensor = None
    dw_requantize_shifts: torch.Tensor = None
    dw_stride: tuple = (1, 1)
    dw_padding: tuple = (1, 1)
    dw_input_offset: int = 0
    dw_output_offset: int = 0
    dw_activation_min: int = -128
    dw_activation_max: int = 127
    project_weight: torch.Tensor = None
    project_bias: Optional[torch.Tensor] = None
    project_requantize_multipliers: torch.Tensor = None
    project_requantize_shifts: torch.Tensor = None
    project_input_offset: int = 0
    project_output_offset: int = 0
    project_activation_min: int = -128
    project_activation_max: int = 127


@dataclass
class FusedStemDwconv2dConv2dLayer:
    """MV2 first-stage chain fused: stem conv (3x3 stride-2 pad-1 Cin=3)
    + B0 dwconv (3x3 stride-1 pad-1) + B0 project (1x1).  Streams stem
    output rows into a 3-row rolling buffer the dwconv consumes
    immediately, then a 1-row dwconv-output scratch fed into the
    project.  Eliminates both the stem output (32ch x 112^2 = 401 KB at
    1.0/r=224) and the dwconv output from the arena.
    """
    kernel: str = "stem_dwconv2d_conv2d_fused_s8"
    input: TensorSlot = None
    output: TensorSlot = None
    # Stem 3x3 stride-2
    stem_weight: torch.Tensor = None  # OHWI [stem_out_c, 3, 3, 3]
    stem_bias: Optional[torch.Tensor] = None  # offset-folded
    stem_weight_packed_32: Optional[torch.Tensor] = None
    stem_requantize_multipliers: torch.Tensor = None
    stem_requantize_shifts: torch.Tensor = None
    stem_input_offset: int = 0  # original (not folded) — needed for pad fill
    stem_output_offset: int = 0
    stem_activation_min: int = -128
    stem_activation_max: int = 127
    # Dwconv 3x3
    dw_weight: torch.Tensor = None
    dw_bias: Optional[torch.Tensor] = None
    dw_bias_with_offset_full: Optional[torch.Tensor] = None
    dw_requantize_multipliers: torch.Tensor = None
    dw_requantize_shifts: torch.Tensor = None
    dw_stride: tuple = (1, 1)
    dw_padding: tuple = (1, 1)
    dw_input_offset: int = 0
    dw_output_offset: int = 0
    dw_activation_min: int = -128
    dw_activation_max: int = 127
    # Project 1x1
    project_weight: torch.Tensor = None
    project_bias: Optional[torch.Tensor] = None
    project_requantize_multipliers: torch.Tensor = None
    project_requantize_shifts: torch.Tensor = None
    project_input_offset: int = 0
    project_output_offset: int = 0
    project_activation_min: int = -128
    project_activation_max: int = 127


@dataclass
class FusedDwconv2dConv2dLayer:
    """MV2 B0-style block (expand_ratio=1): 3x3 dwconv + 1x1 project fused.

    Runtime kernel streams the dwconv output through a single row buffer
    that the project conv consumes immediately, eliminating the full
    HxWxC dwconv intermediate.
    """
    kernel: str = "dwconv2d_conv2d_fused_s8"
    input: TensorSlot = None
    output: TensorSlot = None
    # Dwconv 3x3
    dw_weight: torch.Tensor = None
    dw_bias: Optional[torch.Tensor] = None
    dw_bias_with_offset_full: Optional[torch.Tensor] = None
    dw_requantize_multipliers: torch.Tensor = None
    dw_requantize_shifts: torch.Tensor = None
    dw_stride: tuple = (1, 1)
    dw_padding: tuple = (1, 1)
    dw_input_offset: int = 0
    dw_output_offset: int = 0
    dw_activation_min: int = -128
    dw_activation_max: int = 127
    # Project 1x1
    project_weight: torch.Tensor = None
    project_bias: Optional[torch.Tensor] = None
    project_requantize_multipliers: torch.Tensor = None
    project_requantize_shifts: torch.Tensor = None
    project_input_offset: int = 0
    project_output_offset: int = 0
    project_activation_min: int = -128
    project_activation_max: int = 127


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


def extract_quantized_conv2d_dwconv2d_fused(
    node: Node, program: ExportedProgram, offsets, sizes
) -> FusedConv2dDwconv2dLayer:
    a = _node_args_map(node, [
        "input",
        "expand_weight", "expand_bias",
        "expand_input_offset", "expand_output_offset",
        "expand_requantize_multipliers", "expand_requantize_shifts",
        "expand_activation_min", "expand_activation_max",
        "dw_weight", "dw_bias",
        "dw_stride", "dw_padding",
        "dw_input_offset", "dw_output_offset",
        "dw_requantize_multipliers", "dw_requantize_shifts",
        "dw_activation_min", "dw_activation_max",
    ])

    # ---- Expand: 1x1 conv with offset-fold (same logic as conv2d extractor). ----
    e_w_t = _resolve_tensor(program, a["expand_weight"])
    e_b_t = (
        _resolve_tensor(program, a["expand_bias"])
        if a["expand_bias"] is not None else None
    )
    e_w_int32 = e_w_t.detach().to(torch.int32).contiguous()
    # OHWI [out_c, 1, 1, in_c] — sum over the kH=kW=1=in_c axes gives the
    # per-OC weight sum that the offset-fold needs.
    sum_w_per_oc = e_w_int32.flatten(1).sum(dim=1)
    offset_term = sum_w_per_oc * int(a["expand_input_offset"])
    if e_b_t is not None:
        e_bias_folded = (
            e_b_t.detach().to(torch.int32).flatten() + offset_term
        ).to(torch.int32).contiguous()
    else:
        e_bias_folded = offset_term.to(torch.int32).contiguous()
    e_mults = _resolve_tensor(program, a["expand_requantize_multipliers"]).detach().to(torch.int32).flatten().contiguous()
    e_shifts = _resolve_tensor(program, a["expand_requantize_shifts"]).detach().to(torch.int8).flatten().contiguous()

    # ---- Dwconv: bias_with_offset_full computation. ----
    d_w_t = _resolve_tensor(program, a["dw_weight"])
    d_b_t = (
        _resolve_tensor(program, a["dw_bias"])
        if a["dw_bias"] is not None else None
    )
    dw_bias_with_offset_full = None
    if d_b_t is not None and int(a["dw_input_offset"]) != 0:
        d_w_int32 = d_w_t.detach().to(torch.int32)
        sum_dw_per_c = d_w_int32.sum(dim=(0, 1, 2))  # IHWO -> [C]
        dw_offset_term = sum_dw_per_c * int(a["dw_input_offset"])
        dw_bias_with_offset_full = (
            d_b_t.detach().to(torch.int32).flatten() + dw_offset_term
        ).to(torch.int32).contiguous()
    d_mults = _resolve_tensor(program, a["dw_requantize_multipliers"]).detach().to(torch.int32).flatten().contiguous()
    d_shifts = _resolve_tensor(program, a["dw_requantize_shifts"]).detach().to(torch.int8).flatten().contiguous()

    return FusedConv2dDwconv2dLayer(
        input=_slot_from_node(a["input"], offsets, sizes),
        output=_slot_from_node(node, offsets, sizes),
        expand_weight=e_w_t.detach().to(torch.int8).contiguous(),
        expand_bias=e_bias_folded,
        expand_requantize_multipliers=e_mults,
        expand_requantize_shifts=e_shifts,
        # Folded into bias; runtime treats input_offset as 0.
        expand_input_offset=0,
        expand_output_offset=int(a["expand_output_offset"]),
        expand_activation_min=int(a["expand_activation_min"]),
        expand_activation_max=int(a["expand_activation_max"]),
        dw_weight=d_w_t.detach().to(torch.int8).contiguous(),
        dw_bias=(
            d_b_t.detach().to(torch.int32).flatten().contiguous()
            if d_b_t is not None else None
        ),
        dw_bias_with_offset_full=dw_bias_with_offset_full,
        dw_requantize_multipliers=d_mults,
        dw_requantize_shifts=d_shifts,
        dw_stride=_coerce_int_pair(a["dw_stride"]),
        dw_padding=_coerce_int_pair(a["dw_padding"]),
        dw_input_offset=int(a["dw_input_offset"]),
        dw_output_offset=int(a["dw_output_offset"]),
        dw_activation_min=int(a["dw_activation_min"]),
        dw_activation_max=int(a["dw_activation_max"]),
    )


def extract_quantized_conv2d_dwconv2d_conv2d_fused(
    node: Node, program: ExportedProgram, offsets, sizes
) -> FusedInvertedResidualLayer:
    a = _node_args_map(node, [
        "input",
        "expand_weight", "expand_bias",
        "expand_input_offset", "expand_output_offset",
        "expand_requantize_multipliers", "expand_requantize_shifts",
        "expand_activation_min", "expand_activation_max",
        "dw_weight", "dw_bias",
        "dw_stride", "dw_padding",
        "dw_input_offset", "dw_output_offset",
        "dw_requantize_multipliers", "dw_requantize_shifts",
        "dw_activation_min", "dw_activation_max",
        "project_weight", "project_bias",
        "project_input_offset", "project_output_offset",
        "project_requantize_multipliers", "project_requantize_shifts",
        "project_activation_min", "project_activation_max",
        "residual_input",
        "residual_self_zero_point", "residual_self_multiplier", "residual_self_shift",
        "residual_other_zero_point", "residual_other_multiplier", "residual_other_shift",
        "residual_output_zero_point", "residual_output_multiplier", "residual_output_shift",
        "residual_activation_min", "residual_activation_max",
    ])

    # Expand 1x1 with offset-fold (same as conv1x1 extractor).
    e_w_t = _resolve_tensor(program, a["expand_weight"])
    e_b_t = _resolve_tensor(program, a["expand_bias"]) if a["expand_bias"] is not None else None
    e_w_int32 = e_w_t.detach().to(torch.int32).contiguous()
    sum_w_per_oc_e = e_w_int32.flatten(1).sum(dim=1)
    offset_term_e = sum_w_per_oc_e * int(a["expand_input_offset"])
    if e_b_t is not None:
        e_bias_folded = (
            e_b_t.detach().to(torch.int32).flatten() + offset_term_e
        ).to(torch.int32).contiguous()
    else:
        e_bias_folded = offset_term_e.to(torch.int32).contiguous()

    # Dwconv 3x3 with optional bias_with_offset_full.
    d_w_t = _resolve_tensor(program, a["dw_weight"])
    d_b_t = _resolve_tensor(program, a["dw_bias"]) if a["dw_bias"] is not None else None
    dw_bias_with_offset_full = None
    if d_b_t is not None and int(a["dw_input_offset"]) != 0:
        d_w_int32 = d_w_t.detach().to(torch.int32)
        sum_dw_per_c = d_w_int32.sum(dim=(0, 1, 2))
        dw_offset_term = sum_dw_per_c * int(a["dw_input_offset"])
        dw_bias_with_offset_full = (
            d_b_t.detach().to(torch.int32).flatten() + dw_offset_term
        ).to(torch.int32).contiguous()

    # Project 1x1 with offset-fold.
    p_w_t = _resolve_tensor(program, a["project_weight"])
    p_b_t = _resolve_tensor(program, a["project_bias"]) if a["project_bias"] is not None else None
    p_w_int32 = p_w_t.detach().to(torch.int32).contiguous()
    sum_w_per_oc_p = p_w_int32.flatten(1).sum(dim=1)
    offset_term_p = sum_w_per_oc_p * int(a["project_input_offset"])
    if p_b_t is not None:
        p_bias_folded = (
            p_b_t.detach().to(torch.int32).flatten() + offset_term_p
        ).to(torch.int32).contiguous()
    else:
        p_bias_folded = offset_term_p.to(torch.int32).contiguous()

    residual_slot = None
    if a["residual_input"] is not None:
        residual_slot = _slot_from_node(a["residual_input"], offsets, sizes)

    return FusedInvertedResidualLayer(
        input=_slot_from_node(a["input"], offsets, sizes),
        output=_slot_from_node(node, offsets, sizes),
        expand_weight=e_w_t.detach().to(torch.int8).contiguous(),
        expand_bias=e_bias_folded,
        expand_requantize_multipliers=_resolve_tensor(program, a["expand_requantize_multipliers"]).detach().to(torch.int32).flatten().contiguous(),
        expand_requantize_shifts=_resolve_tensor(program, a["expand_requantize_shifts"]).detach().to(torch.int8).flatten().contiguous(),
        expand_input_offset=0,  # folded
        expand_output_offset=int(a["expand_output_offset"]),
        expand_activation_min=int(a["expand_activation_min"]),
        expand_activation_max=int(a["expand_activation_max"]),
        dw_weight=d_w_t.detach().to(torch.int8).contiguous(),
        dw_bias=d_b_t.detach().to(torch.int32).flatten().contiguous() if d_b_t is not None else None,
        dw_bias_with_offset_full=dw_bias_with_offset_full,
        dw_requantize_multipliers=_resolve_tensor(program, a["dw_requantize_multipliers"]).detach().to(torch.int32).flatten().contiguous(),
        dw_requantize_shifts=_resolve_tensor(program, a["dw_requantize_shifts"]).detach().to(torch.int8).flatten().contiguous(),
        dw_stride=_coerce_int_pair(a["dw_stride"]),
        dw_padding=_coerce_int_pair(a["dw_padding"]),
        dw_input_offset=int(a["dw_input_offset"]),
        dw_output_offset=int(a["dw_output_offset"]),
        dw_activation_min=int(a["dw_activation_min"]),
        dw_activation_max=int(a["dw_activation_max"]),
        project_weight=p_w_t.detach().to(torch.int8).contiguous(),
        project_bias=p_bias_folded,
        project_requantize_multipliers=_resolve_tensor(program, a["project_requantize_multipliers"]).detach().to(torch.int32).flatten().contiguous(),
        project_requantize_shifts=_resolve_tensor(program, a["project_requantize_shifts"]).detach().to(torch.int8).flatten().contiguous(),
        project_input_offset=0,  # folded
        project_output_offset=int(a["project_output_offset"]),
        project_activation_min=int(a["project_activation_min"]),
        project_activation_max=int(a["project_activation_max"]),
        residual_input=residual_slot,
        residual_self_zero_point=int(a["residual_self_zero_point"]),
        residual_self_multiplier=int(a["residual_self_multiplier"]),
        residual_self_shift=int(a["residual_self_shift"]),
        residual_other_zero_point=int(a["residual_other_zero_point"]),
        residual_other_multiplier=int(a["residual_other_multiplier"]),
        residual_other_shift=int(a["residual_other_shift"]),
        residual_output_zero_point=int(a["residual_output_zero_point"]),
        residual_output_multiplier=int(a["residual_output_multiplier"]),
        residual_output_shift=int(a["residual_output_shift"]),
        residual_activation_min=int(a["residual_activation_min"]),
        residual_activation_max=int(a["residual_activation_max"]),
    )


def extract_quantize_stem_dwconv2d_conv2d_fused(
    node: Node, program: ExportedProgram, offsets, sizes
) -> FusedQuantizeStemDwconv2dConv2dLayer:
    a = _node_args_map(node, [
        "input",
        "quant_scale", "quant_zero_point", "quant_qmin", "quant_qmax",
        "stem_weight", "stem_bias",
        "stem_input_offset", "stem_output_offset",
        "stem_requantize_multipliers", "stem_requantize_shifts",
        "stem_activation_min", "stem_activation_max",
        "dw_weight", "dw_bias",
        "dw_stride", "dw_padding",
        "dw_input_offset", "dw_output_offset",
        "dw_requantize_multipliers", "dw_requantize_shifts",
        "dw_activation_min", "dw_activation_max",
        "project_weight", "project_bias",
        "project_input_offset", "project_output_offset",
        "project_requantize_multipliers", "project_requantize_shifts",
        "project_activation_min", "project_activation_max",
    ])

    # Stem: same packed-32 + offset-fold as the non-quant stem extractor.
    s_w_t = _resolve_tensor(program, a["stem_weight"])
    s_b_t = _resolve_tensor(program, a["stem_bias"]) if a["stem_bias"] is not None else None
    s_w_int32 = s_w_t.detach().to(torch.int32).contiguous()
    out_c, kh, kw, in_c = s_w_int32.shape
    flat = s_w_int32.reshape(out_c, kh * kw * in_c)
    pad = torch.zeros((out_c, 32 - flat.shape[1]), dtype=torch.int32)
    s_w_packed_32 = torch.cat([flat, pad], dim=1).to(torch.int8).contiguous()
    sum_w_per_oc = s_w_int32.flatten(1).sum(dim=1)
    s_offset_term = sum_w_per_oc * int(a["stem_input_offset"])
    if s_b_t is not None:
        s_bias_folded = (
            s_b_t.detach().to(torch.int32).flatten() + s_offset_term
        ).to(torch.int32).contiguous()
    else:
        s_bias_folded = s_offset_term.to(torch.int32).contiguous()

    d_w_t = _resolve_tensor(program, a["dw_weight"])
    d_b_t = _resolve_tensor(program, a["dw_bias"]) if a["dw_bias"] is not None else None
    dw_bias_with_offset_full = None
    if d_b_t is not None and int(a["dw_input_offset"]) != 0:
        d_w_int32 = d_w_t.detach().to(torch.int32)
        sum_dw_per_c = d_w_int32.sum(dim=(0, 1, 2))
        dw_offset_term = sum_dw_per_c * int(a["dw_input_offset"])
        dw_bias_with_offset_full = (
            d_b_t.detach().to(torch.int32).flatten() + dw_offset_term
        ).to(torch.int32).contiguous()

    p_w_t = _resolve_tensor(program, a["project_weight"])
    p_b_t = _resolve_tensor(program, a["project_bias"]) if a["project_bias"] is not None else None
    p_w_int32 = p_w_t.detach().to(torch.int32).contiguous()
    sum_w_per_oc_p = p_w_int32.flatten(1).sum(dim=1)
    p_offset_term = sum_w_per_oc_p * int(a["project_input_offset"])
    if p_b_t is not None:
        p_bias_folded = (
            p_b_t.detach().to(torch.int32).flatten() + p_offset_term
        ).to(torch.int32).contiguous()
    else:
        p_bias_folded = p_offset_term.to(torch.int32).contiguous()

    input_node = a["input"]
    input_fake = get_first_fake_tensor(input_node) if hasattr(input_node, "name") else None
    num_input = 1
    if input_fake is not None:
        for dim in input_fake.shape:
            num_input *= int(dim)
    return FusedQuantizeStemDwconv2dConv2dLayer(
        input_name=str(input_node.name) if hasattr(input_node, "name") else "input_float",
        output=_slot_from_node(node, offsets, sizes),
        num_input_elements=num_input,
        quant_scale=float(a["quant_scale"]),
        quant_zero_point=int(a["quant_zero_point"]),
        quant_qmin=int(a["quant_qmin"]),
        quant_qmax=int(a["quant_qmax"]),
        stem_weight=s_w_t.detach().to(torch.int8).contiguous(),
        stem_bias=s_bias_folded,
        stem_weight_packed_32=s_w_packed_32,
        stem_requantize_multipliers=_resolve_tensor(program, a["stem_requantize_multipliers"]).detach().to(torch.int32).flatten().contiguous(),
        stem_requantize_shifts=_resolve_tensor(program, a["stem_requantize_shifts"]).detach().to(torch.int8).flatten().contiguous(),
        stem_input_offset=int(a["stem_input_offset"]),
        stem_output_offset=int(a["stem_output_offset"]),
        stem_activation_min=int(a["stem_activation_min"]),
        stem_activation_max=int(a["stem_activation_max"]),
        dw_weight=d_w_t.detach().to(torch.int8).contiguous(),
        dw_bias=d_b_t.detach().to(torch.int32).flatten().contiguous() if d_b_t is not None else None,
        dw_bias_with_offset_full=dw_bias_with_offset_full,
        dw_requantize_multipliers=_resolve_tensor(program, a["dw_requantize_multipliers"]).detach().to(torch.int32).flatten().contiguous(),
        dw_requantize_shifts=_resolve_tensor(program, a["dw_requantize_shifts"]).detach().to(torch.int8).flatten().contiguous(),
        dw_stride=_coerce_int_pair(a["dw_stride"]),
        dw_padding=_coerce_int_pair(a["dw_padding"]),
        dw_input_offset=int(a["dw_input_offset"]),
        dw_output_offset=int(a["dw_output_offset"]),
        dw_activation_min=int(a["dw_activation_min"]),
        dw_activation_max=int(a["dw_activation_max"]),
        project_weight=p_w_t.detach().to(torch.int8).contiguous(),
        project_bias=p_bias_folded,
        project_requantize_multipliers=_resolve_tensor(program, a["project_requantize_multipliers"]).detach().to(torch.int32).flatten().contiguous(),
        project_requantize_shifts=_resolve_tensor(program, a["project_requantize_shifts"]).detach().to(torch.int8).flatten().contiguous(),
        project_input_offset=0,  # folded into bias
        project_output_offset=int(a["project_output_offset"]),
        project_activation_min=int(a["project_activation_min"]),
        project_activation_max=int(a["project_activation_max"]),
    )


def extract_quantized_stem_dwconv2d_conv2d_fused(
    node: Node, program: ExportedProgram, offsets, sizes
) -> FusedStemDwconv2dConv2dLayer:
    a = _node_args_map(node, [
        "input",
        "stem_weight", "stem_bias",
        "stem_input_offset", "stem_output_offset",
        "stem_requantize_multipliers", "stem_requantize_shifts",
        "stem_activation_min", "stem_activation_max",
        "dw_weight", "dw_bias",
        "dw_stride", "dw_padding",
        "dw_input_offset", "dw_output_offset",
        "dw_requantize_multipliers", "dw_requantize_shifts",
        "dw_activation_min", "dw_activation_max",
        "project_weight", "project_bias",
        "project_input_offset", "project_output_offset",
        "project_requantize_multipliers", "project_requantize_shifts",
        "project_activation_min", "project_activation_max",
    ])

    # Stem 3x3 with packed-32 weight prep + bias_with_offset fold.
    s_w_t = _resolve_tensor(program, a["stem_weight"])
    s_b_t = _resolve_tensor(program, a["stem_bias"]) if a["stem_bias"] is not None else None
    s_w_int32 = s_w_t.detach().to(torch.int32).contiguous()
    out_c, kh, kw, in_c = s_w_int32.shape
    # OHWI flatten = (kh, kw, ic) = 27 elements, pad to 32 with zeros.
    flat = s_w_int32.reshape(out_c, kh * kw * in_c)  # [out_c, 27]
    pad = torch.zeros((out_c, 32 - flat.shape[1]), dtype=torch.int32)
    s_w_packed_32 = torch.cat([flat, pad], dim=1).to(torch.int8).contiguous()
    # Bias-with-offset fold (per OC: bias + stem_input_offset * sum(w)).
    sum_w_per_oc = s_w_int32.flatten(1).sum(dim=1)
    s_offset_term = sum_w_per_oc * int(a["stem_input_offset"])
    if s_b_t is not None:
        s_bias_folded = (
            s_b_t.detach().to(torch.int32).flatten() + s_offset_term
        ).to(torch.int32).contiguous()
    else:
        s_bias_folded = s_offset_term.to(torch.int32).contiguous()

    # Dwconv 3x3 (same logic as dwconv-only extractor).
    d_w_t = _resolve_tensor(program, a["dw_weight"])
    d_b_t = _resolve_tensor(program, a["dw_bias"]) if a["dw_bias"] is not None else None
    dw_bias_with_offset_full = None
    if d_b_t is not None and int(a["dw_input_offset"]) != 0:
        d_w_int32 = d_w_t.detach().to(torch.int32)
        sum_dw_per_c = d_w_int32.sum(dim=(0, 1, 2))
        dw_offset_term = sum_dw_per_c * int(a["dw_input_offset"])
        dw_bias_with_offset_full = (
            d_b_t.detach().to(torch.int32).flatten() + dw_offset_term
        ).to(torch.int32).contiguous()

    # Project 1x1 with offset fold.
    p_w_t = _resolve_tensor(program, a["project_weight"])
    p_b_t = _resolve_tensor(program, a["project_bias"]) if a["project_bias"] is not None else None
    p_w_int32 = p_w_t.detach().to(torch.int32).contiguous()
    sum_w_per_oc_p = p_w_int32.flatten(1).sum(dim=1)
    p_offset_term = sum_w_per_oc_p * int(a["project_input_offset"])
    if p_b_t is not None:
        p_bias_folded = (
            p_b_t.detach().to(torch.int32).flatten() + p_offset_term
        ).to(torch.int32).contiguous()
    else:
        p_bias_folded = p_offset_term.to(torch.int32).contiguous()

    return FusedStemDwconv2dConv2dLayer(
        input=_slot_from_node(a["input"], offsets, sizes),
        output=_slot_from_node(node, offsets, sizes),
        stem_weight=s_w_t.detach().to(torch.int8).contiguous(),
        stem_bias=s_bias_folded,
        stem_weight_packed_32=s_w_packed_32,
        stem_requantize_multipliers=_resolve_tensor(program, a["stem_requantize_multipliers"]).detach().to(torch.int32).flatten().contiguous(),
        stem_requantize_shifts=_resolve_tensor(program, a["stem_requantize_shifts"]).detach().to(torch.int8).flatten().contiguous(),
        stem_input_offset=int(a["stem_input_offset"]),  # kept for pad-fill (-input_offset)
        stem_output_offset=int(a["stem_output_offset"]),
        stem_activation_min=int(a["stem_activation_min"]),
        stem_activation_max=int(a["stem_activation_max"]),
        dw_weight=d_w_t.detach().to(torch.int8).contiguous(),
        dw_bias=d_b_t.detach().to(torch.int32).flatten().contiguous() if d_b_t is not None else None,
        dw_bias_with_offset_full=dw_bias_with_offset_full,
        dw_requantize_multipliers=_resolve_tensor(program, a["dw_requantize_multipliers"]).detach().to(torch.int32).flatten().contiguous(),
        dw_requantize_shifts=_resolve_tensor(program, a["dw_requantize_shifts"]).detach().to(torch.int8).flatten().contiguous(),
        dw_stride=_coerce_int_pair(a["dw_stride"]),
        dw_padding=_coerce_int_pair(a["dw_padding"]),
        dw_input_offset=int(a["dw_input_offset"]),
        dw_output_offset=int(a["dw_output_offset"]),
        dw_activation_min=int(a["dw_activation_min"]),
        dw_activation_max=int(a["dw_activation_max"]),
        project_weight=p_w_t.detach().to(torch.int8).contiguous(),
        project_bias=p_bias_folded,
        project_requantize_multipliers=_resolve_tensor(program, a["project_requantize_multipliers"]).detach().to(torch.int32).flatten().contiguous(),
        project_requantize_shifts=_resolve_tensor(program, a["project_requantize_shifts"]).detach().to(torch.int8).flatten().contiguous(),
        project_input_offset=0,  # folded into bias
        project_output_offset=int(a["project_output_offset"]),
        project_activation_min=int(a["project_activation_min"]),
        project_activation_max=int(a["project_activation_max"]),
    )


def extract_quantized_dwconv2d_conv2d_fused(
    node: Node, program: ExportedProgram, offsets, sizes
) -> FusedDwconv2dConv2dLayer:
    a = _node_args_map(node, [
        "input",
        "dw_weight", "dw_bias",
        "dw_stride", "dw_padding",
        "dw_input_offset", "dw_output_offset",
        "dw_requantize_multipliers", "dw_requantize_shifts",
        "dw_activation_min", "dw_activation_max",
        "project_weight", "project_bias",
        "project_input_offset", "project_output_offset",
        "project_requantize_multipliers", "project_requantize_shifts",
        "project_activation_min", "project_activation_max",
    ])
    # Dwconv with optional bias_with_offset_full.
    d_w_t = _resolve_tensor(program, a["dw_weight"])
    d_b_t = _resolve_tensor(program, a["dw_bias"]) if a["dw_bias"] is not None else None
    dw_bias_with_offset_full = None
    if d_b_t is not None and int(a["dw_input_offset"]) != 0:
        d_w_int32 = d_w_t.detach().to(torch.int32)
        sum_dw_per_c = d_w_int32.sum(dim=(0, 1, 2))
        dw_offset_term = sum_dw_per_c * int(a["dw_input_offset"])
        dw_bias_with_offset_full = (
            d_b_t.detach().to(torch.int32).flatten() + dw_offset_term
        ).to(torch.int32).contiguous()

    # Project 1x1 with offset-fold.
    p_w_t = _resolve_tensor(program, a["project_weight"])
    p_b_t = _resolve_tensor(program, a["project_bias"]) if a["project_bias"] is not None else None
    p_w_int32 = p_w_t.detach().to(torch.int32).contiguous()
    sum_w_per_oc = p_w_int32.flatten(1).sum(dim=1)
    offset_term = sum_w_per_oc * int(a["project_input_offset"])
    if p_b_t is not None:
        p_bias_folded = (
            p_b_t.detach().to(torch.int32).flatten() + offset_term
        ).to(torch.int32).contiguous()
    else:
        p_bias_folded = offset_term.to(torch.int32).contiguous()

    return FusedDwconv2dConv2dLayer(
        input=_slot_from_node(a["input"], offsets, sizes),
        output=_slot_from_node(node, offsets, sizes),
        dw_weight=d_w_t.detach().to(torch.int8).contiguous(),
        dw_bias=d_b_t.detach().to(torch.int32).flatten().contiguous() if d_b_t is not None else None,
        dw_bias_with_offset_full=dw_bias_with_offset_full,
        dw_requantize_multipliers=_resolve_tensor(program, a["dw_requantize_multipliers"]).detach().to(torch.int32).flatten().contiguous(),
        dw_requantize_shifts=_resolve_tensor(program, a["dw_requantize_shifts"]).detach().to(torch.int8).flatten().contiguous(),
        dw_stride=_coerce_int_pair(a["dw_stride"]),
        dw_padding=_coerce_int_pair(a["dw_padding"]),
        dw_input_offset=int(a["dw_input_offset"]),
        dw_output_offset=int(a["dw_output_offset"]),
        dw_activation_min=int(a["dw_activation_min"]),
        dw_activation_max=int(a["dw_activation_max"]),
        project_weight=p_w_t.detach().to(torch.int8).contiguous(),
        project_bias=p_bias_folded,
        project_requantize_multipliers=_resolve_tensor(program, a["project_requantize_multipliers"]).detach().to(torch.int32).flatten().contiguous(),
        project_requantize_shifts=_resolve_tensor(program, a["project_requantize_shifts"]).detach().to(torch.int8).flatten().contiguous(),
        project_input_offset=0,  # folded into bias
        project_output_offset=int(a["project_output_offset"]),
        project_activation_min=int(a["project_activation_min"]),
        project_activation_max=int(a["project_activation_max"]),
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
    "cortex_m.quantized_conv2d_dwconv2d_fused.default": extract_quantized_conv2d_dwconv2d_fused,
    "cortex_m.quantized_conv2d_dwconv2d_conv2d_fused.default": extract_quantized_conv2d_dwconv2d_conv2d_fused,
    "cortex_m.quantized_dwconv2d_conv2d_fused.default": extract_quantized_dwconv2d_conv2d_fused,
    "cortex_m.quantized_stem_dwconv2d_conv2d_fused.default": extract_quantized_stem_dwconv2d_conv2d_fused,
    "cortex_m.quantize_stem_dwconv2d_conv2d_fused.default": extract_quantize_stem_dwconv2d_conv2d_fused,
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
