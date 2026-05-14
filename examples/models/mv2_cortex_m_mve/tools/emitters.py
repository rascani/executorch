# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).
"""
C source emission for the standalone MobileNetV2 inference path.

Produces:
  mv2_weights.h / mv2_weights.c — int8/int32 const arrays for weights, biases, kernel sums.
  mv2_params.h                   — static const LayerParams instances + arena offsets + MV2_ARENA_BYTES.
  input_fixture.h                — input tensor (float NCHW) and EXPECTED_TOP1.

The generated files contain only data; the inference function and kernels are hand-written.
"""

from pathlib import Path
from typing import Iterable

import torch

from .extractors import (
    AvgPool2dLayer,
    Conv2dLayer,
    DepthwiseConv2dLayer,
    DequantOutputLayer,
    FusedConv2dDwconv2dLayer,
    LinearLayer,
    MemcpyLayer,
    ProgramSchedule,
    QuantInputLayer,
    QuantizedAddLayer,
)


def _spatial_dims(shape):
    """Return (H, W, C) from a 4D NCHW-shape tensor."""
    if len(shape) == 4:
        return int(shape[2]), int(shape[3]), int(shape[1])
    raise ValueError(f"Expected 4D shape, got {shape}")


def _hex_byte(b: int) -> str:
    return f"0x{b & 0xFF:02x}"


def _emit_int8_array(name: str, tensor: torch.Tensor) -> str:
    flat = tensor.flatten().tolist()
    lines = [f"const int8_t {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), 16):
        chunk = ", ".join(_hex_byte(int(v)) for v in flat[i : i + 16])
        lines.append(f"  {chunk},")
    lines.append("};\n")
    return "\n".join(lines)


def _emit_int32_array(name: str, tensor: torch.Tensor) -> str:
    flat = tensor.flatten().to(torch.int32).tolist()
    lines = [f"const int32_t {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), 8):
        chunk = ", ".join(f"{int(v)}" for v in flat[i : i + 8])
        lines.append(f"  {chunk},")
    lines.append("};\n")
    return "\n".join(lines)


def _emit_float_array(name: str, tensor: torch.Tensor) -> str:
    flat = tensor.flatten().to(torch.float32).tolist()
    lines = [f"const float {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), 6):
        chunk = ", ".join(f"{v:.9g}f" for v in flat[i : i + 6])
        lines.append(f"  {chunk},")
    lines.append("};\n")
    return "\n".join(lines)


def emit_weights(out_dir: Path, layers: Iterable, header_guard: str = "MV2_WEIGHTS_H_") -> None:
    decls: list[str] = []
    defns: list[str] = []
    for i, layer in enumerate(layers):
        if isinstance(layer, LinearLayer):
            stem = f"L{i}_linear"
            defns.append(_emit_int8_array(f"{stem}_weight", layer.weight))
            decls.append(f"extern const int8_t {stem}_weight[];")
            if layer.bias is not None:
                defns.append(_emit_int32_array(f"{stem}_bias", layer.bias))
                decls.append(f"extern const int32_t {stem}_bias[];")
            if layer.kernel_sum is not None:
                defns.append(_emit_int32_array(f"{stem}_kernel_sum", layer.kernel_sum))
                decls.append(f"extern const int32_t {stem}_kernel_sum[];")
        elif isinstance(layer, Conv2dLayer):
            stem = f"L{i}_conv2d"
            defns.append(_emit_int8_array(f"{stem}_weight", layer.weight))
            decls.append(f"extern const int8_t {stem}_weight[];")
            if layer.weight_packed_32 is not None:
                defns.append(_emit_int8_array(f"{stem}_weight_packed_32", layer.weight_packed_32))
                decls.append(f"extern const int8_t {stem}_weight_packed_32[];")
            if layer.bias is not None:
                defns.append(_emit_int32_array(f"{stem}_bias", layer.bias))
                decls.append(f"extern const int32_t {stem}_bias[];")
            defns.append(_emit_int32_array(f"{stem}_requant_mults", layer.requantize_multipliers))
            defns.append(_emit_int8_array(f"{stem}_requant_shifts", layer.requantize_shifts))
            decls.append(f"extern const int32_t {stem}_requant_mults[];")
            decls.append(f"extern const int8_t  {stem}_requant_shifts[];")
        elif isinstance(layer, DepthwiseConv2dLayer):
            stem = f"L{i}_dwconv2d"
            defns.append(_emit_int8_array(f"{stem}_weight", layer.weight))
            decls.append(f"extern const int8_t {stem}_weight[];")
            if layer.bias is not None:
                defns.append(_emit_int32_array(f"{stem}_bias", layer.bias))
                decls.append(f"extern const int32_t {stem}_bias[];")
            if layer.bias_with_offset_full is not None:
                defns.append(_emit_int32_array(f"{stem}_bias_off", layer.bias_with_offset_full))
                decls.append(f"extern const int32_t {stem}_bias_off[];")
            defns.append(_emit_int32_array(f"{stem}_requant_mults", layer.requantize_multipliers))
            defns.append(_emit_int8_array(f"{stem}_requant_shifts", layer.requantize_shifts))
            decls.append(f"extern const int32_t {stem}_requant_mults[];")
            decls.append(f"extern const int8_t  {stem}_requant_shifts[];")
        elif isinstance(layer, FusedConv2dDwconv2dLayer):
            stem = f"L{i}_fused"
            # Expand 1x1 weight + bias (offset-folded) + requant
            defns.append(_emit_int8_array(f"{stem}_expand_weight", layer.expand_weight))
            decls.append(f"extern const int8_t {stem}_expand_weight[];")
            if layer.expand_bias is not None:
                defns.append(_emit_int32_array(f"{stem}_expand_bias", layer.expand_bias))
                decls.append(f"extern const int32_t {stem}_expand_bias[];")
            defns.append(_emit_int32_array(f"{stem}_expand_requant_mults", layer.expand_requantize_multipliers))
            defns.append(_emit_int8_array(f"{stem}_expand_requant_shifts", layer.expand_requantize_shifts))
            decls.append(f"extern const int32_t {stem}_expand_requant_mults[];")
            decls.append(f"extern const int8_t  {stem}_expand_requant_shifts[];")
            # Dwconv 3x3 weight + bias + bias_off + requant
            defns.append(_emit_int8_array(f"{stem}_dw_weight", layer.dw_weight))
            decls.append(f"extern const int8_t {stem}_dw_weight[];")
            if layer.dw_bias is not None:
                defns.append(_emit_int32_array(f"{stem}_dw_bias", layer.dw_bias))
                decls.append(f"extern const int32_t {stem}_dw_bias[];")
            if layer.dw_bias_with_offset_full is not None:
                defns.append(_emit_int32_array(f"{stem}_dw_bias_off", layer.dw_bias_with_offset_full))
                decls.append(f"extern const int32_t {stem}_dw_bias_off[];")
            defns.append(_emit_int32_array(f"{stem}_dw_requant_mults", layer.dw_requantize_multipliers))
            defns.append(_emit_int8_array(f"{stem}_dw_requant_shifts", layer.dw_requantize_shifts))
            decls.append(f"extern const int32_t {stem}_dw_requant_mults[];")
            decls.append(f"extern const int8_t  {stem}_dw_requant_shifts[];")

    header = (
        f"#ifndef {header_guard}\n"
        f"#define {header_guard}\n\n"
        "#include <stdint.h>\n\n"
        + "\n".join(decls)
        + f"\n\n#endif\n"
    )
    (out_dir / "mv2_weights.h").write_text(header)

    source = (
        '#include "mv2_weights.h"\n'
        '\n'
        + "\n".join(defns)
    )
    (out_dir / "mv2_weights.c").write_text(source)


def _compute_fused_scratch_bytes(schedule: ProgramSchedule) -> int:
    """Size of the rolling-buffer scratch needed by every fused expand+dwconv
    in this schedule.  Three rows of the expand output at the layer's input
    spatial resolution: 3 * in_w * expand_out_c bytes.  Returns the max
    across all fused layers; zero when the model has no fused layers."""
    best = 0
    for layer in schedule.layers:
        if isinstance(layer, FusedConv2dDwconv2dLayer):
            # input shape is (N, C, H, W) per channels-last meta — but for
            # the scratch we need (in_w, expand_out_c).
            _, _, in_w, _ = _spatial_dims_full(layer.input.shape)
            expand_out_c = int(layer.expand_weight.shape[0])
            best = max(best, 3 * in_w * expand_out_c)
    return best


def _spatial_dims_full(shape):
    """Return (in_h, in_c_full=channels, in_w, dummy) compatible with the
    existing _spatial_dims output ordering: (H, W, C).  Wrapper that yields
    a 4-tuple to keep the caller readable."""
    h, w, c = _spatial_dims(shape)
    return (h, c, w, c)


def emit_arena(out_dir: Path, schedule: ProgramSchedule) -> None:
    """Emit mv2_arena.h with arena size + input/output metadata + ACT_OFFSET_*.

    Kept separate from mv2_params.h so arena.c (and other glue files) can
    include just the layout constants without dragging in the LayerParams
    static-const definitions.
    """
    lines: list[str] = [
        "#ifndef MV2_ARENA_H_",
        "#define MV2_ARENA_H_",
        "",
        "#include <stdint.h>",
        "",
        f"#define MV2_ARENA_BYTES {schedule.arena_bytes}u",
        f"#define MV2_INPUT_NUM_ELEMENTS {1 if schedule.input_slot is None else _prod(schedule.input_slot.shape)}u",
        f"#define MV2_OUTPUT_NUM_ELEMENTS {1 if schedule.output_slot is None else _prod(schedule.output_slot.shape)}u",
        f"#define MV2_OUTPUT_SCALE {schedule.output_scale:.9g}f",
        f"#define MV2_OUTPUT_ZERO_POINT {schedule.output_zero_point}",
        f"#define MV2_FUSED_SCRATCH_BYTES {_compute_fused_scratch_bytes(schedule)}u",
        "",
    ]
    seen: set[str] = set()
    for layer in schedule.layers:
        slots = []
        if isinstance(layer, QuantInputLayer):
            slots = [layer.output]
        elif isinstance(layer, (LinearLayer, Conv2dLayer, DepthwiseConv2dLayer,
                                FusedConv2dDwconv2dLayer,
                                AvgPool2dLayer, MemcpyLayer)):
            slots = [layer.input, layer.output]
        elif isinstance(layer, QuantizedAddLayer):
            slots = [layer.self_in, layer.other_in, layer.output]
        elif isinstance(layer, DequantOutputLayer):
            slots = [layer.input]
        for slot in slots:
            if slot is None or slot.name in seen:
                continue
            seen.add(slot.name)
            lines.append(f"#define ACT_OFFSET_{slot.name} {slot.offset}u")
    lines += ["", "#endif", ""]
    (out_dir / "mv2_arena.h").write_text("\n".join(lines))


def emit_params(
    out_dir: Path,
    schedule: ProgramSchedule,
    header_guard: str = "MV2_PARAMS_H_",
) -> None:
    lines: list[str] = [
        f"#ifndef {header_guard}",
        f"#define {header_guard}",
        "",
        '#include "mv2_layer_params.h"',
        '#include "mv2_weights.h"',
        '#include "mv2_arena.h"',
        "",
    ]

    # LayerParams instances.
    for i, layer in enumerate(schedule.layers):
        if isinstance(layer, QuantInputLayer):
            lines.append(
                f"static const QuantInputParams P_L{i}_quant = {{\n"
                f"  .num_elements = {_prod(layer.output.shape)}u,\n"
                f"  .scale = {layer.scale:.9g}f,\n"
                f"  .zero_point = {layer.zero_point},\n"
                f"  .qmin = {layer.qmin},\n"
                f"  .qmax = {layer.qmax},\n"
                "};"
            )
        elif isinstance(layer, LinearLayer):
            out_features, in_features = layer.weight.shape
            bias_ref = f"L{i}_linear_bias" if layer.bias is not None else "(const int32_t*)0"
            ksum_ref = f"L{i}_linear_kernel_sum" if layer.kernel_sum is not None else "(const int32_t*)0"
            lines.append(
                f"static const LinearParams P_L{i}_linear = {{\n"
                f"  .in_features = {in_features}u,\n"
                f"  .out_features = {out_features}u,\n"
                f"  .weight = L{i}_linear_weight,\n"
                f"  .bias = {bias_ref},\n"
                f"  .kernel_sum = {ksum_ref},\n"
                f"  .input_offset = {layer.input_offset},\n"
                f"  .filter_offset = {layer.filter_offset},\n"
                f"  .output_offset = {layer.output_offset},\n"
                f"  .multiplier = {layer.multiplier},\n"
                f"  .shift = {layer.shift},\n"
                f"  .activation_min = {layer.activation_min},\n"
                f"  .activation_max = {layer.activation_max},\n"
                "};"
            )
        elif isinstance(layer, Conv2dLayer):
            in_h, in_w, in_c = _spatial_dims(layer.input.shape)
            out_h, out_w, out_c_ = _spatial_dims(layer.output.shape)
            # weight tensor: OHWI [out_C, kH, kW, in_C]
            out_c, k_h, k_w, weight_in_c = layer.weight.shape
            bias_ref = f"L{i}_conv2d_bias" if layer.bias is not None else "(const int32_t*)0"
            packed_ref = (
                f"L{i}_conv2d_weight_packed_32"
                if layer.weight_packed_32 is not None else "(const int8_t*)0"
            )
            lines.append(
                f"static const Conv2dParams P_L{i}_conv2d = {{\n"
                f"  .in_h = {in_h}u, .in_w = {in_w}u, .in_c = {in_c}u,\n"
                f"  .out_h = {out_h}u, .out_w = {out_w}u, .out_c = {out_c}u,\n"
                f"  .kernel_h = {k_h}u, .kernel_w = {k_w}u,\n"
                f"  .stride_h = {layer.stride[0]}u, .stride_w = {layer.stride[1]}u,\n"
                f"  .pad_h = {layer.padding[0]}u, .pad_w = {layer.padding[1]}u,\n"
                f"  .weight = L{i}_conv2d_weight,\n"
                f"  .bias = {bias_ref},\n"
                f"  .requant_mults = L{i}_conv2d_requant_mults,\n"
                f"  .requant_shifts = L{i}_conv2d_requant_shifts,\n"
                f"  .input_offset = {layer.input_offset},\n"
                f"  .output_offset = {layer.output_offset},\n"
                f"  .activation_min = {layer.activation_min},\n"
                f"  .activation_max = {layer.activation_max},\n"
                f"  .weight_packed_32 = {packed_ref},\n"
                "};"
            )
        elif isinstance(layer, DepthwiseConv2dLayer):
            in_h, in_w, in_c = _spatial_dims(layer.input.shape)
            out_h, out_w, _ = _spatial_dims(layer.output.shape)
            _, k_h, k_w, out_c = layer.weight.shape
            bias_ref = f"L{i}_dwconv2d_bias" if layer.bias is not None else "(const int32_t*)0"
            bias_off_ref = (
                f"L{i}_dwconv2d_bias_off"
                if layer.bias_with_offset_full is not None else "(const int32_t*)0"
            )
            lines.append(
                f"static const DepthwiseConv2dParams P_L{i}_dwconv2d = {{\n"
                f"  .in_h = {in_h}u, .in_w = {in_w}u, .in_c = {in_c}u,\n"
                f"  .out_h = {out_h}u, .out_w = {out_w}u, .out_c = {out_c}u,\n"
                f"  .kernel_h = {k_h}u, .kernel_w = {k_w}u,\n"
                f"  .stride_h = {layer.stride[0]}u, .stride_w = {layer.stride[1]}u,\n"
                f"  .pad_h = {layer.padding[0]}u, .pad_w = {layer.padding[1]}u,\n"
                f"  .weight = L{i}_dwconv2d_weight,\n"
                f"  .bias = {bias_ref},\n"
                f"  .requant_mults = L{i}_dwconv2d_requant_mults,\n"
                f"  .requant_shifts = L{i}_dwconv2d_requant_shifts,\n"
                f"  .input_offset = {layer.input_offset},\n"
                f"  .output_offset = {layer.output_offset},\n"
                f"  .activation_min = {layer.activation_min},\n"
                f"  .activation_max = {layer.activation_max},\n"
                f"  .bias_with_offset_full = {bias_off_ref},\n"
                "};"
            )
        elif isinstance(layer, FusedConv2dDwconv2dLayer):
            in_h, in_w, in_c = _spatial_dims(layer.input.shape)
            # Output is at the dwconv's output spatial — derive from output slot.
            out_h, out_w, _ = _spatial_dims(layer.output.shape)
            # Expand weight: OHWI [expand_out_c, 1, 1, in_c]
            expand_out_c = layer.expand_weight.shape[0]
            # Dwconv weight: IHWO [1, kH, kW, expand_out_c]
            _, k_h, k_w, _ = layer.dw_weight.shape
            stem = f"L{i}_fused"
            e_bias_ref = f"{stem}_expand_bias" if layer.expand_bias is not None else "(const int32_t*)0"
            d_bias_ref = f"{stem}_dw_bias" if layer.dw_bias is not None else "(const int32_t*)0"
            d_bias_off_ref = (
                f"{stem}_dw_bias_off"
                if layer.dw_bias_with_offset_full is not None else "(const int32_t*)0"
            )
            lines.append(
                f"static const FusedConv2dDwconv2dParams P_L{i}_fused = {{\n"
                f"  .in_h = {in_h}u, .in_w = {in_w}u, .in_c = {in_c}u,\n"
                f"  .expand_out_c = {expand_out_c}u,\n"
                f"  .out_h = {out_h}u, .out_w = {out_w}u,\n"
                f"  .kernel_h = {k_h}u, .kernel_w = {k_w}u,\n"
                f"  .stride_h = {layer.dw_stride[0]}u, .stride_w = {layer.dw_stride[1]}u,\n"
                f"  .pad_h = {layer.dw_padding[0]}u, .pad_w = {layer.dw_padding[1]}u,\n"
                f"  .expand_weight = {stem}_expand_weight,\n"
                f"  .expand_bias = {e_bias_ref},\n"
                f"  .expand_requant_mults = {stem}_expand_requant_mults,\n"
                f"  .expand_requant_shifts = {stem}_expand_requant_shifts,\n"
                f"  .expand_input_offset = {layer.expand_input_offset},\n"
                f"  .expand_output_offset = {layer.expand_output_offset},\n"
                f"  .expand_activation_min = {layer.expand_activation_min},\n"
                f"  .expand_activation_max = {layer.expand_activation_max},\n"
                f"  .dw_weight = {stem}_dw_weight,\n"
                f"  .dw_bias = {d_bias_ref},\n"
                f"  .dw_bias_with_offset_full = {d_bias_off_ref},\n"
                f"  .dw_requant_mults = {stem}_dw_requant_mults,\n"
                f"  .dw_requant_shifts = {stem}_dw_requant_shifts,\n"
                f"  .dw_input_offset = {layer.dw_input_offset},\n"
                f"  .dw_output_offset = {layer.dw_output_offset},\n"
                f"  .dw_activation_min = {layer.dw_activation_min},\n"
                f"  .dw_activation_max = {layer.dw_activation_max},\n"
                "};"
            )
        elif isinstance(layer, QuantizedAddLayer):
            num = _prod(layer.output.shape)
            lines.append(
                f"static const QuantizedAddParams P_L{i}_add = {{\n"
                f"  .num_elements = {num}u,\n"
                f"  .self_zero_point = {layer.self_zero_point},\n"
                f"  .self_multiplier = {layer.self_multiplier},\n"
                f"  .self_shift = {layer.self_shift},\n"
                f"  .other_zero_point = {layer.other_zero_point},\n"
                f"  .other_multiplier = {layer.other_multiplier},\n"
                f"  .other_shift = {layer.other_shift},\n"
                f"  .output_zero_point = {layer.output_zero_point},\n"
                f"  .output_multiplier = {layer.output_multiplier},\n"
                f"  .output_shift = {layer.output_shift},\n"
                f"  .activation_min = {layer.activation_min},\n"
                f"  .activation_max = {layer.activation_max},\n"
                "};"
            )
        elif isinstance(layer, AvgPool2dLayer):
            in_h, in_w, channels = _spatial_dims(layer.input.shape)
            out_h, out_w, _ = _spatial_dims(layer.output.shape)
            lines.append(
                f"static const AvgPool2dParams P_L{i}_avgpool = {{\n"
                f"  .in_h = {in_h}u, .in_w = {in_w}u, .channels = {channels}u,\n"
                f"  .out_h = {out_h}u, .out_w = {out_w}u,\n"
                f"  .kernel_h = {layer.kernel_size[0]}u, .kernel_w = {layer.kernel_size[1]}u,\n"
                f"  .stride_h = {layer.stride[0]}u, .stride_w = {layer.stride[1]}u,\n"
                f"  .pad_h = {layer.padding[0]}u, .pad_w = {layer.padding[1]}u,\n"
                f"  .zero_point = {layer.zero_point},\n"
                f"  .multiplier = {layer.multiplier},\n"
                f"  .shift = {layer.shift},\n"
                "};"
            )
        # DequantOutputLayer is implicit — captured via MV2_OUTPUT_SCALE/ZERO_POINT.

    lines += ["", "#endif", ""]
    (out_dir / "mv2_params.h").write_text("\n".join(lines))


def emit_inference_body(out_dir: Path, schedule: ProgramSchedule) -> None:
    """Emit the body of mobilenet_v2_inference() as an #include-able header.

    Phase A/B/C use this to drive synthetic tests without touching the C
    kernels.  Phase D may hand-write a richer entry point that omits the
    generated body and inlines residual-add bookkeeping.
    """
    lines: list[str] = [
        "// Auto-generated. Do not edit.",
        "// Body of mobilenet_v2_inference(). Variables expected from caller:",
        "//   const float* input_float;  int8_t* output_q;",
        "//   extern uint8_t mv2_arena[];",
        "",
    ]
    output_int8_slot = None
    # Identify whether we route the final tensor straight into output_q (skips
    # the trailing dequant) or through the arena and let the caller dequant.
    last_q_layer = None
    for layer in schedule.layers:
        if isinstance(layer, (LinearLayer, Conv2dLayer, DepthwiseConv2dLayer,
                              FusedConv2dDwconv2dLayer,
                              AvgPool2dLayer, QuantizedAddLayer)):
            last_q_layer = layer
    if last_q_layer is not None:
        output_int8_slot = last_q_layer.output.name

    for i, layer in enumerate(schedule.layers):
        if isinstance(layer, QuantInputLayer):
            lines.append(
                f"  quantize_input(input_float, (int8_t*)(mv2_arena + ACT_OFFSET_{layer.output.name}), &P_L{i}_quant);"
            )
        elif isinstance(layer, Conv2dLayer):
            out_buf = (
                "output_q" if layer.output.name == output_int8_slot
                else f"(int8_t*)(mv2_arena + ACT_OFFSET_{layer.output.name})"
            )
            lines.append(
                f"  conv2d_s8((const int8_t*)(mv2_arena + ACT_OFFSET_{layer.input.name}), {out_buf}, &P_L{i}_conv2d);"
            )
        elif isinstance(layer, AvgPool2dLayer):
            out_buf = (
                "output_q" if layer.output.name == output_int8_slot
                else f"(int8_t*)(mv2_arena + ACT_OFFSET_{layer.output.name})"
            )
            lines.append(
                f"  avgpool_s8((const int8_t*)(mv2_arena + ACT_OFFSET_{layer.input.name}), {out_buf}, &P_L{i}_avgpool);"
            )
        elif isinstance(layer, DepthwiseConv2dLayer):
            out_buf = (
                "output_q" if layer.output.name == output_int8_slot
                else f"(int8_t*)(mv2_arena + ACT_OFFSET_{layer.output.name})"
            )
            lines.append(
                f"  dwconv2d_s8((const int8_t*)(mv2_arena + ACT_OFFSET_{layer.input.name}), {out_buf}, &P_L{i}_dwconv2d);"
            )
        elif isinstance(layer, FusedConv2dDwconv2dLayer):
            out_buf = (
                "output_q" if layer.output.name == output_int8_slot
                else f"(int8_t*)(mv2_arena + ACT_OFFSET_{layer.output.name})"
            )
            lines.append(
                f"  conv2d_dwconv2d_fused_s8("
                f"(const int8_t*)(mv2_arena + ACT_OFFSET_{layer.input.name}), "
                f"{out_buf}, &P_L{i}_fused);"
            )
        elif isinstance(layer, QuantizedAddLayer):
            out_buf = (
                "output_q" if layer.output.name == output_int8_slot
                else f"(int8_t*)(mv2_arena + ACT_OFFSET_{layer.output.name})"
            )
            lines.append(
                f"  add_s8((const int8_t*)(mv2_arena + ACT_OFFSET_{layer.self_in.name}), "
                f"(const int8_t*)(mv2_arena + ACT_OFFSET_{layer.other_in.name}), "
                f"{out_buf}, &P_L{i}_add);"
            )
        elif isinstance(layer, LinearLayer):
            out_buf = (
                "output_q" if layer.output.name == output_int8_slot
                else f"(int8_t*)(mv2_arena + ACT_OFFSET_{layer.output.name})"
            )
            lines.append(
                f"  gemv_s8((const int8_t*)(mv2_arena + ACT_OFFSET_{layer.input.name}), {out_buf}, &P_L{i}_linear);"
            )
        elif isinstance(layer, MemcpyLayer):
            # View / dim-order clone / identity-pad: bytes are unchanged.
            # If the planner aliased the slots (same offset) the memcpy is a
            # no-op overlap and we can skip emitting it.
            if layer.input.offset != layer.output.offset:
                lines.append(
                    f"  __builtin_memcpy("
                    f"(void*)(mv2_arena + ACT_OFFSET_{layer.output.name}), "
                    f"(const void*)(mv2_arena + ACT_OFFSET_{layer.input.name}), "
                    f"{layer.nbytes}u);"
                )
        # DequantOutputLayer skipped: int8 output is the final hand-off.
    lines.append("")
    (out_dir / "mv2_inference_body.h").write_text("\n".join(lines))


def emit_input_fixture(out_dir: Path, input_tensor: torch.Tensor, expected_top1: int = -1) -> None:
    lines = [
        "#ifndef MV2_INPUT_FIXTURE_H_",
        "#define MV2_INPUT_FIXTURE_H_",
        "",
        "#include <stdint.h>",
        "",
        f"#define MV2_FIXTURE_EXPECTED_TOP1 {expected_top1}",
        "",
        _emit_float_array("mv2_fixture_input", input_tensor),
        "",
        "#endif",
        "",
    ]
    (out_dir / "input_fixture.h").write_text("\n".join(lines))


def _prod(seq):
    p = 1
    for v in seq:
        p *= int(v)
    return p
