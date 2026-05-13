#!/usr/bin/env python3
# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).
"""
Export MobileNetV2 through the Cortex-M quantization pipeline and dump
C artifacts for the standalone Cortex-M55 + MVE inference runner.

Usage:
    python -m examples.models.mv2_cortex_m_mve.tools.export_mv2 /tmp/mv2_full_dump
    python -m examples.models.mv2_cortex_m_mve.tools.export_mv2 /tmp/mv2_full_dump --random-weights
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision import models  # type: ignore[import-untyped]

from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from .dump_mv2_artifacts import dump


def _imagenet_normalize(t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (t - mean) / std


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MV2 for Cortex-M55 MVE")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--random-weights", action="store_true",
        help="Use random weights instead of pretrained (for testing only)",
    )
    parser.add_argument(
        "--num-calibration", type=int, default=100,
        help="Number of calibration samples.  Defaults to 100 — fewer (32)"
             " has been observed to produce a degenerate output scale of 1.19e-07"
             " (single-precision float epsilon) on MV2, causing every output"
             " logit to saturate at int8 -128.  Matches the cortex_m unit"
             " test's calibration count for parity.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.random_weights:
        model = models.mobilenet_v2(weights=None).eval()
    else:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).eval()

    example_input = _imagenet_normalize(
        torch.rand(1, 3, 224, 224)
    ).to(memory_format=torch.channels_last)

    captured = export(model, (example_input,)).module()
    quantizer = CortexMQuantizer()
    quantizer.transform_for_annotation(captured)
    prepared = prepare_pt2e(captured, quantizer)

    print(f"Calibrating with {args.num_calibration} samples...")
    for i in range(args.num_calibration):
        cal_input = _imagenet_normalize(
            torch.rand(1, 3, 224, 224)
        ).to(memory_format=torch.channels_last)
        prepared(cal_input)

    quantized = convert_pt2e(prepared)
    exported = export(quantized, (example_input,))

    edge = to_edge_transform_and_lower(
        exported,
        compile_config=EdgeCompileConfig(
            preserve_ops=[torch.ops.aten.linear.default],
            _check_ir_validity=False,
        ),
    )
    program = CortexMPassManager(
        edge.exported_program(), CortexMPassManager.pass_list
    ).transform()

    fixture = example_input.permute(0, 2, 3, 1).contiguous().reshape(1, 1, -1)

    ref_float = program.module()(example_input)
    top1 = int(ref_float.argmax(dim=-1).item())
    print(f"Float top-1: {top1}")

    schedule = dump(program, args.out_dir, input_fixture=fixture, expected_top1=top1)

    scale = schedule.output_scale
    zp = schedule.output_zero_point
    print(f"Output quant: scale={scale}, zp={zp}")

    ref_flat = ref_float.flatten()
    ref_int8 = torch.clamp(torch.round(ref_flat / scale) + zp, -128, 127).to(torch.int32)
    unique = ref_int8.unique()
    print(f"Reference int8: {len(unique)} unique values, range [{ref_int8.min().item()}, {ref_int8.max().item()}]")

    ref_path = args.out_dir / "_ref_int8.json"
    ref_path.write_text(json.dumps({
        "ref_int8": ref_int8.tolist(),
        "scale": scale,
        "zp": zp,
    }))
    print(f"Artifacts written to {args.out_dir}")


if __name__ == "__main__":
    main()
