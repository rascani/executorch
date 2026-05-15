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
    parser.add_argument(
        "--width-mult", type=float, default=1.0,
        help="MobileNetV2 width multiplier (channel scaling). Non-1.0 widths "
             "have no torchvision pretrained weights, so --random-weights is "
             "implied when width != 1.0.",
    )
    parser.add_argument(
        "--input-size", type=int, default=224,
        help="Input spatial resolution (square). Default 224.",
    )
    parser.add_argument(
        "--fuse-expand-dwconv", action="store_true",
        help="Enable the MV2 expand+dwconv fusion AOT pass. Replaces every "
             "1x1 expand conv -> 3x3 depthwise conv pair with a single fused "
             "op that the runtime kernel streams via a 3-row rolling buffer, "
             "eliminating the full HxWxC_expand intermediate.",
    )
    parser.add_argument(
        "--fuse-inverted-residual", action="store_true",
        help="Extend --fuse-expand-dwconv to also absorb the project 1x1 "
             "conv and the optional residual add into one fused op per MV2 "
             "inverted-residual block. Eliminates both the dwconv-output and "
             "project-output intermediates from the arena. Implies "
             "--fuse-expand-dwconv.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    use_random = args.random_weights or args.width_mult != 1.0
    if use_random:
        model = models.mobilenet_v2(width_mult=args.width_mult, weights=None).eval()
    else:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).eval()

    R = args.input_size
    example_input = _imagenet_normalize(
        torch.rand(1, 3, R, R)
    ).to(memory_format=torch.channels_last)

    captured = export(model, (example_input,)).module()
    quantizer = CortexMQuantizer()
    quantizer.transform_for_annotation(captured)
    prepared = prepare_pt2e(captured, quantizer)

    print(f"Calibrating with {args.num_calibration} samples...")
    for i in range(args.num_calibration):
        cal_input = _imagenet_normalize(
            torch.rand(1, 3, R, R)
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

    fixture = example_input.permute(0, 2, 3, 1).contiguous().reshape(1, 1, -1)

    def _build_schedule(pass_list):
        """Re-run the post-edge pipeline with the given pass list and run
        the dump (which invokes the memory planner).  Returns the schedule
        and a reference float-output (computed from the pass-transformed
        program so that the int8 reference saved at the end matches the
        actual artifacts in the dump directory)."""
        prog = CortexMPassManager(
            edge.exported_program(), pass_list
        ).transform()
        ref = prog.module()(example_input)
        top1 = int(ref.argmax(dim=-1).item())
        return prog, ref, top1

    # When 3/4-op fusion is requested, run both that pipeline and the 2-op
    # fallback, then dump whichever has the smaller total (arena + fused
    # scratch).  This avoids the small-shape regression where the planner
    # finds zero arena reduction from the extra fusion and the scratch
    # addition is pure overhead.  Width-1.0 and width-0.5 invariably pick
    # ires; width-0.35 invariably picks 2-op.
    import shutil
    import tempfile
    candidate_pass_lists: list[tuple[str, list]] = []
    if args.fuse_inverted_residual:
        candidate_pass_lists = [
            ("inverted_residual", CortexMPassManager.pass_list_with_inverted_residual_fusion),
            ("expand_dwconv", CortexMPassManager.pass_list_with_expand_dwconv_fusion),
        ]
    elif args.fuse_expand_dwconv:
        candidate_pass_lists = [
            ("expand_dwconv", CortexMPassManager.pass_list_with_expand_dwconv_fusion),
        ]
    else:
        candidate_pass_lists = [("none", CortexMPassManager.pass_list)]

    best_total = None
    best_tag = None
    best_dump_dir: Path | None = None
    best_schedule = None
    best_ref_float = None
    best_top1 = None
    for tag, pl in candidate_pass_lists:
        prog, ref_float, top1 = _build_schedule(pl)
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"mv2_export_{tag}_"))
        sched = dump(prog, tmp_dir, input_fixture=fixture, expected_top1=top1)
        # Total = arena + fused scratch.
        scratch = 0
        arena_h = (tmp_dir / "mv2_arena.h").read_text()
        import re
        m = re.search(r"MV2_FUSED_SCRATCH_BYTES\s+(\d+)u", arena_h)
        if m is not None:
            scratch = int(m.group(1))
        total = sched.arena_bytes + scratch
        print(f"  candidate '{tag}': arena={sched.arena_bytes} scratch={scratch} total={total}")
        if best_total is None or total < best_total:
            if best_dump_dir is not None:
                shutil.rmtree(best_dump_dir, ignore_errors=True)
            best_total = total
            best_tag = tag
            best_dump_dir = tmp_dir
            best_schedule = sched
            best_ref_float = ref_float
            best_top1 = top1
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    assert best_dump_dir is not None
    print(f"Picked pass pipeline: {best_tag} (total bytes = {best_total})")

    # Move the winning artifacts into the user-requested out_dir.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for child in best_dump_dir.iterdir():
        dest = args.out_dir / child.name
        if dest.exists():
            if dest.is_file():
                dest.unlink()
            else:
                shutil.rmtree(dest)
        shutil.move(str(child), dest)
    shutil.rmtree(best_dump_dir, ignore_errors=True)

    schedule = best_schedule
    ref_float = best_ref_float
    top1 = best_top1
    print(f"Float top-1: {top1}")

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
