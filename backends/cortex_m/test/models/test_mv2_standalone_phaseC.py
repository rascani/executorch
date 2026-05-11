# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).
"""
Phase C: one MobileNetV2 inverted-residual block.

Expand 1x1 conv (8 -> 24) → depthwise 3x3 → project 1x1 (24 -> 8) +
residual add.  Validates dwconv2d_s8 and add_s8 plus the residual-hold
liveness in the AOT memory plan.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from executorch.backends.cortex_m.test.models._mv2_standalone_helpers import (
    run_standalone_inference,
)
from executorch.examples.models.mv2_cortex_m_mve.tools.dump_mv2_artifacts import dump


class InvertedResidualBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.expand = torch.nn.Conv2d(8, 24, kernel_size=1, bias=False)
        self.dw = torch.nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1,
                                  groups=24, bias=False)
        self.project = torch.nn.Conv2d(24, 8, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.expand(x)
        y = self.dw(y)
        y = self.project(y)
        return x + y


def _lower_phase_c(example_input: torch.Tensor):
    from executorch.backends.cortex_m.passes.cortex_m_pass_manager import (
        CortexMPassManager,
    )
    from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from torch.export import export
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    model = InvertedResidualBlock().eval()
    captured = export(model, (example_input,)).module()
    quantizer = CortexMQuantizer()
    quantizer.transform_for_annotation(captured)
    prepared = prepare_pt2e(captured, quantizer)
    for _ in range(8):
        prepared(torch.randn_like(example_input))
    quantized = convert_pt2e(prepared)
    exported = export(quantized, (example_input,))
    edge = to_edge_transform_and_lower(
        exported,
        compile_config=EdgeCompileConfig(
            preserve_ops=[torch.ops.aten.linear.default],
            _check_ir_validity=False,
        ),
    )
    return CortexMPassManager(
        edge.exported_program(), CortexMPassManager.pass_list
    ).transform()


@pytest.fixture(scope="module")
def cortex_m_available() -> bool:
    try:
        from executorch.backends.cortex_m.quantizer.quantizer import (  # noqa: F401
            CortexMQuantizer,
        )
    except ModuleNotFoundError:
        return False
    return True


@pytest.mark.skipif(
    shutil.which("cmake") is None,
    reason="cmake required for either host or FVP build",
)
def test_phase_c_inverted_residual(
    tmp_path: Path, cortex_m_available: bool
) -> None:
    if not cortex_m_available:
        pytest.skip("Cortex-M backend dependencies not installed")

    torch.manual_seed(0)
    example_input = torch.randn(1, 8, 8, 8).to(memory_format=torch.channels_last)
    program = _lower_phase_c(example_input)

    result = run_standalone_inference(
        program, example_input, tmp_path, dump_fn=dump,
    )
    backend = "FVP" if result.fvp_used else "host"
    print(
        f"Phase C {backend}: out[:8]={result.host_int8[:8]} "
        f"ref[:8]={result.ref_int8[:8]} diff={result.max_int8_diff}"
    )
    assert result.max_int8_diff <= 10, (
        f"Phase C: inference diverged from cortex_m runtime reference.\n"
        f"  ref_int8[:16] = {result.ref_int8[:16]}\n"
        f"  out_int8[:16] = {result.host_int8[:16]}\n"
        f"  max int8 diff = {result.max_int8_diff}"
    )
