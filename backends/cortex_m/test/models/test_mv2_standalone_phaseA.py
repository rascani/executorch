# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).
"""
Phase A: TinyLinear (8 -> 4) through quantize + cortex_m::quantized_linear
+ dequantize.  Validates the dumper, exir memory plan integration, FVP
build/run wiring, and `scalar_requantize` bit-exactness against the
cortex_m runtime reference.

When `arm-none-eabi-gcc` + `FVP_Corstone_SSE-300_Ethos-U55` are on PATH
(set up by `examples/arm/arm-scratch/setup_path.sh`), the test runs on
the Corstone-300 FVP; otherwise it falls back to a host build for
local-loop iteration.  Both paths assert the same `qtol` against the
cortex_m runtime reference.
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


class TinyLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _lower_tiny_linear(example_input: torch.Tensor):
    from executorch.backends.cortex_m.passes.cortex_m_pass_manager import (
        CortexMPassManager,
    )
    from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from torch.export import export
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    model = TinyLinear().eval()
    captured = export(model, (example_input,)).module()
    quantizer = CortexMQuantizer()
    quantizer.transform_for_annotation(captured)
    prepared = prepare_pt2e(captured, quantizer)
    for _ in range(8):
        prepared(torch.randn(1, 8))
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
def test_phase_a_tiny_linear(tmp_path: Path, cortex_m_available: bool) -> None:
    if not cortex_m_available:
        pytest.skip("Cortex-M backend dependencies not installed")

    torch.manual_seed(0)
    example_input = torch.randn(1, 8)
    program = _lower_tiny_linear(example_input)

    result = run_standalone_inference(
        program, example_input, tmp_path, dump_fn=dump,
    )
    print(
        f"Phase A {'FVP' if result.fvp_used else 'host'}: "
        f"out={result.host_int8} ref={result.ref_int8} diff={result.max_int8_diff}"
    )
    # Phase A is a single requantize step; bit-exact match required.
    assert result.max_int8_diff == 0, (
        f"Phase A: host inference diverged from cortex_m runtime reference.\n"
        f"  ref_int8  = {result.ref_int8}\n"
        f"  out_int8  = {result.host_int8}\n"
        f"  max diff  = {result.max_int8_diff}"
    )
