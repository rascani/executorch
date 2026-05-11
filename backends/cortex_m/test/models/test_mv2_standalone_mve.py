# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).
"""
Phase D end-to-end test: full torchvision MobileNetV2 through the standalone
Cortex-M55 + Helium MVE inference path.

The model is lowered with the same CortexMQuantizer + CortexMPassManager
pipeline used by the existing test_implementation_mv2 (currently xfailed),
the dumper emits per-layer params + weights into a fresh build dir, the
runner is compiled, the int8 output is collected, and compared against
the cortex_m runtime reference within `qtol=10` — the same tolerance the
existing implementation test targets.

When `arm-none-eabi-gcc` + `FVP_Corstone_SSE-300_Ethos-U55` are on PATH
(via `examples/arm/arm-scratch/setup_path.sh`), the test runs on the
Corstone-300 FVP; otherwise it falls back to a host build.

The test uses `weights=None` (random init) so it runs offline; the
parity check against the runtime reference is unaffected.  Top-1
matching against MobileNet_V2_Weights.DEFAULT plus a real ImageNet
fixture is a follow-up.
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


def _lower_mobilenet_v2(example_input: torch.Tensor):
    from executorch.backends.cortex_m.passes.cortex_m_pass_manager import (
        CortexMPassManager,
    )
    from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from torch.export import export
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    from torchvision import models  # type: ignore[import-untyped]

    model = models.mobilenet_v2(weights=None).eval()
    captured = export(model, (example_input,)).module()
    quantizer = CortexMQuantizer()
    quantizer.transform_for_annotation(captured)
    prepared = prepare_pt2e(captured, quantizer)
    for _ in range(20):
        prepared(
            torch.randn(1, 3, 224, 224).to(memory_format=torch.channels_last)
        )
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
def test_phase_d_full_mobilenet_v2(
    tmp_path: Path, cortex_m_available: bool
) -> None:
    if not cortex_m_available:
        pytest.skip("Cortex-M backend dependencies not installed")

    torch.manual_seed(0)
    example_input = torch.randn(1, 3, 224, 224).to(memory_format=torch.channels_last)
    program = _lower_mobilenet_v2(example_input)

    # Full MV2 on FVP can take several minutes of wall-clock simulation time.
    result = run_standalone_inference(
        program, example_input, tmp_path, dump_fn=dump, fvp_timeout_s=1800,
    )
    backend = "FVP" if result.fvp_used else "host"
    print(
        f"Phase D {backend}: "
        f"ref[:8]={result.ref_int8[:8]} out[:8]={result.host_int8[:8]} "
        f"diff={result.max_int8_diff}"
    )
    assert result.max_int8_diff <= 10, (
        f"Phase D: inference diverged from cortex_m runtime reference.\n"
        f"  max int8 diff = {result.max_int8_diff}\n"
        f"  ref[:16] = {result.ref_int8[:16]}\n"
        f"  out[:16] = {result.host_int8[:16]}"
    )
