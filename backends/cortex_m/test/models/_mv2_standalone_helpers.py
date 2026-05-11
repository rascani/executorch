# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).
"""
Shared helpers for the test_mv2_standalone_* pytest suite.

The same lower → dump → build → run → compare pipeline is exercised by
Phase A/B/C/D tests; the only thing that varies is the model and example
input.  Each test calls `run_standalone_inference(...)` which returns
the int8 outputs from the host runner or the Corstone-300 FVP and
compares them against the cortex_m runtime reference within `qtol`.

The FVP target is preferred when `arm-none-eabi-gcc` and
`FVP_Corstone_SSE-300_Ethos-U55` are on PATH (set up via
`examples/arm/arm-scratch/setup_path.sh`).  Falls back to host-build
otherwise.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.export import ExportedProgram

REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_DIR = REPO_ROOT / "examples" / "models" / "mv2_cortex_m_mve"
if not PROJECT_DIR.exists():
    PROJECT_DIR = REPO_ROOT / "executorch" / "examples" / "models" / "mv2_cortex_m_mve"
SCRATCH_DIR = REPO_ROOT / "examples" / "arm" / "arm-scratch"
TOOLCHAIN_FILE = PROJECT_DIR / "fvp" / "toolchain-arm-none-eabi.cmake"
FVP_BIN = "FVP_Corstone_SSE-300_Ethos-U55"


def _add_arm_paths_to_env() -> dict:
    """Return os.environ + the arm-scratch bin directories on PATH."""
    env = dict(os.environ)
    extra_paths: list[str] = []
    fvp_dir = SCRATCH_DIR / "FVP-corstone300" / "models" / "Linux64_GCC-9.3"
    gcc_dir = SCRATCH_DIR / "arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi" / "bin"
    for p in (fvp_dir, gcc_dir):
        if p.is_dir():
            extra_paths.append(str(p))
    if extra_paths:
        env["PATH"] = os.pathsep.join(extra_paths + [env.get("PATH", "")])
    return env


def fvp_available() -> bool:
    env = _add_arm_paths_to_env()
    return (
        shutil.which("arm-none-eabi-gcc", path=env["PATH"]) is not None
        and shutil.which(FVP_BIN, path=env["PATH"]) is not None
        and TOOLCHAIN_FILE.is_file()
    )


@dataclass
class StandaloneRun:
    fvp_used: bool
    host_int8: list[int]
    ref_int8: list[int]
    max_int8_diff: int


def _flatten_input(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        return x.permute(0, 2, 3, 1).contiguous().reshape(-1)
    return x.flatten()


def _parse_arena(generated_dir: Path) -> tuple[float, int, int]:
    text = (generated_dir / "mv2_arena.h").read_text()
    scale = float(re.search(r"MV2_OUTPUT_SCALE ([0-9.eE+-]+)f?", text).group(1))
    zp = int(re.search(r"MV2_OUTPUT_ZERO_POINT (-?\d+)", text).group(1))
    n = int(re.search(r"MV2_OUTPUT_NUM_ELEMENTS (\d+)u", text).group(1))
    return scale, zp, n


def _reference_int8(
    program: ExportedProgram, example: torch.Tensor, scale: float, zp: int
) -> torch.Tensor:
    ref_float = program.module()(example)
    if ref_float.dim() == 4:
        ref_float = ref_float.permute(0, 2, 3, 1).contiguous().reshape(-1)
    else:
        ref_float = ref_float.flatten()
    return torch.clamp(torch.round(ref_float / scale) + zp, -128, 127).to(torch.int32)


def _build_host(generated_dir: Path, build_dir: Path) -> Path:
    env = _add_arm_paths_to_env()
    subprocess.run(
        [
            "cmake",
            "-S", str(PROJECT_DIR),
            "-B", str(build_dir),
            f"-DMV2_GENERATED_DIR={generated_dir}",
        ],
        check=True,
        env={**env, "CMAKE_GENERATOR": "Unix Makefiles"},
        capture_output=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "-j"],
        check=True, capture_output=True, env=env,
    )
    return build_dir / "mv2_runner_host"


def _build_fvp(generated_dir: Path, build_dir: Path) -> Path:
    env = _add_arm_paths_to_env()
    subprocess.run(
        [
            "cmake",
            "-S", str(PROJECT_DIR),
            "-B", str(build_dir),
            f"-DCMAKE_TOOLCHAIN_FILE={TOOLCHAIN_FILE}",
            "-DMV2_BUILD_FVP=ON",
            f"-DMV2_GENERATED_DIR={generated_dir}",
        ],
        check=True, env={**env, "CMAKE_GENERATOR": "Unix Makefiles"},
        capture_output=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "-j"],
        check=True, capture_output=True, env=env,
    )
    return build_dir / "mv2_runner_fvp.elf"


def _run_host_binary(binary: Path, input_flat: torch.Tensor, num_outputs: int) -> list[int]:
    payload = b"".join(struct.pack("<f", float(v)) for v in input_flat.tolist())
    result = subprocess.run(
        [str(binary)], input=payload, capture_output=True, check=True, timeout=600
    )
    return list(struct.unpack(f"<{num_outputs}b", result.stdout))


def _run_fvp(elf: Path, num_outputs: int, timeout_s: int) -> list[int]:
    env = _add_arm_paths_to_env()
    proc = subprocess.run(
        [
            FVP_BIN,
            "-C", "ethosu.num_macs=128",
            "-C", "mps3_board.visualisation.disable-visualisation=1",
            "-C", "mps3_board.telnetterminal0.start_telnet=0",
            "-C", "mps3_board.uart0.out_file=-",
            "-C", "mps3_board.uart0.shutdown_on_eot=1",
            "-C", "cpu0.semihosting-enable=1",
            "-C", "cpu0.semihosting-stack_base=0",
            "-C", "cpu0.semihosting-heap_limit=0",
            "-a", str(elf),
            "--timelimit", str(timeout_s),
        ],
        capture_output=True, text=True, check=False, env=env, timeout=timeout_s + 30,
    )
    out = proc.stdout
    # The semihosted runner emits the int8 logits between RESULT_BEGIN / RESULT_END
    # markers, one decimal value per line.
    m = re.search(r"RESULT_BEGIN\s*(.*?)RESULT_END", out, re.DOTALL)
    if not m:
        raise RuntimeError(
            f"FVP run did not emit RESULT_BEGIN/RESULT_END.\nstdout tail:\n{out[-2000:]}"
        )
    values: list[int] = []
    for line in m.group(1).strip().splitlines():
        line = line.strip()
        if not line:
            continue
        values.append(int(line))
    if len(values) != num_outputs:
        raise RuntimeError(
            f"FVP returned {len(values)} values, expected {num_outputs}"
        )
    return values


def run_standalone_inference(
    program: ExportedProgram,
    example_input: torch.Tensor,
    tmp_path: Path,
    dump_fn: Callable[..., object],
    qtol: int = 10,
    prefer_fvp: bool = True,
    fvp_timeout_s: int = 900,
) -> StandaloneRun:
    """Lower-already-done → dump → build → run → compare.

    `dump_fn` is the dump callable from
    examples.models.mv2_cortex_m_mve.tools.dump_mv2_artifacts.dump
    (caller passes it explicitly so the test side controls fixture
    handling).  Returns a StandaloneRun the caller can assert on.
    """
    generated_dir = tmp_path / "generated"
    fixture = _flatten_input(example_input).reshape(1, 1, -1)
    dump_fn(program, generated_dir, input_fixture=fixture)

    scale, zp, num_out = _parse_arena(generated_dir)
    ref_int8 = _reference_int8(program, example_input, scale, zp)

    use_fvp = prefer_fvp and fvp_available()
    if use_fvp:
        elf = _build_fvp(generated_dir, tmp_path / "build_fvp")
        out = _run_fvp(elf, num_out, fvp_timeout_s)
    else:
        bin_path = _build_host(generated_dir, tmp_path / "build_host")
        out = _run_host_binary(bin_path, _flatten_input(example_input), num_out)

    diff = (ref_int8 - torch.tensor(out, dtype=torch.int32)).abs().max().item()
    return StandaloneRun(
        fvp_used=use_fvp,
        host_int8=out,
        ref_int8=ref_int8.tolist(),
        max_int8_diff=int(diff),
    )
