# Standalone KWT-1 inference on Cortex-M55 + Helium (MVE)

A hand-written C inference path for [KWT-1 / Keyword Spotting
Transformer](https://arxiv.org/abs/2104.00769)'s encoder block on
Cortex-M55 with the Helium (MVE) extension — **without** the ExecuTorch
C++ runtime, **without** CMSIS-NN, and **without** CMSIS-DSP.  Outputs
are bit-exact with the `cortex_m` backend's Python reference
(`backends/cortex_m/ops/operators.py`), which is the same reference
that gates the `test_implementation_kwt_1` runtime path.

KWT-1 is the first **transformer**-class model on the standalone
cortex_m path; the [`mv2_cortex_m_mve/`](../mv2_cortex_m_mve/) sibling
covers CNNs.  The kernel set this directory ships is therefore new
ground for the cortex_m backend as a whole:

| Kernel | Why CMSIS-NN couldn't ship it |
|---|---|
| `layer_norm_s8`         | CMSIS-NN has no float-internal LN |
| `gelu_lut_s8`           | CMSIS-NN has no GELU |
| `batch_matmul_s8`       | Used by the unfused QK^T / AV path |
| `softmax_s8`            | CMSIS-NN's `arm_softmax_s8` differs from torch in the low bits |
| `attention_fused_s8`    | Streaming fusion of QK^T → softmax → AV (no S×S score matrix materialized) |
| `add_s8`, `linear_s8`, `transpose_s8`, `kwt_1_quantize_input` | Standalone counterparts of the cortex_m ops |

Phase 0-10 status: full canonical KWT-1 architecture (`KWT1(seq_len=98,
mfcc_dim=40, d=64, d_ff=128, num_blocks=12, num_classes=35)`) lowered
→ MVE-tuned → validated bit-exact on host with the dumper's
`--full-kwt1 --seq-len 98` flag.  The `OneBlockKWT` unit used for
Phase 0-8 bring-up is the same module stacked 12× by `KWT1Encoder`.

### Architectural caveats

The standalone path implements a subset of the published KWT-1
architecture today; gaps that affect parity with the
ARM-software/keyword-transformer published checkpoints:

| Piece | Published KWT-1 | This directory |
|---|---|---|
| Sequence pooling | `[CLS]` token + slice | Mean over the sequence dim |
| Final pre-head LayerNorm | Yes | Skipped (CortexMQuantizer doesn't currently annotate it) |
| Learned positional encoding | Yes | Plumbed (`AddLayer.{self,other}_const`, see `kwt_1_kernels.c`) but disabled — PT2E flags an annotator warning when an Add has a parameter input |

Encoder body (12 stacked blocks with 1-head attention, d=64,
d_ff=128, GELU activation) matches bit-for-bit, so a published
checkpoint's encoder weights transfer cleanly via
`tools/load_arm_kwt1.py`.  The classification head was trained
against CLS-pooled features rather than mean-pooled, so it needs a
short fine-tune on Speech Commands v2 to recover most of the
published top-1.  See `tools/load_arm_kwt1.py`'s module docstring
for the full weight-mapping table.

See [`docs/BENCHMARK.md`](docs/BENCHMARK.md) for the cycle progression
through Phase 8 (1.81 M → 327 k PMU cycles, 5.52×, bit-exact at every
step) and [`docs/PLAN.md`](docs/PLAN.md) for the original 8-phase
implementation plan.

## Layout

```
kwt_1_cortex_m_mve/
  tools/                       # AOT artifact dumper (Python)
    dump_kwt_1_artifacts.py      entry point + OneBlockKWT model
    extractors.py                per-op extraction from the lowered graph
    emitters.py                  C array + struct + arena-offset + body
  include/                     # public headers
    kwt_1_inference.h            extern "C" kwt_1_inference(...)
    kwt_1_kernels.h              kernel forward declarations
    kwt_1_layer_params.h         per-kernel LayerParams struct typedefs
    kwt_1_profile.h              per-kernel cycle profile macros (opt-in)
  src/
    kwt_1_inference.c            entry point; #includes generated body
    kwt_1_kernels.c              hand-written kernels (scalar + MVE paths)
    arena.c                      static activation arena
    runner_host.c                stdin/stdout harness for host build
    runner_fvp.cpp               semihosted FVP runner
  fvp/
    toolchain-arm-none-eabi.cmake
    corstone-300.ld              customized linker script for the FVP
  generated/                   # gitignored; produced by the dumper
  docs/
    PLAN.md                      original phased implementation plan
    BENCHMARK.md                 cycle progression through Phase 8
```

## AOT pipeline

The dumper takes a `nn.Module`, runs it through the same
`prepare_pt2e` / `convert_pt2e` / `to_edge_transform_and_lower` /
`CortexMPassManager` pipeline that the runtime path uses, then walks
the post-lowering `ExportedProgram`'s graph in topological order and
emits per-layer artifacts:

| File | Content |
|---|---|
| `kwt_1_weights.{c,h}`     | `static const` int8 weights, int32 kernel-sums, float32 γ/β, int8 GELU LUTs |
| `kwt_1_params.h`          | `static const LayerParams P_L<i>_*` per layer |
| `kwt_1_arena.h`           | `KWT_1_ARENA_BYTES`, I/O sizes, `ACT_OFFSET_<tensor>` for every activation slot |
| `kwt_1_inference_body.h`  | The body of `kwt_1_inference()` — one kernel call per cortex_m op |
| `input_fixture.h`         | The float32 input used by the runner |
| `_ref_int8.bin` (opt.)    | The python-reference int8 output, for bit-exact validation |

Each kernel call site is wrapped in `KWT_1_PROFILE_BEGIN`/`END` macros
that compile to no-ops by default, or to PMU `CCNTR` snapshots when
the build is configured with `-DKWT_1_PROFILE=ON` — see
`docs/BENCHMARK.md` for the resulting per-kernel cycle table.

Determinism caveat: PT2E calibration is currently non-deterministic
across Python invocations even with `torch.manual_seed`, so the
recommended verification flow is to pass `--save-ref` to the dumper
and compare against the `_ref_int8.bin` it emits *in the same Python
invocation* that produced the headers.

## Build

Host (no MVE, for local correctness iteration):

```bash
python -m executorch.examples.models.kwt_1_cortex_m_mve.tools.dump_kwt_1_artifacts \
    --save-ref \
    examples/models/kwt_1_cortex_m_mve/generated/
cmake -S examples/models/kwt_1_cortex_m_mve -B build/host
cmake --build build/host
./build/host/kwt_1_runner_host < <(python -c 'import sys, struct
nums = open("examples/models/kwt_1_cortex_m_mve/generated/input_fixture.h").read()
# parse the fixture as needed; the verification driver in tools/ does this for you
') > output.bin
```

FVP (Corstone-300, Cortex-M55 + Helium MVE):

```bash
source examples/arm/arm-scratch/setup_path.sh
cmake -S examples/models/kwt_1_cortex_m_mve -B build/fvp \
      -DCMAKE_TOOLCHAIN_FILE=examples/models/kwt_1_cortex_m_mve/fvp/toolchain-arm-none-eabi.cmake \
      -DKWT_1_BUILD_FVP=ON
cmake --build build/fvp
FVP_Corstone_SSE-300_Ethos-U55 \
    -C ethosu.num_macs=128 \
    -C mps3_board.visualisation.disable-visualisation=1 \
    -C mps3_board.telnetterminal0.start_telnet=0 \
    -C mps3_board.uart0.out_file=- \
    -C mps3_board.uart0.shutdown_on_eot=1 \
    -C cpu0.semihosting-enable=1 \
    -C cpu0.semihosting-stack_base=0 \
    -C cpu0.semihosting-heap_limit=0 \
    -a build/fvp/kwt_1_runner_fvp.elf \
    --timelimit 300
```

`Ethos-U55` is in the FVP binary name but the NPU is not used — the
runner only touches the Cortex-M55 + Helium pipeline.  Add
`-DKWT_1_PROFILE=ON` to either build for per-kernel cycle accounting
(the FVP runner prints a `PROFILE_BEGIN` … `PROFILE_END` block before
the `RESULT_*` markers).

## Tests

Phased pytest suite under `backends/cortex_m/test/models/` exercises
each phase end-to-end and asserts parity with the cortex_m runtime
reference; tests prefer the FVP path when `arm-none-eabi-gcc` and
`FVP_Corstone_SSE-300_Ethos-U55` are on PATH, and fall back to host
otherwise.  Phase tests track the rollout in `docs/PLAN.md`:

- Phase 1 — `layer_norm_s8`
- Phase 2 — `gelu_lut_s8`
- Phase 3 — `batch_matmul_s8`
- Phase 4 — `softmax_s8`
- Phase 5 — fused attention
- Phase 6 — `add_s8`, `linear_s8`
- Phase 7 — codegen (extractors + emitters + inference body)
- Phase 8 — MVE tuning + per-kernel profiling

Authored with assistance from Claude (claude.ai/code).
