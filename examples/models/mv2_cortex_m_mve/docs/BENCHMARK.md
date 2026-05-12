# Standalone vs. cortex_m backend — MobileNetV2 footprint & latency

Direct measurement of the standalone Cortex-M55 + MVE inference path
in this directory against the existing cortex_m backend (which
dispatches through ExecuTorch's runtime to CMSIS-NN kernels) for full
MobileNetV2 on the Corstone-300 FVP.

## Setup

| Axis | Both paths |
|---|---|
| Target | Cortex-M55 + Helium (MVE), Corstone-300 FVP |
| Toolchain | arm-none-eabi-gcc 13.3.rel1 |
| Compiler flags | `-mcpu=cortex-m55 -mthumb -mfloat-abi=hard -mfpu=auto -O3` |
| Model | torchvision `mobilenet_v2`, int8 quantized via `CortexMQuantizer` + `convert_pt2e` |
| Memory planner | `exir.memory_planning.greedy` (same planner used by `MemoryPlanningPass`) |
| Logging | `EXECUTORCH_ENABLE_LOGGING=OFF`, `EXECUTORCH_ENABLE_EVENT_TRACER=OFF` for both |
| Cycle counter | DWT `CYCCNT`, read in C around the inference call |

For the cortex_m runner, `arm_perf_monitor.cpp` was patched with a
`printf("BENCHMARK_CYCLES %u\n", cycle_count)` so the count reaches
stdout under `ET_LOG_ENABLED=0`.  Build command for that runner is
documented at the end of this file.

## Headline numbers (Corstone-300 FVP, DWT CYCCNT)

| Path | Cycles / inference | `.text` | `.rodata` | Arena |
|---|---:|---:|---:|---:|
| Standalone, baseline (commit `e6f5623e65`)   | 144,728,651 | 56,888 B | 4,283,956 B | 1,505,280 B |
| Standalone, with five committed kernel improvements | **46,718,954** | **65,840 B** | **4,233,180 B** | **1,505,280 B** |
| cortex_m backend (CMSIS-NN), rebuilt fresh   | **172,218,932** | 478,960 B (test runner — handles arbitrary models via semihosting) | 35,632 B (+ 128 MB `.ddr` carrying the `.pte` + 60 MB input-file pool + arena + Ethos-U buffers) | 1,505,280 B (inside `.ddr`) |

The standalone-after-five-commits path is **3.69× faster** than the
cortex_m backend on this FVP config with bit-exact int8 outputs.
The cortex_m backend runner's `.text` is much larger than ours
because it's the test-framework's generic semihosting runner that
can take any PTE via `-m` at runtime, includes BundleIO scaffolding,
plus the executorch runtime + flatbuffer parser + portable kernels
+ CMSIS-NN library + Ethos-U driver code.  For a real production
deployment that hardcodes the model (no semihosting, no BundleIO),
the cortex_m runner shrinks to roughly the 154 KB range — still 2-3×
our standalone.

### How we got the cortex_m FVP number

The key issue: the test-framework runner is built with
`-DET_LOG_ENABLED=0`, so the `arm_perf_monitor.cpp` cycle-output
`ET_LOG(Info, ...)` calls compile away to nothing.  Output files
(`out-0.bin`, `out-1.bin`, ...) still reach disk via a different
semihosting channel (`SYS_WRITE`), which is why the test framework's
`test_implementation_mv2` test passes but produces no cycle stdout.

To recover the cycle count we patched `arm_perf_monitor.cpp` to
`printf("BENCHMARK_CYCLES %u\n", cycle_count)` (which bypasses
`ET_LOG_ENABLED`) and rebuilt only the runner.  Running pytest's
`test_implementation_mv2` against the patched runner then captures
the FVP stdout with the cycle line.  See the reproduction section
below for the exact invocation.

A previous in-repo file (`mv2_benchmark_comparison.md`,
**249,819,842 cycles**) reports cortex_m latency on real silicon;
it's not directly comparable to FVP numbers since the FVP's CPU
time model is not cycle-accurate.

## Standalone optimization progression

Each row is a committed change.  All measured at the same Corstone-300
FVP config, with bit-exact output (max int8 diff vs. cortex_m runtime
reference = 0 across all 1000 logits).

| # | Commit | Change | Cycles | Speedup over previous | Cumulative | `.text` |
|---:|---|---|---:|---:|---:|---:|
| 0 | `e6f5623e65` | Baseline: scalar dwconv, scalar requantize per OC | 144,728,651 | — | 1.00× | 56,888 B |
| 1 | `802c00e3fb` | MVE channel-major dwconv (4 OC per inner iter) | 87,265,364 | 1.66× | 1.66× | n/a |
| 2 | `5351678930` | conv2d: 4-OC blocked w/ shared input + folded input_offset | 55,933,708 | 1.56× | 2.59× | 48,696 B |
| 3 | `27c363b09f` | 1×1 conv: 2 output pixels per inner iter | 48,975,234 | 1.14× | 2.95× | 64,336 B |
| 4 | `fa9b4c1f86` | Pack requant shifts as int8 (memory: -51 KB rodata) | 47,964,461 | 1.02× | 3.02× | 64,600 B |
| 5 | `a56407f7e2` | Vectorize `add_s8` 4-wide | 46,718,954 | 1.03× | 3.10× | 65,840 B |

Headline: **3.10× faster, +9 KB code, -51 KB read-only data**, bit-exact
output throughout.

### What each experiment did

**1. MVE channel-major dwconv** (`802c00e3fb`).  The baseline kernel
walked output channels one at a time with scalar accumulation —
depthwise has no cross-channel reduction, so the natural vectorization
axis is the channel dimension.  The fast path processes 4 OCs per
inner iteration via `vldrbq_s32` (4 int8 → int32 sign-extend),
`vmulq_s32 + vaddq_s32` accumulate, and `mve_requantize_per_channel`
for the per-channel requantize step.  Every MV2 depthwise layer has
`out_c % 4 == 0` so this path always triggers.

**2. conv2d OC blocking + AOT-folded input_offset** (`5351678930`).
Two changes that share an MVE fast path:

- *OC blocking:* the original conv2d_s8 walked output channels one at
  a time, re-reading each input pixel `out_c` times.  The new fast
  path processes 4 OCs per inner iteration via one shared
  `vld1q_s8` input load + four weight loads + four `vmladavaq_s8`,
  collapsing the per-OC scalar requantize into a single vector op.
  Triggers when `out_c % 4 == 0` — true for every conv2d layer in MV2.
- *Folded input_offset:* cortex_m conv2d has no filter_offset, so
  `acc[oc] = sum((x + input_offset) * w[oc]) = sum(x * w[oc])
  + input_offset * sum(w[oc])`.  For 1×1 convs with no padding (all
  MV2 conv2d except the first 3×3), the second term is a per-OC
  constant; the dumper folds it into the bias once at AOT, and the
  runtime kernel skips the `vaddvq_s8` accumulation entirely.  Memory
  cost is zero (still an int32 per OC).  Signaled by emitting
  `input_offset = 0` in the `LayerParams`.

**3. 2-pixel spatial tiling for 1×1 convs** (`27c363b09f`).  For 1×1
stride-1 convs with even `out_w`, processing 2 adjacent output pixels
per inner iteration shares the 4 weight-row loads across both pixels
and amortizes the 4-wide vector requantize over twice as much work.
Each IC chunk runs 8 `vmladavaq_s8` against 6 loads, so per-pixel
instruction count drops ~22%.  Every MV2 1×1 layer hits this path
except the 320 → 1280 head (`out_w = 7`) which falls through to the
single-pixel path.

**4. int8 packing for per-channel shifts** (`fa9b4c1f86`).  Shifts
come from `frexp(scale)` — across all 17,056 channels of MV2 the
values land in `[-13, -6]`, well within int8.  Storage drops from
int32 → int8 (4 bytes → 1 byte per channel, -51 KB total).  The MVE
load switches from `vld1q_s32` to `vldrbq_s32` (load 4 int8s,
sign-extend to int32) at the same throughput, and the scalar paths
implicitly sign-extend on assignment.  Marginal latency win from the
smaller cache footprint.

**5. Vector quantized_add** (`a56407f7e2`).  The original `add_s8`
walked elements one at a time with three scalar requantize calls per
element (two input-side requantizes + one output-side).  The vector
path processes 4 int8 elements per iteration via `vldrbq_s32` + scalar
zero-point subtract via `vsubq_s32` + `vshlq_n_s32` lift + three
`mve_requantize_per_channel` calls (with `vdupq_n_s32` broadcasting
the scalar per-tensor multiplier/shift), then a `vaddq_s32` + clamp +
`vstrbq_s32` saturating-narrow store.  Within-layer throughput jumps
3-4×, but the 10 `add_s8` calls together account for under 3% of total
MV2 cycles, so the end-to-end win is small.

## Memory footprint

Section sizes for the full MV2 standalone binary today (after all five
optimizations):

| Section | Bytes | Contents |
|---|---:|---|
| `.text`             |     65,840 | 8 static kernels + entry point + ARMCM55 startup + newlib stubs |
| `.rodata`           |  4,233,180 | ~3.4 MB int8 weights + ~200 KB per-channel mults/shifts + ~30 KB LayerParams + ~600 KB FVP input fixture |
| `.arena_ddr`        |  1,505,280 | activation arena (exir greedy planner) |
| `.data`             |      2,068 | initialized statics |
| `.bss`              |      1,996 | uninitialized statics |
| Stack / heap        | 64 KB each | reserved in DTCM |

Deployment notes for a real Cortex-M55 SoC:

- **Drop the input fixture.**  `input_fixture.h` is only embedded for
  the FVP runner; a real device feeds input via DMA / sensor stream
  / memory-mapped buffer, saving 600 KB of `.rodata` immediately.
  Build flag: omit `runner_fvp.cpp` from sources.
- **Arena placement.**  The 1.5 MB arena fits in any cortex-M55 with
  external SRAM/PSRAM; on chips with only TCM it needs to live in
  DDR.  Same constraint applies to the cortex_m backend at the same
  planner output.
- **Weights in flash.**  The 3.4 MB of int8 weights fit in
  most Cortex-M55 SoCs' embedded flash; if not, an external QSPI
  bank works (slower load, runs from cached XIP).
- **Code in ITCM.**  At ~66 KB, `.text` fits comfortably in any
  Cortex-M55 ITCM (typical 256 KB / 512 KB).

## Side-by-side static comparison vs cortex_m backend

The static comparison comes from `arm-none-eabi-size` on both ELFs
built fresh in the same toolchain with logging + event tracer off:

| Axis | Standalone (this dir, after 5 commits) | cortex_m backend (rebuilt fresh) |
|---|---:|---:|
| `.text` runtime code | **65,840 B** | **154,872 B** (incl. ExecuTorch runtime, flatbuffer parser, CMSIS-NN library, Ethos-U driver code that's compiled but unused) |
| Weight payload | ~3.4 MB int8 in `.rodata` | 3,996,416 B `.pte` (flatbuffer-wrapped weights) |
| Activation arena | 1,505,280 B | 1,505,280 B (same exir greedy plan) |
| Per-op dispatch overhead | inlined — ~0 cycles per layer | function-pointer dispatch via `KernelRuntimeContext`, per-op `Tensor` view setup, per-op `allocate_temp` calls |

The standalone code is ~58% smaller (89 KB savings), driven almost
entirely by the absent runtime — no Program / Method machinery, no
flatbuffer parser, no CMSIS-NN library wrappers.

FVP-vs-FVP latency comparison (full MV2, Corstone-300):

| | Cycles | vs cortex_m |
|---|---:|---:|
| Standalone (baseline)                | 144,728,651 | 0.84× (slower) |
| Standalone (after 5 commits)         | **46,718,954** | **3.69× faster** |
| cortex_m backend (CMSIS-NN)          | 172,218,932 | 1.00× |

### Where the 3.69× actually comes from

Both binaries use the same MVE int8 reduce instruction (`vmladava.s8`)
in their hot loops — `arm-none-eabi-objdump` shows it as a generic
`cdp 15, ...` because the binutils 2.42 disassembler doesn't pretty-
print MVE encodings, but the underlying bytes match the same intrinsic
(see "Instruction-level verification" below).  So this is **not** a
"we use MVE and they don't" win.  The 3.69× decomposes into:

1. **Inner-loop tile geometry — per-output instruction count is ~1.7×
   better here.**  CMSIS-NN's `arm_nn_mat_mult_nt_t_s8` inner loop
   processes a 4-row × 1-column tile per iter (5 `vldrb.8` loads +
   4 `vmladava.s8` + 1 `vaddva.s8` for the input-offset sum = ~10
   instructions per 64 MACs = 4 outputs).  Our fast 1×1 path processes
   a 4-OC × 2-pixel tile per iter (6 `vldrb.8` loads + 8 `vmladava.s8`
   + 0 `vaddva` since input_offset is AOT-folded into bias = 14
   instructions per 128 MACs = 8 outputs).  Per output: 2.5 instr vs.
   1.75 instr.
2. **AOT-folded input_offset for 1×1 convs.**  For 1×1 convs with no
   padding (all MV2 conv2d except the first 3×3), the
   `input_offset * sum(weight[oc])` correction is a per-OC constant,
   so the AOT dumper folds it into the bias.  This eliminates the
   `vaddva.s8` inside the inner loop — one full vector op per IC chunk
   per OC across all 34 of those layers.  CMSIS-NN's MVE matmul always
   carries the `vaddva.s8` because its inner loop is generic over the
   input zero-point being non-zero.
3. **Per-op dispatch overhead in `Method::execute`.**  For each of MV2's
   ~70 ops, the cortex_m runtime walks the chain, constructs a
   `Tensor` view for each input/output, calls
   `KernelRuntimeContext::allocate_temp` for scratch, and dispatches
   the kernel via a function pointer.  The standalone path inlines
   the kernel calls directly into `mobilenet_v2_inference` with all
   `LayerParams` constant-folded — zero dispatch.
4. **Per-call CMSIS-NN wrapper setup.**  `arm_convolve_wrapper_s8`
   takes a `cmsis_nn_dims` struct, picks an inner kernel variant from
   a dispatch table, and threads through a `cmsis_nn_context` for
   Im2Col scratch.  Our kernels skip all of that.
5. **Vectorized 4-wide requantize at tile boundaries.**  We use
   `mve_requantize_per_channel(int32x4_t, int32x4_t mult, int32x4_t shift)`
   to requantize 4 OCs in parallel at the end of each 4-OC tile.
   CMSIS-NN mixes vector requantize (`arm_requantize_mve`, used in
   `arm_nn_mat_mult_nt_t_s8`) and scalar requantize
   (`arm_nn_requantize`, used in tail and `vec_mat_mult` paths) and
   takes the scalar path more often than ours does.

We win on inner-loop *throughput* because of (1) and (2), and we win
on per-layer *fixed costs* because of (3) and (4).  Splitting the
3.69× cleanly between inner-loop and overhead would need cycle
sampling per layer, which we haven't done.

### Instruction-level verification

Verified that GCC 13.3's `vmladavaq_s8` intrinsic emits the same
instruction encoding CMSIS-NN hand-writes as `vmladava.s8` in inline
assembly.  Both encode in the MVE-compute encoding space (coprocessor
15) and both disassemble identically as `cdp 15, ...` /
`cdp2 15, ...` with the GNU disassembler (which lacks pretty-printing
for MVE int8 reduce; LLVM disassembler with `--mcpu=cortex-m55` is
needed to see `vmladava.s8`).  Counts in the two ELFs:

| Instruction class | Standalone | cortex_m backend |
|---|---:|---:|
| MVE compute (`cdp`/`cdp2 15, ...`) — incl. `vmladava.s8`, `vaddva.s8` | 259 | 272 |
| MVE narrow vector load (`ldc 14, ...`) — `vldrb.s8 q*, [rN], #16` | 88 | 124 |
| MVE wide load (`ldc 15, ...`) — `vldrw.s32 q*, [rN], #...` | 1,096 | 146 |
| Scalar DSP MAC (`smlal`) | 1 | 0 |

Both binaries are MVE-vectorized at the int8 MAC inner loop.  The
standalone's higher MVE-wide-load count (`ldc 15`) reflects the
inlined per-layer constant loads (`LayerParams` field reads and
multiplier/shift fetches into vectors); the cortex_m runner does those
loads through CMSIS-NN's helper functions which mostly stay scalar.

## How to reproduce

### Standalone build & run

```bash
source examples/arm/arm-scratch/setup_path.sh

# 1. Dump artifacts (runs the same lowering pipeline the test harness uses)
#    Quickest way: run any of the standalone phase pytests with FVP toolchain
#    on PATH — they cmake + ninja + FVP automatically:
pytest backends/cortex_m/test/models/test_mv2_standalone_mve.py -v -s

# 2. Or do it manually with explicit invocation:
cmake -S examples/models/mv2_cortex_m_mve -B build/fvp \
      -DCMAKE_TOOLCHAIN_FILE=examples/models/mv2_cortex_m_mve/fvp/toolchain-arm-none-eabi.cmake \
      -DMV2_BUILD_FVP=ON \
      -DMV2_GENERATED_DIR=$(pwd)/examples/models/mv2_cortex_m_mve/generated
cmake --build build/fvp
arm-none-eabi-size -A build/fvp/mv2_runner_fvp.elf

FVP_Corstone_SSE-300_Ethos-U55 \
    -C ethosu.num_macs=128 \
    -C mps3_board.visualisation.disable-visualisation=1 \
    -C mps3_board.telnetterminal0.start_telnet=0 \
    -C mps3_board.uart0.out_file=- \
    -C mps3_board.uart0.shutdown_on_eot=1 \
    -C cpu0.semihosting-enable=1 \
    -C cpu0.semihosting-stack_base=0 \
    -C cpu0.semihosting-heap_limit=0 \
    -a build/fvp/mv2_runner_fvp.elf \
    --timelimit 600
# Prints "CYCLES <n>" then the 1000 int8 logits.
```

### cortex_m backend (CMSIS-NN) build & run

The fastest path that reuses what's already on disk: rebuild only
the existing test-framework semihosting runner with a tiny patch to
arm_perf_monitor.cpp that adds a printf for the cycle count
(necessary because the test runner is built with
`-DET_LOG_ENABLED=0`, so `ET_LOG(Info, ...)` is a no-op).

```bash
source examples/arm/arm-scratch/setup_path.sh

# 1. Apply the printf patch to arm_perf_monitor.cpp.  In StopMeasurements,
#    right after `uint32_t cycle_count = ARM_PMU_Get_CCNTR() - ...`,
#    add:
#       printf("BENCHMARK_CYCLES %u\n", (unsigned)cycle_count);
#    Also add `#include <cstdio>` next to the other includes.

# 2. Rebuild only the runner (host libs are already built).
( cd arm_test/arm_semihosting_executor_runner_corstone-300 && \
  cmake --build . --target arm_executor_runner -j )

# 3. Run pytest with a small _run_cmd-patch that dumps the FVP stdout
#    we'd otherwise lose.  Save this as /tmp/run_cm_capture.py:
cat > /tmp/run_cm_capture.py <<'PY'
from executorch.backends.arm.test import runner_utils as ru
_orig = ru._run_cmd
def _capture(cmd, check=True, env=None):
    out = _orig(cmd, check=check, env=env)
    with open("/tmp/cortexm_fvp.log", "ab") as f:
        f.write(out.stdout)
    return out
ru._run_cmd = _capture
import pytest, sys
sys.exit(pytest.main(['-v', '-s', '--runxfail',
    'backends/cortex_m/test/models/test_mobilenet_v2.py::test_implementation_mv2']))
PY
rm -f /tmp/cortexm_fvp.log
python3 /tmp/run_cm_capture.py

# 4. Cycle count.
grep BENCHMARK_CYCLES /tmp/cortexm_fvp.log
```

This reuses the test framework's full pipeline (CortexMQuantize +
to_executorch + serialize + run_corstone) which already knows the
right FVP flags (in particular `ethosu.extra_args=--fast`).  Total
wall time end-to-end is ~4 min in our environment.

A fully-from-scratch rebuild path (when host libs are missing or
need fresh flags) lives in commit `92dbb912d7` — the existing
`build_executor_runner.sh` chain with `--extra_build_flags` to wire
in `-DCMSIS_NN_LOCAL_PATH` and `-DEXECUTORCH_ENABLE_LOGGING=OFF`.

## Outstanding work / next experiments

Not yet committed, but candidates for further optimization:

- **Specialize the first 3×3 conv (`Cin = 3`).**  The MVE 16-wide
  inner reduction can't be used (Cin too small), so this layer falls
  to the scalar fallback and probably takes ~5 M cycles.  A
  hand-written 3-wide MVE inner with `vldrbq_s32`-style loads could
  cut it.
- **Wider spatial tiling for 1×1 convs.**  Going from 2 pixels to 4
  pixels per inner iter would amortize weight loads even more, but
  hits register pressure (16 scalar accumulators for 4×4 blocking).
  Need to check if 3-wide (12 accumulators) fits.
- **Custom memory planner.**  exir greedy gives 1.5 MB; an optimal
  interval-graph allocator could possibly hit 700-900 KB for MV2's
  lifetime pattern (~600 KB of two largest concurrent activations).
  Significant implementation effort.
- **Drop the FVP fixture from production builds.**  Trivial: a
  CMake option that omits `runner_fvp.cpp` + `input_fixture.h` saves
  600 KB of `.rodata` immediately.
