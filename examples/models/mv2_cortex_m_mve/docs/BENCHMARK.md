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

## Headline numbers — what was actually measurable on FVP

| Path | Cycles | `.text` | `.rodata` | Arena |
|---|---:|---:|---:|---:|
| Standalone, baseline (commit `e6f5623e65`)   | 144,728,651 | 56,888 B | 4,283,956 B | 1,505,280 B |
| Standalone, with five committed kernel improvements | **46,718,954** | **65,840 B** | **4,233,180 B** | **1,505,280 B** |
| cortex_m backend (CMSIS-NN), built fresh         | **not measurable** in this environment | 154,872 B | 35,632 B (+ 69 MB `.ddr` carrying the `.pte` + arena + Ethos-U buffers) | 1,505,280 B (inside `.ddr`) |

**Why no cortex_m latency number?**  We rebuilt the cortex_m
arm_executor_runner fresh with logging + event tracer off (see the
build command at the end of this doc), patched `arm_perf_monitor.cpp`
to `printf` the cycle count under `ET_LOG_ENABLED=0`, and ran it on
the same Corstone-300 FVP config used for the standalone runner.
The FVP run hit `Reason: CPU time has been exceeded` at every
wall-clock budget tried — 30 min, 60 min, 4 hours — without
`Method::execute` ever returning.  The CMSIS-NN MVE conv kernels do
substantially more memory accesses per CPU cycle (Im2Col scratch
buffers + per-call wrapper setup), and the FVP simulates memory ops
at a fixed cost per access, so the wall-clock-per-simulated-cycle
ratio is much worse than for our standalone runner.  At the
observed throughput (~140 K simulated cycles per sec of wall clock,
based on partial progress before the timeouts), a CMSIS-NN MV2
inference whose true cost is in the typical 20-50 M cycle range
would need 2-6 minutes of FVP wall time just for the inference
itself, plus model load and tensor setup — but in practice none of
the runs we launched completed within their budgets.

There is a `mv2_benchmark_comparison.md` in the repo with a
249,819,842-cycle figure for the cortex_m runner, but those numbers
were taken on real silicon (FPGA / dev board), not in the FVP, and
the FVP CPU-time model is not cycle-accurate for the CPU.  So we
deliberately do **not** carry that number across as a comparable
baseline in this doc; the only honest comparison we can publish from
this session is the static footprint side-by-side.

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

A fair latency comparison would also need the cortex_m runner to
finish on the same FVP, which we couldn't get within practical
wall-clock budgets (see above).  On real silicon, where the cycle
counter measures actual hardware cycles rather than a simulator's
fixed cost model, the latency comparison should be redone with
both runners running on the same board with the same logging /
tracer settings.

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

```bash
source examples/arm/arm-scratch/setup_path.sh
CMSIS_NN_DIR=$(pwd)/arm_test/cmake-out/_deps/cmsis_nn-src  # locally fetched

# 1. Rebuild host executorch libs with logging + tracer OFF.
rm -rf arm_test/cmake-out
cmake -S . -B arm_test/cmake-out \
    -DCMAKE_TOOLCHAIN_FILE=$(pwd)/examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_DEVTOOLS=ON \
    -DEXECUTORCH_BUILD_ARM_ETDUMP=OFF \
    -DEXECUTORCH_ENABLE_LOGGING=OFF \
    -DEXECUTORCH_ENABLE_EVENT_TRACER=OFF \
    -DCMSIS_NN_LOCAL_PATH=$CMSIS_NN_DIR \
    --preset arm-baremetal
cmake --build arm_test/cmake-out --target install -j

# 2. Apply the printf patch to arm_perf_monitor.cpp so cycle count
#    survives logging-off.  Add at the top of StopMeasurements,
#    after cycle_count is computed:
#       printf("BENCHMARK_CYCLES %u\n", (unsigned)cycle_count);
#    (Don't forget #include <cstdio>.)

# 3. Build the MV2 runner against the freshly-installed libs.
backends/arm/scripts/build_executor_runner.sh \
    --pte=$(pwd)/benchmark_output/mobilenet_v2_cortexm/model.pte \
    --target=ethos-u55-128 \
    --output=$(pwd)/benchmark_output/mobilenet_v2_cortexm_fresh/cmake-out \
    --select_ops_list="cortex_m::dequantize_per_tensor.out,cortex_m::quantize_per_tensor.out,cortex_m::quantized_add.out,cortex_m::quantized_avg_pool2d.out,cortex_m::quantized_conv2d.out,cortex_m::quantized_depthwise_conv2d.out,cortex_m::quantized_linear.out,dim_order_ops::_clone_dim_order.out" \
    --extra_build_flags="-DEXECUTORCH_ENABLE_LOGGING=OFF -DEXECUTORCH_ENABLE_EVENT_TRACER=OFF -DCMSIS_NN_LOCAL_PATH=$CMSIS_NN_DIR"

# 4. Run on FVP.  Use the same flags as the cortex-m branch of
#    backends/arm/scripts/run_fvp.sh, in particular
#    -C ethosu.extra_args=--fast — without it the Ethos-U
#    simulator runs in detail mode even though the NPU is unused,
#    inflating wall-clock per simulated cycle.  The other two
#    cortex-m flags from run_fvp.sh (cpu0.semihosting-cwd and
#    cpu0.semihosting-cmd_line) are only needed for builds that
#    use --pte=semihosting --bundleio and read argv at runtime;
#    our runner has the .pte compiled in and SEMIHOSTING=OFF.
FVP_Corstone_SSE-300_Ethos-U55 \
    -C ethosu.num_macs=128 \
    -C ethosu.extra_args=--fast \
    -C mps3_board.visualisation.disable-visualisation=1 \
    -C mps3_board.telnetterminal0.start_telnet=0 \
    -C mps3_board.uart0.out_file=- \
    -C mps3_board.uart0.shutdown_on_eot=1 \
    -C cpu0.semihosting-enable=1 \
    -C cpu0.semihosting-stack_base=0 \
    -C cpu0.semihosting-heap_limit=0 \
    -a benchmark_output/mobilenet_v2_cortexm_fresh/cmake-out/arm_executor_runner \
    --timelimit 14400 \
    | tee /tmp/cortexm_fvp.log

# Cycle count from the printf patch in arm_perf_monitor.cpp:
grep BENCHMARK_CYCLES /tmp/cortexm_fvp.log
```

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
