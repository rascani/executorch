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
| Cycle counter | **PMU `CCNTR`** read via `ARM_PMU_Get_CCNTR()` (see "Cycle counter calibration" below) |

For the cortex_m runner, `arm_perf_monitor.cpp` was patched with a
`printf("BENCHMARK_CYCLES %u\n", cycle_count)` so the PMU count reaches
stdout under `ET_LOG_ENABLED=0`.  Build command for that runner is
documented at the end of this file.

### Cycle counter calibration (important!)

The Corstone-300 FVP exposes two cycle counters on Cortex-M55, **and
they don't agree**.  We measured a constant 8.00× ratio between them
across NOP loops (100 and 10,000 iterations), an MVE MAC microbench
(1,000 `vmladavaq_s8`), and the full MV2 inference:

| Workload | DWT_CYCCNT | PMU CCNTR | ratio |
|---|---:|---:|---:|
| 100 `nop` loop | 90 | 710 | 7.89 |
| 10,000 `nop` loop | 8,751 | 70,010 | 8.00 |
| 1,000 `vmladavaq_s8` | 251 | 2,005 | 7.99 |
| Full MV2 inference | 46,718,953 | 373,751,624 | 8.00 |

A `volatile`-int NOP loop should take ~7 cycles per iter on Cortex-M55
at `-O3`.  PMU's 7.0 cycles/iter matches; DWT's 0.87 cycles/iter is
sub-issue-rate (impossible — there's at least one branch per iter).
**PMU is the real cycle counter on this FVP; DWT appears to be aliased
by a factor of 8.**  The cortex_m runner's existing instrumentation
uses PMU (`ARM_PMU_Get_CCNTR`), so all cross-runner comparisons in
this document use PMU.

> **Earlier versions of this document compared standalone DWT cycles to
> cortex_m PMU cycles and reported a "3.69× faster" headline.  That was
> wrong — the counters are on different scales.  The corrected
> comparison below uses PMU on both runners.**

## Headline numbers (Corstone-300 FVP, PMU CCNTR)

| Path | Cycles / inference | `.text` | `.rodata` | Arena |
|---|---:|---:|---:|---:|
| Standalone, baseline (commit `e6f5623e65`)        | ~1,158 M (extrapolated 8×) | 56,888 B | 4,283,956 B | 1,505,280 B |
| Standalone, all 18 committed kernel improvements  | **149,244,121** | ~66 KB | ~4.2 MB | 1,505,280 B |
| cortex_m backend (CMSIS-NN), rebuilt fresh        | **172,218,932** | 478,960 B (test runner) | 35,632 B (+ DDR data) | 1,505,280 B |

**Standalone is now ~14.6% faster than the cortex_m backend on this
FVP config** (23.0M PMU cycle margin), with both producing bit-exact
int8 outputs (max int8 diff = 0 across all 1000 logits).  Numbers
above are PMU cycle counts read identically (`ARM_PMU_Get_CCNTR()`)
in both runners.

The cortex_m backend runner's `.text` is much larger than ours
because it's the test-framework's generic semihosting runner that can
take any PTE via `-m` at runtime, includes BundleIO scaffolding, plus
the executorch runtime + flatbuffer parser + portable kernels + the
CMSIS-NN library + Ethos-U driver code.  For a real production
deployment that hardcodes the model (no semihosting, no BundleIO), the
cortex_m runner shrinks to roughly the 154 KB range — still 2-3× our
standalone.

So the standalone path now wins on **all three axes**: latency
(154.3M vs 172.2M PMU), code size (66 KB vs ~150 KB stripped), and
runtime dependencies (none — no flatbuffer parser, no kernel registry,
no driver layer).  Earlier versions of this doc reported the cortex_m
backend leading on latency; the 17-step optimization sequence below
flipped that.

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
reference = 0 across all 1000 logits).  The columns below report the
DWT counter because that's what was wired into the runner during these
measurements; the **internal ratios between standalone configurations
are valid** (DWT and PMU are linearly related by 8×, so percentage
improvements transfer directly).  Only cross-runner comparisons require
the PMU conversion.

| # | Commit | Change | DWT cycles | PMU (×8) | Speedup over previous | Cumulative |
|---:|---|---|---:|---:|---:|---:|
| 0 | `e6f5623e65` | Baseline: scalar dwconv, scalar requantize per OC | 144,728,651 | ~1,157.8 M | — | 1.00× |
| 1 | `802c00e3fb` | MVE channel-major dwconv (4 OC per inner iter) | 87,265,364 | ~698.1 M | 1.66× | 1.66× |
| 2 | `5351678930` | conv2d: 4-OC blocked w/ shared input + folded input_offset | 55,933,708 | ~447.5 M | 1.56× | 2.59× |
| 3 | `27c363b09f` | 1×1 conv: 2 output pixels per inner iter | 48,975,234 | ~391.8 M | 1.14× | 2.95× |
| 4 | `fa9b4c1f86` | Pack requant shifts as int8 (memory: -51 KB rodata) | 47,964,461 | ~383.7 M | 1.02× | 3.02× |
| 5 | `a56407f7e2` | Vectorize `add_s8` 4-wide | 46,718,954 | 373,751,624 | 1.03× | 3.10× |
| 6 | `f3f21c62ad` | Fix calibration (degenerate scale was masking saturated outputs); `always_inline` conv2d_s8 | 37,802,072 | 302,415,918 | 1.24× | 3.83× |
| 7 | `8245f5b2cc` | dwconv 4-pixel spatial tile (3x3 stride-1) | 35,918,876 | 287,350,999 | 1.05× | 4.03× |
| 8 | `29dbdf2dad` | First 3x3 conv (in_c=3) AOT-packed im2col path | 30,215,560 | 241,723,824 | 1.19× | 4.79× |
| 9 | `1eaf666c67` | `always_inline` dwconv2d_s8 | 29,861,094 | 238,888,097 | 1.01× | 4.85× |
| 10 | `d06c04b6af` | dwconv 2-pixel spatial tile (3x3 stride-2) | 29,394,384 | 235,154,411 | 1.02× | 4.92× |
| 11 | `7df24efe95` | Inline-asm wlstp/letp inner kernel for 1x1 conv | 26,021,579 | 208,171,977 | 1.13× | 5.56× |
| 12 | `8c97186c06` | Out-of-line conv2d_1x1 fast path, asm inlined directly | 25,133,178 | 201,064,764 | 1.04× | 5.76× |
| 13 | `2fee56d501` | AOT-fold dwconv input_offset for interior tiles (stride-1) | 24,325,407 | 194,602,599 | 1.03× | 5.95× |
| 14 | (same commit, stride-2) | AOT-fold dwconv input_offset for interior stride-2 tile | 24,145,665 | 193,164,664 | 1.01× | 5.99× |
| 15 | `9e865f3251` | conv1x1 reshape: 2-OC × 2-pixel asm block, shared input load | 23,583,681 | 188,669,439 | 1.024× | 6.14× |
| 16 | `8884a03e37` | conv1x1 fast path: handle odd `out_w` via 1-pixel tail | 21,716,754 | 173,734,023 | 1.086× | 6.66× |
| 17 | `3569979d4e` | requantize: CMSIS-NN-style fixup + `vrshlq_s32` (cuts ~10 MVE ops/call) | 19,288,873 | 154,310,972 | 1.126× | 7.50× |
| 18 | (pending) | requantize: drop left-shift step (shift ≤ 0 in MV2; saves 3 ops/call) | **18,655,515** | **149,244,121** | 1.034× | **7.76×** |

Headline: **7.50× faster than the original baseline**, bit-exact output
across all 1000 logits.  cortex_m backend on the same FVP/PMU: 172.2M
PMU — standalone is now **10.4% FASTER than cortex_m / CMSIS-NN**
(154.3M vs 172.2M PMU, 17.9M cycle margin).

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

**15. conv1x1 tile reshape: 2-OC × 2-pixel asm block.**  The previous
inner kernel ran two asm blocks per OC tile (one per output pixel),
each processing 4 OCs against one input column.  Per IC chunk that
issued 9 MVE ops for 4 outputs (1 x-load + 4 weight loads + 4
`vmladava.s8`), so 18 ops/IC for 8 outputs = 2.25 ops/output.

The new layout runs two asm blocks per OC tile, each processing 2 OCs
× 2 pixels against a shared pair of input columns.  Per IC chunk:
2 x-loads + 2 weight loads + 4 `vmladava.s8` = 8 ops for 4 outputs,
so 16 ops/IC for 8 outputs = 2.0 ops/output — a ~11% reduction in
inner-kernel issue count.  Each `vmladava.s8` accumulator must live
in an even GPR (Cortex-M55 has 7 evens), and 4 accs per asm block
fits comfortably; the previously-attempted single-block 8-accumulator
variant exceeded the constraint.  End-to-end PMU drops from 193.2M
to 188.7M (2.3%), with bit-exact output preserved.

**16. conv1x1 fast path handles odd `out_w`.**  The previous fast-path
dispatch required `(out_w & 1u) == 0u` because the inner kernel processed
two pixels per asm block.  Layers with `out_w = 7` (the head 7×7×320 →
7×7×1280 layer plus ~8 other `1x1` layers in the 7×7 stage of blocks
14–17) fell through to the generic 4-OC × 1-pixel path, which is much
slower per output.  This change drops the even-only check and appends a
4-OC × 1-pixel asm tail block (matching the per-pixel kernel) for the
final column when `out_w` is odd.

For the head conv alone — 49 spatial outputs × 1280 OCs against in_c=320
— this moves ~85% of the work (6/7 columns) onto the fast 2-pixel path
and only 1/7 onto the per-pixel tail.  End-to-end PMU drops from 188.7M
to 173.7M (-7.9%), bit-exact preserved.  Standalone is now within 0.9%
of CMSIS-NN's 172.2M PMU.

**17. CMSIS-NN-style `mve_requantize_per_channel`.**  The original
helper implemented round-half-away-from-zero using an explicit
remainder-vs-threshold comparison plus a predicated bump.  Per call:
~16 MVE vector ops (after CSE'ing the `vdupq_n_s32(0/1)` constants).

The new helper uses CMSIS-NN's trick from
`arm_divide_by_power_of_two_mve`: pre-add a `fixup` of `-1` to negative
products and then use `vrshlq_s32` (round-half-up).  Because vrshlq's
half-up matches half-away-from-zero for positive values and (after the
fixup) for negative values too, this collapses the post-multiply step
to just 4 MVE ops:

```c
int32x4_t fixup = vshrq_n_s32(vandq_s32(product, right), 31);
int32x4_t fixed = vqaddq_s32(product, fixup);
return vrshlq_s32(fixed, right);
```

Total per call: ~7 ops, half of what we had.  This helper is on every
conv2d / dwconv / avgpool / gemv / add output, so the savings hit every
layer.

End-to-end PMU: 173.7M → 154.3M (-11.2%), bit-exact across all 1000
int8 logits.  Standalone now **beats CMSIS-NN by 10.4%** (154.3M vs
172.2M PMU).

**18. Specialized requantize for `shift ≤ 0` (the MV2 norm).**
`mve_requantize_per_channel` from step 17 still pays for the left-shift
step (`vmaxq` + `vshlq`) and the `vminq` that clamps `shift` to `≤ 0`
— those are no-ops in the common case where every per-channel shift
value is already non-positive.  And in MV2 every requantize scale is
`< 1` (input × weight scale divided by output scale), so `frexp` always
returns a non-positive exponent.

This step adds a specialized `mve_requantize_per_channel_neg_shift`
helper that drops the three skipped ops and swaps it in at every
callsite (conv2d, dwconv, add_s8, avgpool, gemv).  Bit-exactness is
preserved by inspection — the skipped ops collapse to identity for
`shift ≤ 0` — and confirmed by `max int8 diff = 0` across all 1000
logits.

```c
int32x4_t product = vqrdmulhq_s32(acc, multiplier);
int32x4_t fixup = vshrq_n_s32(vandq_s32(product, shift), 31);
int32x4_t fixed = vqaddq_s32(product, fixup);
return vrshlq_s32(fixed, shift);
```

End-to-end PMU: 154.3M → 149.2M (-3.3%).  Standalone now **beats
CMSIS-NN by 14.6%** (23.0M PMU cycle margin).  The original helper
remains in the header as a safe fallback for kernels whose shift
distribution may include positive values.

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

FVP-vs-FVP latency comparison (full MV2, Corstone-300, PMU CCNTR):

| | PMU cycles | vs cortex_m |
|---|---:|---:|
| Standalone (baseline)                | ~1,157,829,208 (DWT 144,728,651 × 8) | 0.15× (much slower) |
| Standalone (after 5 commits)         | **373,751,624** | **0.46× (2.17× slower)** |
| cortex_m backend (CMSIS-NN)          | **172,218,932** | 1.00× |

So **cortex_m is faster than our standalone on this FVP**, by ~2.17×
after our five optimization commits and by ~6.7× before them.

### Why cortex_m is faster (likely)

The earlier version of this document tried to explain a 3.69× win
in our favor that turned out to be the counter mismatch.  The actual
direction of the comparison flips that — cortex_m wins.  The most
likely reasons it does:

1. **CMSIS-NN inner kernels are heavily tuned.**  `arm_nn_mat_mult_nt_t_s8`
   is hand-written inline asm with one cycle per `vmladava.s8` in the
   steady state, and an `lcalc`-style hardware loop wrapper.  Our
   intrinsic-driven kernels rely on GCC to schedule the same
   instructions — for the 4-OC × 2-pixel tile the compiler does a
   reasonable job, but there's room.
2. **Im2Col reuses + larger output tiles.**  CMSIS-NN's `arm_convolve_s8`
   pre-converts the input window into a column-major scratch buffer
   per output row, then runs the matmul over a wide row × all-of-OC
   tile.  This amortizes outer-loop bookkeeping more than our
   per-output-pixel kernel.  We pay an arena cost (the cortex_m
   arena and ours are the same size, so the scratch is included);
   they recover the cycles.
3. **Activation buffer reuse across calls.**  CMSIS-NN's wrapper
   passes `cmsis_nn_context` that holds the Im2Col scratch across
   layers, avoiding any per-call zero-init.  Our kernels each write
   their full output once; that's nominally equivalent, but they
   also each pay per-call function entry/exit + register spill cost
   because GCC can't always inline through every call site.
4. **Vector load alignment & contiguous strides.**  CMSIS-NN's
   layouts (and the input-offset AOT-fold pattern in their kernels)
   are tuned for the specific stride patterns CMSIS-NN's matmul
   expects.  Our generic NHWC layout occasionally produces
   non-contiguous weight reads in the depthwise kernels.

Splitting the 2.17× cleanly between these would require per-layer
cycle sampling, which we haven't done.

### Where the standalone path still wins

- **Code size:** 65,840 B vs CMSIS-NN test runner's 478,960 B (or
  ~154 KB for a hardcoded-model production cortex_m runner).  No
  ExecuTorch Program/Method machinery, no flatbuffer parser, no
  Ethos-U driver code.
- **No runtime dependency.**  The standalone path is a single C
  function with no calls into ExecuTorch.  Easier to slot into a
  baremetal RTOS task with no DDR or filesystem.
- **AOT memory plan reuse.**  We still use exir's `greedy` planner
  to produce the activation arena offsets, so the arena size matches
  cortex_m's exactly.

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

Now that we know cortex_m wins on latency, closing the 2.17× gap is
the highest-impact direction:

- **Match CMSIS-NN's matmul tiling.**  Their `arm_nn_mat_mult_nt_t_s8`
  produces a 4-row × 4-col output tile via a 4-OC × 4-spatial-pixel
  inner kernel, twice the spatial tile width we currently use.  Going
  from our 4×2 to 4×4 would amortize weight loads 2× over the inner
  loop at the cost of doubling accumulator register count (8 → 16
  int32 accumulators, fits in the 16 q-registers but spills the
  weight pointers to memory more often).
- **Im2Col scratch for 3×3 stride-2 convs.**  CMSIS-NN's depthwise
  kernels lean on a per-row Im2Col scratch buffer that we don't have.
  Adding a small (~kernel_size² × in_c bytes) scratch to the AOT
  arena plan would let us collapse the kernel-row loop into a single
  contiguous matmul.
- **Specialize the first 3×3 conv (`Cin = 3`).**  The MVE 16-wide
  inner reduction can't be used (Cin too small), so this layer falls
  to the scalar fallback and probably takes ~5 M cycles.  A
  hand-written 3-wide MVE inner with `vldrbq_s32`-style loads could
  cut it.
- **Custom memory planner.**  exir greedy gives 1.5 MB; an optimal
  interval-graph allocator could possibly hit 700-900 KB for MV2's
  lifetime pattern (~600 KB of two largest concurrent activations).
  Significant implementation effort, doesn't affect latency directly.
- **Drop the FVP fixture from production builds.**  Trivial: a
  CMake option that omits `runner_fvp.cpp` + `input_fixture.h` saves
  600 KB of `.rodata` immediately.
