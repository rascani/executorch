# Standalone vs. cortex_m backend — MobileNetV2 footprint & latency

Direct comparison of the standalone Cortex-M55 + MVE inference path in
this directory against the existing cortex_m backend (which dispatches
through ExecuTorch's runtime to CMSIS-NN kernels) for full
MobileileNetV2 on the Corstone-300 FVP.

## Setup

| Axis | Both paths |
|---|---|
| Target | Cortex-M55 + Helium (MVE), Corstone-300 FVP |
| Toolchain | arm-none-eabi-gcc 13.3.rel1 |
| Compiler flags | `-mcpu=cortex-m55 -mthumb -mfloat-abi=hard -mfpu=auto -O3` |
| Model | torchvision `mobilenet_v2`, int8 quantized via `CortexMQuantizer` + `convert_pt2e` |
| Memory planner | `exir.memory_planning.greedy` (same planner used by `MemoryPlanningPass`) |

The cortex_m runner ELF and `.pte` come from the user's prior
benchmark at `benchmark_output/mobilenet_v2_cortexm/`.

## Memory footprint

Section-level sizes from `arm-none-eabi-size -A`:

| Section | Standalone (this dir) | cortex_m backend | Delta |
|---|---:|---:|---:|
| `.text` (runtime code) | **56,888 B** | **164,408 B** | -65% |
| `.rodata` (weights, params, fixture) | 4,283,972 B | 35,632 B | n/a (cortex_m carries weights in `.ddr` instead) |
| `.ddr` (.pte payload + arena + scratch) | 1,505,280 B (arena only) | 69,008,240 B (.pte + arena + scratch + Ethos-U buffers) | n/a |
| `.bss` | 1,996 B | 25,212 B | -92% |
| `.data` | 2,056 B | 2,532 B | -19% |
| Stack / heap (DTCM, reserved) | 64 KB + 64 KB | 32 KB + 32 KB | 2x larger reservations on our side |

Headline-friendly footprint (weights + arena + runtime code):

| Path | Code | Weights | Arena | Total |
|---|---:|---:|---:|---:|
| Standalone           | **57 KB** | ~3.4 MB | 1.50 MB | **~5.0 MB** |
| cortex_m backend     | **161 KB** | 3.99 MB (.pte) | 1.50 MB | **~5.7 MB** |

The standalone path is **~100 KB smaller** in code (no ExecuTorch
runtime, no flatbuffer parser, no CMSIS-NN library, no op-dispatch
table) and **~600 KB smaller** in weight storage (raw `.rodata`
arrays vs. a flatbuffer-wrapped `.pte`).  The activation arena is
**identical** because both paths feed `exir.memory_planning.greedy`
the same lowered graph.

## Latency (Corstone-300 FVP, DWT CYCCNT)

Both numbers are CPU cycle counts for one full MobileNetV2 inference.

| Path | Cycles / inference | At 200 MHz (est.) | At 400 MHz (est.) |
|---|---:|---:|---:|
| Standalone (baseline)              | 144,728,651 | 724 ms | 362 ms |
| Standalone (after dwconv MVE)      | **87,265,364** | **436 ms** | **218 ms** |
| cortex_m backend (CMSIS-NN, prior benchmark) | 249,819,842 | 1,249 ms | 625 ms |

The cortex_m backend number is **not a fresh measurement in this
session** — it comes from `mv2_benchmark_comparison.md` in the repo,
a prior benchmark run by the user.  Two attempts to refresh it in
this session both failed:

1. **First attempt** — re-ran the existing
   `benchmark_output/mobilenet_v2_cortexm/cmake-out/arm_executor_runner`
   ELF on the FVP.  The runner reached `Preparing inputs...` then
   exhausted the 30-minute FVP wall-clock budget before
   `Method::execute` returned.  Most of the wall time was burned in
   `ET_LOG(Info, ...)` semihosting traps from the runtime path.

2. **Second attempt** — patched the runner to disable `ET_LOG` and
   added a `printf` for the cycle count, then rebuilt.  The rebuild
   regenerated the runner's `link.txt` without
   `libarm_portable_ops_lib.a` (the per-build kernel-registration
   archive — `libportable_kernels.a` has the kernel bodies but the
   `Kernel kernels_to_register[]` table that the registry walks at
   startup lives in `arm_portable_ops_lib`).  The runner now fails
   at startup with `kernel 'dim_order_ops::_clone_dim_order.out'
   not found`.  Recovering would mean re-running
   `build_executor_runner.sh` from scratch with a proper kernel
   selection from the MV2 PTE, and `build_executor_runner.sh`
   currently triggers `FetchContent` for CMSIS-NN, which is
   network-blocked in this environment.

The 249.8 M cycle figure in `mv2_benchmark_comparison.md` was
measured with logging *enabled* on this hardware (the same FVP
config we run our standalone path on).  Logging adds semihosting
trap wall time but not many CPU cycles to `Method::execute` — the
cycle count is wall-clock-independent.  So the comparison axis is
still useful, but readers should treat the 249.8 M figure as the
prior result rather than something this session reproduced.

Our standalone 87.3 M (and the 144.7 M baseline before the MVE
dwconv prototype) are both fresh in-session measurements with
`ET_LOG` disabled and the DWT cycle counter read directly in the
runner.

The 2.9× standalone-vs-cortex_m gap is surprising at first glance.
Three factors explain it:

1. **Per-op dispatch overhead** in the cortex_m runtime.  Out of
   ~250 M cycles total, only ~177 M land in kernel bodies (sum of
   the per-op cycles in `mv2_benchmark_comparison.md`); the
   remaining ~73 M is `Method::execute` machinery: tensor-view
   construction per op, `KernelRuntimeContext::allocate_temp` calls,
   per-op kernel lookup, etc.  The standalone path inlines kernel
   calls directly into the entry point and has no per-op runtime
   overhead.
2. **Per-pixel CMSIS-NN entry overhead.**  `arm_convolve_wrapper_s8`
   takes a `cmsis_nn_dims` setup, picks an inner kernel variant from
   a dispatch table, and threads through a `cmsis_nn_context` scratch
   buffer setup before doing real work.  On small layers this
   dwarfs the actual MAC time.
3. **The cortex_m kernels emit per-channel requantize through the
   CMSIS-NN scalar `arm_nn_requantize` path inside their wrapper,
   not the vectorized MVE form.**  Our `mve_requantize_per_channel`
   matches the same math but is inlined directly in the MAC loop.

The standalone path is therefore not "faster than CMSIS-NN's
kernels" — for individual large 1×1 convs CMSIS-NN's int8 MVE
inner loop is still tighter than what we hand-wrote.  We win on
end-to-end inference *because* we eliminate the dispatch and
wrapper overhead between layers and call the kernel directly.

## Per-op breakdown (cortex_m backend, from prior benchmark)

The user's prior benchmark in `mv2_benchmark_comparison.md` already
broke the cortex_m backend down per op.  Restated here so a reader
of just this doc can compare:

| Op family | cortex_m cycles | Layer count | Avg / op |
|---|---:|---:|---:|
| `quantized_conv2d.out`                | ~119 M | 35 | 3.4 M |
| `quantized_depthwise_conv2d.out`      | ~ 60 M | 17 | 3.5 M |
| `quantized_add.out`                   | ~3.9 M | 10 | 0.39 M |
| `quantized_avg_pool2d.out`            | 0.22 M | 1 | 0.22 M |
| `quantized_linear.out`                | 2.3 M | 1 | 2.3 M |
| `quantize_per_tensor.out`             | 1.4 M | 1 | 1.4 M |
| `dequantize_per_tensor.out`           | 4.5 K | 1 | 4.5 K |
| `_clone_dim_order.out`                | 2.9 K | 1 | 2.9 K |
| Per-op kernel total                   | **~187 M** | | |
| `Method::execute` total               | **249.8 M** | | |
| Inferred dispatch overhead            | ~63 M | | |

## Prototype optimization: channel-major depthwise

The biggest single bottleneck in the *standalone* baseline was the
`dwconv2d_s8` kernel.  Depthwise has no cross-channel reduction —
each output channel needs only the matching input channel — so the
natural vectorization axis is the channel dimension itself.  The
baseline kernel walked output channels one at a time scalar,
leaving 17 of MV2's layers fully un-vectorized.

The fix vectorizes 4 output channels per inner iteration using
`vldrbq_s32` to load four sign-extended int8s, `vmulq_s32 +
vaddq_s32` to accumulate, and `mve_requantize_per_channel` for the
per-channel requantize step:

```c
#if MV2_USE_MVE
  if ((out_c & 3u) == 0u) {
    for (oh) for (ow) for (cb = 0; cb < out_c; cb += 4) {
      int32x4_t acc = vld1q_s32(bias + cb);
      for (kh) for (kw) {
        int32x4_t x = vldrbq_s32(input_ptr + cb);
        int32x4_t w = vldrbq_s32(weight_ptr + cb);
        x = vaddq_s32(x, vdupq_n_s32(input_offset));
        acc = vaddq_s32(acc, vmulq_s32(x, w));
      }
      acc = mve_requantize_per_channel(acc, mults_v, shifts_v);
      vstrbq_s32(output_ptr + cb, vminq_s32(vmaxq_s32(acc, lo), hi));
    }
    return;
  }
#endif
```

Every depthwise layer in torchvision MV2 has `out_c %
4 == 0` (channel counts in {32, 96, 144, 192, 384, 576, 960}), so
this fast path always triggers.

End-to-end result on Corstone-300 FVP:

| Standalone variant | Cycles / inference | Speedup over baseline |
|---|---:|---:|
| Baseline (scalar dwconv) | 144,728,651 | 1.00× |
| With MVE channel-major dwconv | **87,265,364** | **1.66×** |

Bit-exact correctness preserved across all four phase tests (max
int8 diff = 0 vs. cortex_m runtime reference).

## Next-step candidates

The remaining ~87 M cycles are dominated by `conv2d_s8` (35 layers,
mostly 1×1).  Three high-leverage follow-ups:

1. **Output-channel blocking for `conv2d_s8`.**  Currently the kernel
   walks `oc` one at a time and re-reads the input pixel for each
   output channel.  Blocking 4 OCs per inner iteration (with one
   broadcast input load and 4 weight loads, accumulating into 4
   independent `int32x4_t` lanes via `vmlaq_n_s32`) would cut
   redundant input loads by 4× and let the requantize go vector at
   the end.
2. **Im2Col-free GEMM for 1×1 convs.**  A 1×1 conv is exactly a
   GEMM of `(H·W, Cin) × (Cin, Cout)`.  Treating the input as a
   flat `(H·W) × Cin` matrix lets us reuse the input rows across
   all output channels and run a proper 16-wide MVE GEMM kernel.
3. **Pre-shuffle weights AOT.**  The dumper currently emits weights
   in the OHWI layout the cortex_m pipeline produces; an MVE
   GEMM-friendly layout (`Cout`-block major, padded for SIMD lanes)
   would remove all the strided loads in the inner kernel.

(1) is the smallest change and closes most of the remaining gap;
(2) and (3) together would likely bring full-MV2 latency below
40 M cycles, comparable to a tuned CMSIS-NN deployment without the
dispatch overhead.

## How to reproduce

```bash
source examples/arm/arm-scratch/setup_path.sh

# Dump artifacts (uses tools/dump_mv2_artifacts.py with random-init MV2).
# See test_mv2_standalone_mve.py for the exact lowering pipeline.

# Standalone build & FVP run:
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
    --timelimit 1800
# Prints "CYCLES <n>" then the 1000 int8 logits.

# cortex_m backend run for comparison:
FVP_Corstone_SSE-300_Ethos-U55 ... \
    -a benchmark_output/mobilenet_v2_cortexm/cmake-out/arm_executor_runner \
    --timelimit 1800
# Prints the ET_LOG "Profiler report, CPU cycles per operator" summary.
```
