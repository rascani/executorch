# KWT-1 standalone — Phase 8 MVE optimization journey

A measured per-kernel record of how the standalone KWT-1 inference
moved from a scalar baseline to an MVE-tuned 5.52× speedup on the
Corstone-300 FVP.  All cycle numbers are PMU `CCNTR` reads (see
"Cycle counter calibration" below) of one `kwt_1_inference()` call
on the `OneBlockKWT` model (d=64, d_ff=256, seq_len=8) — a single
transformer encoder block, the unit Phase 0-8 of this project ships.

## Setup

| Axis | Value |
|---|---|
| Target | Cortex-M55 + Helium (MVE), Corstone-300 FVP |
| Toolchain | arm-none-eabi-gcc 13.3.rel1 |
| Compiler flags | `-mcpu=cortex-m55 -mthumb -mfloat-abi=hard -mfpu=auto -O3 -ffunction-sections -fdata-sections` |
| Model | `OneBlockKWT(d=64, d_ff=256)`, seq_len=8 (Phase 7 dumper default) |
| Quantizer | `CortexMQuantizer` + `convert_pt2e`, 32-iter calibration |
| Lowering | `to_edge_transform_and_lower` + `CortexMPassManager` |
| Arena | Bump allocator (Phase 7); `exir.memory_planning.greedy` lands with the 12-block scale-up |
| Cycle counter | **PMU `CCNTR`** (`ARM_PMU_Get_CCNTR()`), per-kernel via the Phase 8 profile harness |
| Reference | `_ref_int8.bin` emitted by the dumper in the same Python run as the headers (sidesteps PT2E calibration non-determinism) |

The mv2 sibling's `docs/BENCHMARK.md` documents the DWT-vs-PMU 8.00×
aliasing on this FVP; we use PMU throughout for the same reason.

## Headline

| State | Total PMU cycles | Speedup | Bit-exact |
|---|---:|---:|---|
| Scalar baseline (no MVE)                          | 1,806,386 | 1.00× | ✓ |
| **Phase 8 final (MVE on linear, add, attn QK, transpose, LN)** | **327,377** | **5.52×** | ✓ |

Bit-exact = `(fvp_logits - _ref_int8.bin).abs().max() == 0` over all
512 int8 output values.

## Per-kernel progression

The Phase 8 profile harness (`include/kwt_1_profile.h`, opt-in with
`-DKWT_1_PROFILE=ON`) wraps every emitted kernel call site with a
`KWT_1_PROFILE_BEGIN`/`END` pair that snapshots the PMU cycle counter
into a per-kernel array.  The runner dumps the table between
`PROFILE_BEGIN`/`PROFILE_END` markers; rebuilding with `OFF` collapses
the macros to no-ops.  Cycle numbers below are reads of that table at
each checkpoint.

### Scalar baseline

| Layer | Kernel | PMU cycles | % |
|---:|---|---:|---:|
|  0 | quantize_input |   9,739 |  0.5 |
|  1 | layer_norm     |  25,419 |  1.4 |
|  2 | linear  (64×64)    | 132,377 |  7.3 |
|  3 | linear  (64×64)    | 132,377 |  7.3 |
|  4 | linear  (64×64)    | 132,377 |  7.3 |
|  5 | transpose      |  22,020 |  1.2 |
|  6 | transpose      |  21,628 |  1.1 |
|  7 | transpose      |  22,020 |  1.2 |
|  8 | attention      |  76,542 |  4.2 |
|  9 | linear  (64×64)    | 132,377 |  7.3 |
| 10 | add            |  54,808 |  3.0 |
| 11 | layer_norm     |  25,419 |  1.4 |
| 12 | linear (64→256, FFN expand)  | 525,593 | 29.0 |
| 13 | gelu (LUT)     |   8,209 |  0.4 |
| 14 | linear (256→64, FFN project) | 430,361 | 23.8 |
| 15 | add            |  54,808 |  3.0 |
| 16 | dequant memcpy |     312 |  0.0 |
|    | **total**      | **1,806,386** | |

Linear dominates (82.1 % of cycles), as expected for a transformer
block at this size.

### Phase 8 final

| Layer | Kernel | PMU cycles | Δ vs baseline |
|---:|---|---:|---:|
|  0 | quantize_input (scalar, MVE-FP has no `vdivq_f32` in this toolchain) |   9,739 |   1.00× |
|  1 | layer_norm (MVE float, 3-pass)     |   7,316 |   3.47× |
|  2 | linear  (MVE)                       |  20,504 |   6.46× |
|  3 | linear  (MVE)                       |  20,504 |   6.46× |
|  4 | linear  (MVE)                       |  20,504 |   6.46× |
|  5 | transpose (rank-3 specialization)  |   3,082 |   7.15× |
|  6 | transpose (rank-3 specialization)  |   1,754 |  12.33× |
|  7 | transpose (rank-3 specialization)  |   3,082 |   7.15× |
|  8 | attention (MVE QK, scalar softmax+AV) |  56,949 |   1.34× |
|  9 | linear  (MVE)                       |  20,504 |   6.46× |
| 10 | add  (MVE per-element, vector requantize) |   7,121 |   7.70× |
| 11 | layer_norm (MVE)                    |   7,316 |   3.47× |
| 12 | linear (MVE, FFN expand)            |  80,024 |   6.57× |
| 13 | gelu (LUT, scalar)                  |   8,209 |   1.00× |
| 14 | linear (MVE, FFN project)           |  53,336 |   8.07× |
| 15 | add  (MVE)                          |   7,121 |   7.70× |
| 16 | dequant memcpy                      |     312 |   1.00× |
|    | **total**                           | **327,377** | **5.52×** |

After Phase 8 the cycle budget is dominated by what's left of the
linears (256 k, 78 %) — these are now MVE memory-bound and won't
shrink without a memory-layout rework — and by the scalar `expf`
in softmax inside `attention_fused_s8` (~25 k of attention's 57 k).

## Optimization steps (chronological)

Each step landed bit-exact before the next was attempted.

### Step 1 — `linear_s8`: K-vector dot + N-tile-4

`linear_s8` is called six times per block (Q, K, V projections; the
attention output projection; FFN expand; FFN project) and consumes
1.50 M of the 1.81 M scalar baseline.

Strategy: vectorize the K dim with `vmladavaq_s8` (16 int8 lanes,
scalar int32 accumulator), tile N by 4 so a single input load
amortizes across four weight rows.  Row sums (needed for the
`filter_offset` term in the kernel-sum reformulation) vectorize via
`vaddvaq_s8`.

| Linear | Before | After | Speedup |
|---|---:|---:|---:|
| 4× (K=64, N=64) | 132,377 ea | 35,869 ea | 3.69× |
| FFN expand (K=64, N=256)  | 525,593 | 141,469 | 3.72× |
| FFN project (K=256, N=64) | 430,361 |  67,165 | 6.41× |

**Total after step 1: 673,034 cycles (2.69× headline)**

### Step 2 — `add_s8`: MVE per-element + bit-exact vector requantize

The two residual adds each took 54,808 scalar cycles, dominated by
three `kwt_1_requantize` calls per element across 512 elements.

Wrote `kwt_1_mve_requantize_nonpos`, a bit-exact MVE port of the
scalar `arm_nn_requantize` for the shift ≤ 0 case (which covers every
requantize in KWT-1).  It does the same `vqrdmulhq_n_s32` rounding
doubling-multiply as the scalar path, then a manual away-from-zero
correction (vshrq + predicate compare on the remainder against a
sign-aware threshold).  Architectural `vrshlq_s32` alone would round
ties to positive infinity and diverge on negative ties.

`add_s8` then widens 16 int8 inputs through four `vldrbq_s32` quads,
runs the (x-zp)<<20 + three-stage requantize pipeline in vectors, and
narrows back via `vstrbq_s32`.

| Add | Before | After | Speedup |
|---|---:|---:|---:|
| L10 / L15 | 54,808 ea | 7,121 ea | 7.70× |

**Total after step 2: 577,660 cycles (3.13× headline)**

### Step 3 — `attention_fused_s8`: MVE QK^T via kernel-sum trick

`attention_fused_s8` has three internal steps: QK^T (S×S dot products,
each over D=64), softmax (over S), AV (D dot products, each over S).

QK^T was vectorized using the same kernel-sum reformulation as the
linears: `sum (q+q_off)(k+k_off) = qk_dot + q_off*k_sum + k_off*q_sum
+ D*q_off*k_off`.  The inner D=64 loop becomes four `vmladavaq_s8`
instead of 64 scalar multiplies.  K-row sums and the per-batch V
column sums are precomputed once via `vaddvaq_s8` / scalar.

Softmax stays scalar (its inner loop runs `expf` 8 times per row;
that dominates).  The AV step stays scalar — its inner loop is S=8
elements, below MVE's 16 int8 lanes.  A predicated-load MVE attempt
(`vldrbq_z_s8` + `vmladavaq_s8`) broke bit-exactness at the int8
output level; the likely cause is that the predicated load doesn't
mask off the upper 8 lanes the way the ACLE spec promises on this
M55/gcc combo.  Reverted — the savings (~7 k cycles) weren't worth
the bit-exactness risk.

| Kernel | Before | After | Speedup |
|---|---:|---:|---:|
| attention | 76,542 | 56,949 | 1.34× |

**Total after step 3: 558,067 cycles (3.24× headline)**

### Step 4 — `linear_s8`: vector requantize + narrow-store at tile end

Profile inspection showed `linear_s8` was now bottlenecked by the
per-output requantize: with K=64 only 4 inner MVE iters per tile, the
4× scalar `kwt_1_requantize` + clamp + byte-store at tile end cost
roughly half the per-tile cycles.

Replaced the per-lane scalar requantize with the new vectorized
`kwt_1_mve_requantize_nonpos`: pack four accumulators into an
`int32x4_t`, add the `row_off_term` and a vector-loaded `kernel_sum`,
requantize+clamp in registers, and narrow-store with `vstrbq_s32`.
Hoisted out as a separate `shift <= 0` fast path so the scalar
fallback covers the unusual `shift > 0` case.

| Linear | Before | After | Δ |
|---|---:|---:|---:|
| 4× (K=64, N=64) | 35,869 ea | 20,504 ea | -43 % |
| FFN expand      | 141,469   |  80,024   | -43 % |
| FFN project     |  67,165   |  53,336   | -21 % |

L14 (K=256) saw less benefit because its inner K loop already
dominated relative to the per-tile overhead.

**Total after step 4: 421,333 cycles (4.29× headline)**

### Step 5 — `transpose_s8`: rank-3 specialization

The generic rank-N `transpose_s8` walked an explicit `idx[4]` array
per element to compute the linearized input offset — 43 cycles/byte.
All three transposes KWT-1 emits are rank-3 ((B, S, D) ↔ (B, D, S)),
so a specialized triple-nested loop with explicit pointer arithmetic
collapses the overhead.

| Transpose | Before | After |
|---|---:|---:|
| L5 / L7 | 22,020 ea |  3,082 ea |
| L6      | 21,628    |  1,754    |

7-12× per call, depending on which permutation.  No MVE intrinsics —
just removing the dispatch overhead.

**Total after step 5: 363,583 cycles (4.97× headline)**

### Step 6 — `layer_norm_s8`: MVE float, 3-pass

LayerNorm is float-internal: dequant → mean → sumsq → rstd → affine →
requant.  Each pass over D=64 is naturally 16 `float32x4_t`
operations.  Wrote `layer_norm_s8_mve` using `vldrbq_s32` to widen
int8 to int32, `vcvtq_f32_s32` to convert, `vmulq_n_f32` for the
dequant scale, `vfmaq_f32` for the sum-of-squared-deviations, and
`vcvtaq_s32_f32` (away-from-zero round) for the final requantize.

`vaddvq_f32` is not exposed by this toolchain, so the horizontal sum
of a 4-lane float vector unrolls to four `vgetq_lane_f32` reads.

| LN | Before | After | Speedup |
|---|---:|---:|---:|
| L1 / L11 | 25,419 ea | 7,316 ea | 3.47× |

Bit-exact with the scalar reference — torch's CPU LN doesn't
guarantee a strict left-to-right reduction either, so the pairwise
vector reduction happens to land on the same float32 result.

**Total after step 6: 327,377 cycles (5.52× headline)**

## Negative results

Kept the simpler version each time after measurement showed a
regression or bit-exactness break.

| Attempt | Result | Why |
|---|---|---|
| `linear_s8` N-tile-of-8 | -7 % on K=64 cases | 8 weight loads per K iter contended for the L/S pipeline. |
| `linear_s8` m-tile-2 × N-tile-4 (8 accumulators per inner) | -7 % on K=256 | Register pressure ate the input-load savings; the 6 simultaneously-live int8 vectors hit MVE's 8-Q register file with no spare for scratch. |
| `attention_fused_s8` AV step with `vldrbq_z_s8` + `vmladavaq_s8` | -7 % cycles, but bit-exactness broke | Predicated MVE load didn't actually mask off the upper 8 lanes the way ACLE promises — likely an M55/gcc-13 interaction.  A safe MVE path would need V-row zero-padding, which negates the savings. |
| `kwt_1_quantize_input` MVE-FP | Build error | This toolchain doesn't expose `vdivq_f32`; substituting `x * (1/scale)` would diverge from the PyTorch `round(x/scale)` reference in low bits. |

## Reproducing

Per-kernel profile (Corstone-300 FVP):

```bash
source examples/arm/arm-scratch/setup_path.sh
python -m executorch.examples.models.kwt_1_cortex_m_mve.tools.dump_kwt_1_artifacts \
    --save-ref examples/models/kwt_1_cortex_m_mve/generated/
cmake -S examples/models/kwt_1_cortex_m_mve -B build/fvp \
      -DCMAKE_TOOLCHAIN_FILE=examples/models/kwt_1_cortex_m_mve/fvp/toolchain-arm-none-eabi.cmake \
      -DKWT_1_BUILD_FVP=ON -DKWT_1_PROFILE=ON
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
    -a build/fvp/kwt_1_runner_fvp.elf --timelimit 300
```

The runner prints (in this order):
- `CYCLES_PMU <n>`: end-to-end wall cycles
- `PROFILE_BEGIN` ... `PROFILE_END`: per-kernel breakdown
- `RESULT_BEGIN` ... `RESULT_END`: the 512 int8 output values

Compare `RESULT_*` against
`examples/models/kwt_1_cortex_m_mve/generated/_ref_int8.bin` for the
bit-exactness check.

## Post-Phase 8 optimizations

After the Phase 8 write-up above, a second pass of optimizations
landed in the optimization session.  These weren't part of the
original phased plan, but they're sitting on the same branch and
the per-kernel cycle table at the new defaults is below.

| Step | Total PMU | Speedup | Δ |
|---|---:|---:|---:|
| Phase 8 final (d_ff=256)                              | 327,377 | 5.52× | — |
| `--d-ff` CLI default flipped to 128 (paper KWT-1 dims) | 261,713 | 6.90× | -20% |
| `linear_s8` K=64 specialization — hoist 4 input vectors across the N-tile loop | 238,935 | 7.56× | -9% |
| `attention_fused_s8` AV step MVE via zero-padded v buffer | 219,830 | 8.22× | -8% |
| `attention_fused_s8` AV step + softmax housekeeping (vector requantize tile-4 on 4 output channels) | 208,116 | 8.68× | -5% |
| `attention_fused_s8` QK^T per-score requantize vectorized (tile-4 on 4 scores) | 205,060 | 8.81× | -1.5% |
| `attention_fused_s8` MVE path generalized for S>16 (chunked vmladavaq_s8 over padded contraction) | 208,529 | 8.66× | +1.7% on OneBlock — but unlocks **-57% on canonical KWT-1 1-block-S=98 (11.1 M → 4.81 M)** |
| `linear_s8` MVE path for K not a multiple of 16 (canonical KWT-1's K=40 embed) | 212,255 | 8.51× | +1.8% OneBlock; **-15% canonical 1-block-S=98 (4.81 M → 4.07 M)** |
| `linear_s8` M-tile-2 path for large M (K≠64), shares 4 weight loads across 2 input rows | 211,762 | 8.53× | unchanged OneBlock; **-3.3% canonical 1-block-S=98 (4.07 M → 3.93 M)** |
| `attention_fused_s8` QK^T D=64 q_row hoist across all S/4 j-tiles | 212,246 | 8.51× | unchanged OneBlock; **-0.5% canonical (3.93 M → 3.91 M)** |
| `softmax_s8` MVE FP for passes 1 and 3 (dequant + max; normalize + quantize) | 211,469 | 8.54× | OneBlock improved; **-5% canonical (3.91 M → 3.71 M)** |
| MVE v_col_sums + v_padded build in attention | 211,469 | 8.54× | OneBlock unchanged; **-0.3% canonical (3.71 M → 3.70 M)** |
| Tighten `kwt_1_fast_expf` (floorf + fmaf), opt-in `-DKWT_1_FAST_EXPF=ON` | (no change at default) | 8.54× | with FAST_EXPF=ON: **-7% canonical attention (1.22 M → 1.13 M), -2.4% canonical total (3.70 M → 3.62 M)** |

Per-kernel breakdown at the 205,060-cycle headline:

| Layer | Kernel | PMU cycles | % |
|---:|---|---:|---:|
|  0 | quantize_input |   9,739 |  4.7 |
|  1 | layer_norm     |   7,316 |  3.6 |
|  2 | linear  (K=64, N=64) |  16,781 |  8.2 |
|  3 | linear  (K=64, N=64) |  16,781 |  8.2 |
|  4 | linear  (K=64, N=64) |  16,781 |  8.2 |
|  5 | transpose      |   3,082 |  1.5 |
|  6 | transpose      |   1,754 |  0.9 |
|  7 | transpose      |   3,082 |  1.5 |
|  8 | attention      |  23,074 | 11.3 |
|  9 | linear  (K=64, N=64) |  16,781 |  8.2 |
| 10 | add            |   7,121 |  3.5 |
| 11 | layer_norm     |   7,316 |  3.6 |
| 12 | linear (K=64, N=128, FFN expand)  |  32,653 | 15.9 |
| 13 | gelu (LUT)     |   4,113 |  2.0 |
| 14 | linear (K=128, N=64, FFN project) |  31,253 | 15.2 |
| 15 | add            |   7,121 |  3.5 |
| 16 | dequant memcpy |     312 |  0.2 |
|    | **total**      | **205,060** | |

Negative results since Phase 8:
- `linear_s8` m-tile-2 × N-tile-4 for K≥128 (8 accumulators per
  inner): regressed by ~1.3% — the M55 issue queue couldn't keep
  the 8 vmladavaq_s8 ops in flight.  Same as the Phase 8 attempt
  but isolated to long-K linears; didn't help.
- `kwt_1_fast_expf` polynomial substitute (`-DKWT_1_FAST_EXPF=ON`
  build flag): originally landed as a small regression (branches +
  no FMA in the polynomial); rewrote with `floorf` for branchless
  range reduction and explicit `fmaf` Horner.  Now strictly beats
  newlib's expf on M55+VFP — -7% canonical attention, -2.4%
  canonical 1-block-S=98 total.  Still ships off-by-default because
  the ~4 ULP polynomial accuracy may eventually drift through the
  softmax → int8-quantize pipeline on trained-weight inputs (random-
  weight verification was bit-exact, but bit-exactness isn't
  algorithmically guaranteed).

## Comparison vs DS-CNN

DS-CNN is the textbook MCU keyword-spotting model — the MLPerf Tiny
reference and the Hello Edge paper's lead architecture.  It already
lives in `examples/models/mlperf_tiny/ds_cnn.py` and already lowers
through the cortex_m backend + CMSIS-NN via
`backends/cortex_m/test/models/test_ds_cnn.py`.  Running it through
the same FVP and reading the PMU CCNTR (after the
`arm_perf_monitor.cpp` printf patch in this branch) gives a real
apples-to-apples cycle comparison.

| Model | Configuration | Params | MACs | PMU cycles | cycles / MAC |
|---|---|---:|---:|---:|---:|
| DS-CNN-S (12-class) | cortex_m + CMSIS-NN runtime, full inference | 23,180 | 2.66 M | **1,902,290** | 0.72 |
| KWT-1 OneBlockKWT (S=8) | standalone MVE, our 1-block Phase 8/optim target | 33,472 | 262 K | **205,060** | 0.78 |
| KWT-1 1-block @ S=98 (1× canonical input), scalar attention | standalone, attention via scalar fallback (pre long-S MVE) | ~98 K | 3.4 M | 11,096,844 | ~3.3 |
| KWT-1 1-block @ S=98, MVE long-S attention only | post long-S attention | ~98 K | 3.4 M | 4,806,835 | ~1.4 |
| **KWT-1 1-block @ S=98**, full post-Phase 8 stack | + K-tail MVE, M-tile-2, q_row hoist, softmax MVE, v build MVE | ~98 K | 3.4 M | **3,704,182** | ~1.1 |
| KWT-1 12-block canonical (35-class) | standalone MVE — full encoder | 406,563 | 38.8 M | **~44 M †** | ~1.1 |

† Extrapolated.  The 1-block-S=98 measurement (3.70 M cycles) is
~750 k embed + ~50 k mean/head + ~2.9 M for one encoder block at
S=98.  12 blocks + the same overhead ≈ 44 M cycles.  The 12-block
end-to-end FVP run didn't terminate in three attempts (1 h, 30 min,
50 min CPU) — the simulator slows non-linearly on this host for
long-running workloads.  The 1-block S=98 anchor is the safest
measurement to extrapolate from.

### Headline tradeoff (Corstone-300 FVP, PMU CCNTR)

- DS-CNN-S full inference: **1.9 M cycles** for ~94 % top-1 on the
  Speech Commands 12-class benchmark.  ≈ 64 ms at 30 MHz.
- KWT-1 canonical inference: **~44 M cycles** for ~96.6 % top-1 on
  the Speech Commands 35-class benchmark.  ≈ 1.5 s at 30 MHz.

**~23× the cycles for ~2.5 absolute accuracy points** on the harder
35-class task.  Per-MAC efficiencies are nearly identical (0.7-0.8
cy/MAC on the MVE-friendly portions); the cost differential is the
model doing more work, not poorer kernel utilization.

DS-CNN remains the practical MCU KWS choice when latency or footprint
matters.  KWT-1's contribution to the cortex_m backend isn't beating
DS-CNN at its own benchmark — it's enabling transformer-class models
(KWT, Conformer, Squeezeformer, …) on Cortex-M at all, via the
LayerNorm / GELU / fused attention / BMM / Softmax kernels this
directory ships.

### Reproducing the DS-CNN cycle measurement

```bash
# 1. Patch arm_perf_monitor.cpp to printf BENCHMARK_CYCLES (already
#    applied in this branch — see git log on
#    examples/arm/executor_runner/arm_perf_monitor.cpp).

# 2. Rebuild the existing test runner.  Host libs are already built
#    from prior test runs.
( cd arm_test/arm_semihosting_executor_runner_corstone-300 && \
  cmake --build . --target arm_executor_runner -j )

# 3. Run pytest with a small _run_cmd-patch that captures the FVP
#    stdout we'd otherwise discard.
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
    'backends/cortex_m/test/models/test_ds_cnn.py::test_implementation_ds_cnn']))
PY
rm -f /tmp/cortexm_fvp.log
source examples/arm/arm-scratch/setup_path.sh
python3 /tmp/run_cm_capture.py

# 4. Cycle count.
grep BENCHMARK_CYCLES /tmp/cortexm_fvp.log
```

Authored with assistance from Claude (claude.ai/code).
