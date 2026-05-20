# KWT-1 transformer on Cortex-M55 + MVE — experiment summary

A condensed record of the standalone KWT-1 implementation: what was
built, what worked, what didn't, how it compares to its predecessor
project (MV2 CNN) on the same hardware target, and what we learned
about applying an AI agent to a new model architecture on a thin MCU
runtime.

## Project arc

Opened 2026-05-15 as a follow-on to the MV2 fusion experiment.  Ran
for ~6 days end-to-end across two distinct phases.

### Phase A — kernel build-out (Phases 0-7, ~3 days)

Built every transformer-specific kernel from scratch.  Each
"sub-phase" ended at a bit-exact end-to-end checkpoint:

| Phase | Op landed | Cortex-M op | Standalone kernel |
|---|---|---|---|
| 0 | scaffolding | — | host + FVP harness, round-trip stub |
| 1 | LayerNorm | new `cortex_m::quantized_layer_norm` + convert pass | `layer_norm_s8` (float-internal) |
| 2 | GELU | new `cortex_m::quantized_gelu` + convert pass | `gelu_lut_s8` (256-byte LUT) |
| 3 | BMM | cortex_m existing wrapper | `batch_matmul_s8` |
| 4 | Softmax | cortex_m existing wrapper | `softmax_s8` (float-internal, matches torch) |
| 5 | Fused attention | new `cortex_m::quantized_fused_attention` + `AttentionFusionPass` | `attention_fused_s8` (streaming, no S×S matrix materialized) |
| 6 | Full encoder block | `quantized_add` + `quantized_linear` (existing) | `add_s8`, `linear_s8` (CMSIS-NN kernel-sum form) |
| 7 | Codegen | `tools/extractors.py` + `tools/emitters.py` + `dump_kwt_1_artifacts.py` | one-shot dumper of weights, params, arena offsets, inference body |

Phase A end state: OneBlockKWT (`d=64, d_ff=256, S=8`) lowered →
dumper → standalone C runs bit-exact on host and Corstone-300 FVP.
**1.81 M PMU cycles** scalar baseline.

### Phase B — MVE tuning (Phase 8, ~1 day)

Six optimization steps each gated by bit-exact verification:

1. `linear_s8`: K-vectorize via `vmladavaq_s8` + tile-N=4
2. `add_s8`: MVE per-element + bit-exact vector requantize helper
3. `attention_fused_s8`: kernel-sum reformulation lets MVE handle QK^T
4. `linear_s8` (part 2): vector requantize + narrow-store at tile end
5. `transpose_s8`: rank-3 specialization (no MVE — just drop the rank-N idx walk)
6. `layer_norm_s8`: MVE FP4 across all three passes

Phase B end state: **327 k PMU cycles**, 5.52× over the scalar baseline,
bit-exact, on OneBlockKWT at d_ff=256.

### Phase C — scaling + architectural matching (Phases 9-11, ~1 day)

Stacking from 1 block to 12, swapping the bump allocator for
`exir.memory_planning.greedy`, adding the full KWT-1 architecture
(input embed + mean pool + classification head), and flipping
defaults to match the published KWT-1 paper (d_ff=128, optional pos
encoding + final LN paths).  ARM-software-checkpoint loader stub
shipped in `tools/load_arm_kwt1.py` for the day a trained `.h5` lands.

### Phase D — canonical KWT-1 optimization sprint (~1 day)

Once `--full-kwt1 --seq-len 98` ran end-to-end bit-exact on host, the
real shapes (S=98, K=40 embed, 12 stacked blocks) exposed paths the
Phase B kernels didn't cover.  Eight steps:

1. d_ff CLI default flipped 256 → 128 (paper KWT-1 dims)
2. K=64 input-vector hoist across N-tile loop
3. AV step MVE via zero-padded v buffer
4. Vector requantize tile-4 in AV
5. Vector requantize tile-4 in QK^T
6. Long-S MVE attention (`v_padded` sized to `ceil(S/16)*16`)
7. K-tail MVE for K not a multiple of 16 (embed Linear K=40)
8. M-tile-2 path for large M, K ≠ 64
9. QK^T D=64 q_row hoist
10. Softmax MVE FP for passes 1 and 3
11. v_col_sums + v_padded build MVE
12. Tighter `kwt_1_fast_expf` (floorf + fmaf, opt-in)

Phase D end state on canonical 1-block-S=98: **3.70 M cycles**
(default) / **3.62 M** (with `KWT_1_FAST_EXPF=ON`), down from 11.1 M
with the scalar-attention fallback.

## Headline results

| Configuration | PMU cycles | Bit-exact |
|---|---:|---|
| OneBlockKWT (S=8) scalar baseline | 1,806,386 | ✓ |
| OneBlockKWT (S=8) Phase 8 final (d_ff=256) | 327,377 | ✓ |
| OneBlockKWT (S=8) all post-Phase 8 work (d_ff=128) | **~211,000** | ✓ |
| 1-block @ S=98 (canonical input, 1 encoder block) | **3,704,182** | ✓ |
| 1-block @ S=98 with KWT_1_FAST_EXPF=ON | 3,616,918 | ✓ (random weights) |
| 12-block canonical (extrapolation from 1-block-S=98 anchor) | **~44 M** | — |

### Memory

The greedy planner amortizes activation storage across the 12 stacked
blocks: every block's intermediates die by the end of the block, so the
planner recycles the same slots.

| Configuration | Arena bytes |
|---|---:|
| OneBlockKWT (S=8) | 4,608 |
| KWT1Encoder (12 stacked OneBlocks, S=8) | **4,608** (same!) |
| Canonical KWT-1 (12-block, S=98) full model | 37,632 |

### Vs DS-CNN-S on the same FVP

| Model | Cycles | Top-1 | Classes |
|---|---:|---|---|
| DS-CNN-S | 1,902,290 | ~94 % | 12 |
| KWT-1 canonical | ~44 M | ~96.6 % | 35 |

**~23× the cycles for ~2.5 absolute accuracy points** on the harder
35-class task.

## What worked

| Technique | Where | Outcome |
|---|---|---|
| Hand-tuned MVE intrinsics per kernel | `kwt_1_kernels.c` | 5.5×–8.5× over scalar baseline |
| Bit-exact vector requantize (`kwt_1_mve_requantize_nonpos`) | linear, add, attention | enables the requantize chain to stay on MVE; matches CMSIS-NN's tie-away-from-zero rounding |
| K-vector + N-tile-4 in linear_s8 | every linear in the model | the load-bearing optimization; 4× input-load amortization |
| Vector requantize+narrow-store at tile end | linear, AV, QK^T | eliminates per-output scalar requantize cost |
| K=64 input-vector hoist | most linears (Q/K/V/O/FFN-expand) | -18% on K=64 linears by keeping the row's 4 vectors in registers across N-tiles |
| Long-S attention with `v_padded[D × ceil(S/16)·16]` | `attention_fused_s8` | unlocked canonical KWT-1 (S=98) — -57% on 1-block-S=98 |
| K-tail MVE for K not a multiple of 16 | embed Linear (K=40) | -65% on the K=40 linear alone |
| M-tile-2 for large M, K ≠ 64 | canonical KWT-1's Q/K/V/O at S=98 | shares 4 weight loads across 2 input rows |
| MVE FP4 across LN passes 1/2/3 | `layer_norm_s8_mve` | -71% per LN call |
| MVE FP4 across softmax passes 1+3 | `softmax_s8` | -14% on attention (98-iteration savings amortize) |
| Polynomial expf with floorf+fmaf | `kwt_1_fast_expf` (opt-in) | beats newlib's hardware-VFP expf on M55 with FMA |
| `exir.memory_planning.greedy` | dumper memory plan | 12 stacked blocks fit in the same 4.6 KB arena as 1 block |
| Save-ref reference workflow | `dump_kwt_1_artifacts.py --save-ref` | sidesteps PT2E calibration non-determinism by producing reference and headers in one Python invocation |
| Per-kernel cycle profile harness | `kwt_1_profile.h` + `dump_kwt_1_artifacts` emitter | every optimization gated by per-kernel PMU evidence, not vibes |

## What didn't pan out

| Investigation | Outcome | Why it failed |
|---|---|---|
| `linear_s8` N-tile-of-8 (vs tile-of-4) | -7% regression on K=64 | 8 weight loads per K-iter contended for L/S unit |
| `linear_s8` m-tile-2 for K=64 (any M) | +17% regression at M=8, +19% at M=98 | 8 simultaneous `vmladavaq_s8` saturate the M55 issue queue; trip count doesn't help |
| AV step with `vldrbq_z_s8` predicated load | Bit-exactness broke | gcc-13 + M55 didn't actually zero masked lanes; fixed with explicit zero-padded buffer |
| MVE `kwt_1_quantize_input` | Build error | toolchain doesn't expose `vdivq_f32`; `x * (1/scale)` would diverge from `round(x/scale)` in low bits |
| First `kwt_1_fast_expf` attempt | Slower than newlib expf | branchy range reduction + no FMA hints; later rewritten to beat newlib |

## Comparison with the MV2 standalone experiment

Both projects targeted Corstone-300 FVP with the same toolchain
(arm-none-eabi-gcc 13.3) and shipped from the same repo
(`examples/models/{mv2,kwt_1}_cortex_m_mve/`).  They differ in
fundamental ways that shaped the optimization process.

### Architecture surface area

| Axis | MV2 | KWT-1 |
|---|---|---|
| Kernel set | Conv2d, dwconv2d, avg_pool2d, gemv, add | LayerNorm, GELU, BMM, softmax, fused attention, linear, add, transpose, mean_dim |
| Kernels already in cortex_m / CMSIS-NN | All — the project competed against CMSIS-NN | None of the transformer-specific ones — had to author every op + convert pass + standalone kernel |
| Memory regularity | Highly regular (Conv has rectangular weight + bounded halo) | More irregular (attention has S×D × D×S × S contraction) |

KWT-1 needed Phase 0-6 just to *land* the kernel set in the cortex_m
backend at all.  MV2 jumped straight into optimization because every
op it needed already existed in CMSIS-NN form.

### Optimization process shape

| Axis | MV2 | KWT-1 |
|---|---|---|
| Number of optimization steps tracked | 20 named kernel commits + 3 fusion phases | 6 in Phase 8 + ~12 in Phase D |
| Speedup achieved | 17× scalar → MVE (after 20 steps) | 5.5× (Phase 8) → ~9× cumulative (Phase D extension) |
| Final beat vs runtime alternative | 1.13× faster than cortex_m + CMSIS-NN at MV2 | 23× *slower* than DS-CNN-S at the KWS task — KWT-1 was solving a different problem (35 vs 12 class) |
| Fusion phases | 3 (2-op, 3/4-op, B0 stem chain) | 0 — the encoder block topology has no analog of MV2's inverted-residual fusion opportunity |
| Pareto sweep | 20 FVP configs at (width × resolution) | 1 config (canonical KWT-1 dims) |

MV2's optimization went deeper because the kernel set was small (5
ops) and the model used them many times (35 conv2ds, 17 dwconvs),
so each kernel commit affected many call sites.  KWT-1's kernels
fire fewer times per inference but do more compute per call (long
sequences in attention, larger linears).

### Memory results

| Axis | MV2 | KWT-1 |
|---|---:|---:|
| Arena, headline config | 390 KB (after 3 fusion phases, full canonical input) | 38 KB (greedy planner alone) |
| Memory reduction technique | AOT operator fusion (4 fused ops absorb 17 unfused ones) | Just slot reuse via `exir.memory_planning.greedy` — block intermediates die at block boundary |

KWT-1's arena is **10× smaller** than MV2's not because of better
optimization but because of the model shape: each encoder block's
intermediates die at the block boundary, so the greedy planner
recycles the same slots 12 times.  MV2 has long-lived residual
tensors that prevent slot reuse without explicit fusion.

### Where the "competing" model wins

- **MV2 vs CMSIS-NN at MV2's own task**: the standalone path *won*,
  by 1.13× at full input and a growing margin at smaller widths.
  This was the project goal — show the standalone path is
  competitive.
- **KWT-1 vs DS-CNN at the KWS task**: the standalone path *lost*
  by 23×.  But that's not a fair comparison — KWT-1 was solving a
  fundamentally larger problem (35 classes vs 12, transformer
  flexibility vs CNN feature pyramid).  The right comparison is
  "what does KWT-1 cost on cortex_m vs nothing being possible
  before" — and the answer is "44 M PMU cycles, transformer KWS
  enabled."

## Learnings extracted across both projects

### 1. MVE optimization patterns transfer across architectures

The same three tricks shipped to both projects:

- **Tile by 4 outputs + scalar accumulators** for vmladavaq_s8 (linear
  in KWT-1, conv1x1 in MV2)
- **Hoist input vectors across the N-tile loop** when register
  budget allows (K=64 hoist in linear_s8, similar trick in MV2's
  conv1x1)
- **Vector requantize + narrow-store at tile end** to eliminate per-
  output scalar overhead

The MVE intrinsic surface area is small enough that ~20 patterns
cover almost any int8 model.

### 2. Predicated MVE loads are unreliable on this gcc-13 + M55 combo

In KWT-1, `vldrbq_z_s8` with `vctp8q(S)` failed to actually zero the
masked-out lanes — the AV step diverged.  The workaround (explicit
zero-padded scratch buffer, unpredicated load) costs ~1-7 KB of stack
per call but works reliably.  Worth checking other gcc + MVE
combinations or filing upstream.

### 3. Issue-queue saturation is a real constraint on Cortex-M55

`linear_s8` m-tile-2 with 8 simultaneous `vmladavaq_s8` accumulators
regressed at every M we tried.  The dispatcher can't keep that many
scalar destinations in flight.  Stay at 4 accumulators per inner;
above that, the compute throughput drops below the load-amortization
gains.

### 4. Float ops on M55+VFP are surprisingly fast

Initially expected `expf` to be a serious bottleneck — turned out
newlib's hardware-VFP `expf` is ~50 cycles, and a tight polynomial
substitute (floorf + fmaf Horner) only beats it by maybe 2×.  Same
held for LayerNorm: the float-internal version vectorizes through
the same MVE FP4 path the polynomial expf uses, and the rsqrt + mean
+ variance reductions all land near peak.  No reason to fight to
keep transformer-flavored math in int8 when MVE FP4 is right there.

### 5. The cost of "bit-exact at every step" is small; the value is huge

Both projects gated every optimization on bit-exact output vs the
python reference.  Twice in KWT-1 (the AV predicated load, the
positional-encoding add path) and several times in MV2, this caught
real bugs before they accumulated.  The discipline costs maybe 30
seconds per iteration (run the verify script); skipping it would
have produced "5× faster on canonical input, wrong on edge cases"
results that take days to debug.

### 6. PT2E calibration is non-deterministic

KWT-1 hit this hard: running the dumper twice produced different
quant multipliers because of an unknown source of non-determinism
inside the quantizer.  The fix was the `--save-ref` workflow —
produce the reference int8 output in the *same* Python invocation as
the C headers, so they always agree.  Probably worth lifting into
the mv2 dumper too.

### 7. Memory planner does more heavy lifting than expected

`exir.memory_planning.greedy` produces ~optimal allocations for both
models without any project-side tuning.  The 12-stacked-block KWT-1
fits in the same arena as 1 block because the planner figured out the
liveness story.  Custom planners considered for MV2 — explicitly
abandoned because greedy was already at the interval-graph lower
bound.

### 8. Per-kernel cycle profiling is the load-bearing dev tool

Both projects' optimization sprints depended on a per-kernel cycle
profile harness (PMU CCNTR snapshots wrapped around each kernel
call).  Without it, "the model is slow" is unactionable.  With it,
every optimization step picks the next target deterministically.
The harness is ~50 lines per project and pays for itself in the
first optimization iteration.

### 9. Architectural similarity to existing models determines effort budget

MV2's optimization was a 3-week project because CMSIS-NN already
shipped every kernel it needed; the work was making them faster /
fuse better.  KWT-1's was a 6-day project because the kernels didn't
exist *anywhere* — Phase 0-6 was just standing them up.  When
sizing a "port model X to MCU" project, look at the kernel inventory
first; if CMSIS-NN doesn't have it, multiply your estimate by ~3.

### 10. Transformer cost is concentrated where MVE struggles

By the end of KWT-1's optimization, **softmax (expf-bound) and
attention's per-row glue** were 30%+ of the canonical-KWT-1 budget,
and MVE doesn't help either.  The remaining 70% is dense linear
algebra at near-peak MVE throughput.  Future transformer-on-MCU
gains likely require either:
- A vector expf (not in current MVE-FP), or
- Algorithmic restructuring (multi-row softmax batching, attention
  approximation like Performer / linear attention), or
- Different hardware (Ethos-U on the same chip handles the dense
  parts much faster, removing the optimization budget for the
  softmax tail).

## Recommended next direction

Ordered by expected leverage:

1. **Load real ARM-software KWT-1 weights** (already stubbed via
   `tools/load_arm_kwt1.py`) and produce an actual Speech Commands
   top-1 number.  The biggest remaining unknown — current numbers
   are random-weight bit-exact only.

2. **Real-silicon validation** (same recommendation as MV2's
   SUMMARY).  Closes the FVP-vs-deployment gap.

3. **Extend the loader to Conformer / Squeezeformer** — the kernel
   set this directory ships covers transformer encoders broadly.
   Conformer is "KWT + convolutional augmentation"; should mostly
   compose.

4. **Multi-row softmax batching** — the one large remaining
   optimization vector for transformers on M55.  Batch 4 query rows
   together and vectorize the expf polynomial across 4 lanes of
   MVE FP4.  Bit-exactness risk; significant restructuring; likely
   only worth it if real KWT-1 deployment numbers turn out to be
   latency-bound (i.e., real weights matter and 44 M PMU isn't fast
   enough).

5. **Apply the standalone-path approach to another transformer**
   (Whisper, encoder of small audio LLMs, vision transformer for
   embedded camera).  The kernel set generalizes; the question is
   whether the AOT plumbing does.

Authored with assistance from Claude (claude.ai/code).
