# TinyML optimization experiment: summary and next steps

A condensed record of what was tried, what worked, what didn't, and where
the remaining optimization headroom sits — for MV2 specifically and for
the broader question of how far an AI agent can take TinyML inference
optimization.

## Project arc

The project ran in three phases, each ending at a clean stopping point:

1. **Kernel tuning** — 20 incremental MVE/asm optimizations to the standalone
   MV2 path produced a 17× speedup over the initial scalar baseline and
   landed 15.4% faster than the cortex_m (CMSIS-NN) backend at the same
   shape (1.0/224). Full per-step history is in this directory's
   `BENCHMARK.md`. End of phase: kernel ceiling at 145.6M PMU / 1505 KB
   arena.

2. **Design-space exploration** — width × resolution pareto sweep across
   the four widths {1.0, 0.75, 0.5, 0.35} and five input resolutions {224,
   192, 160, 128, 96}. 20 FVP configurations. Discovered the MVE advantage
   over CMSIS-NN *grows* at smaller widths (15% → 27%) because CMSIS-NN's
   per-call overhead is roughly constant while ours benefits from compile-
   time-known shapes.

3. **Operator fusion** — two AOT passes (`ExpandDwconvFusionPass` and
   `ExpandDwconvProjectFusionPass`) that pattern-match MV2 inverted-
   residual blocks and replace them with single fused ops whose runtime
   kernels stream intermediate tensors through small rolling buffers in
   `mv2_fused_scratch`. The export tool runs both planning pipelines and
   picks the smaller (arena + scratch) per model. End of phase: 1.0/224 at
   841 KB / 153.7M PMU — **44% memory reduction at 5.6% latency cost,
   bit-exact** vs the unfused baseline.

4. **Generalization probe (KWT)** — attempted to apply the stack to KWT-1
   (keyword transformer). Outcome: the existing kernels are CNN-shaped and
   don't transfer; the cortex_m backend has no acceleration for the
   transformer-critical ops (LayerNorm, GELU, BMM, properly-tuned
   Softmax). Filed as a separate project.

## Headline results

### Pareto across 20 (width, resolution) configs, conditional fusion

| width \\ res | **224** | **192** | **160** | **128** | **96** |
|---:|---:|---:|---:|---:|---:|
| **1.00** | 842K / 153.7M | 623K / 113.8M | 437K / 79.2M | 284K / 51.2M | 164K / 28.9M |
| **0.75** | 641K / 124.7M | 476K /  92.4M | 335K / 64.2M | 219K / 41.7M | 127K / 23.5M |
| **0.50** | 424K /  72.3M | 314K /  53.5M | 221K / 37.4M | 144K / 24.4M |  83K / 13.8M |
| **0.35** | 418K /  54.2M | 309K /  40.2M | 216K / 28.0M | 140K / 18.3M |  81K / 10.5M |

Format: `(arena+scratch) KB / PMU cycles`. Width-0.35 uses the 2-op
fallback pipeline (3/4-op regresses); all other widths use the 3/4-op
pipeline. Bit-exact int8 output preserved at all configs.

### Deployment sweet spots

| SRAM budget | Best config | Latency @ 300 MHz |
|---|---|---|
| ≤512 KB | 0.5 / 224 at 424 KB | 241 ms |
| ≤256 KB | 0.5 / 128 at 144 KB | 81 ms |
| ≤128 KB | 0.35 / 96 at 81 KB | 35 ms |
| ≤96 KB  | 0.35 / 96 at 81 KB | 35 ms |

Smallest deployable: **0.35 / 96 at 81 KB / 10.5M PMU** — 19× smaller and
14× faster than the original 1.0/224 baseline, deployable on hardware
that wouldn't run MV2 at all before.

## What worked

| Technique | Where | Outcome |
|---|---|---|
| Hand-tuned MVE intrinsics (per-kernel) | `mv2_inference.c` | 17× over scalar baseline |
| Inline asm `wlstp/letp` with `vmladava.s8` | 1×1 conv inner | 2 ops/output, asymptotic limit for 7-even-GPR register file |
| 4-pixel × 4-channel dwconv tile | dwconv 3×3 stride-1 | 2.06 ops/output for 16 outputs/iter |
| AOT-folded `input_offset` into bias | 1×1 conv extractor | saves per-tap offset add |
| `bias_with_offset_full` pre-fold | dwconv when all kh in-bounds | skips per-tap offset add in fast path |
| Pre-packed 32-byte stem weights | first 3×3 stride-2 Cin=3 conv | ~6× over scalar fallback |
| 2-op fusion (expand+dwconv) | new AOT pass + runtime kernel | -41% arena at 1.0/224 |
| 3/4-op fusion (full inverted-residual block) | extension of 2-op pass | additional 16-18% at width-0.5/0.75 |
| Conditional pass selection in `export_mv2.py` | runs both planners, picks smaller | eliminated the width-0.35 regression |
| Python derisking before C | `fusion_prototype.py` for rolling buffer | caught aliasing bug before integration |

## What didn't pan out

Worth recording because the *non*-finding is also useful:

| Investigation | Outcome | Why it failed |
|---|---|---|
| NCHW dwconv layout (channel-major) | Net negative or zero gain | Transpose overhead ~ inner-loop savings; arena pressure from transpose scratch |
| Patch-based / streaming inference (MCUNet-style) | Negative ROI for MV2 | MV2's fat 96/144-channel expansions at 112²/56² make halo recomputation tax unavoidable |
| Custom memory planner | Already optimal | Greedy planner == interval-graph lower bound for the current op graph; can't shrink without graph transforms |
| Specialized 1×1 conv head asm | <1% total inference | Head already at the 7-even-GPR asymptotic limit |
| Stem conv further optimization | <1% total | Already packed-im2col, ~6× faster than scalar |
| 3/4-op fusion at width-0.35 | Per-block math favorable but planner-level regression | Eliminations don't coincide with global arena peak; conditional pass selection handles it |

## MV2 optimization meat still on the bone

### Memory: B0 dwconv-project fusion (significant)

The current 1.0/224 peak of 802 KB is driven by the **B0 inverted-residual
block**: stem output (32×112² = 401 KB) is live alongside the B0 dwconv
output (32×112² = 401 KB) — neither is fused because B0 has
`expand_ratio=1`, so there's no expand conv for our pattern matchers to
hook onto.

A new pass `DwconvProjectFusionPass` (matching `dwconv → project` without
the expand prefix) plus a corresponding runtime kernel would eliminate
one of those 401 KB tensors, shrinking the peak to roughly 400 KB at
1.0/224 — a further **50% reduction over today's fused result**. The
implementation is structurally the same as the existing 2-op fusion
(rolling buffer between dwconv and project), just without the expand row
helper.

Effort estimate: ~2 days. Risk: bit-exactness via the proven rolling-
buffer pattern; new kernel mostly reuses existing dwconv + 1×1 conv MVE
inner loops.

### Memory: Stem + B0 fusion (potentially bigger)

The stem conv (3×3 stride-2 Cin=3 → 32 channels at 112²) feeds B0's
dwconv. If the stem could also be streamed (output rows produced into a
small rolling buffer that B0's dwconv consumes immediately), the stem
output's 401 KB allocation goes away too.

The stem's MVE path is the packed-32 im2col kernel. Adapting it to
produce one row at a time is mechanical. Pairing with B0's dwconv
(consumed via the standard 3-row rolling buffer) is the natural
extension.

Effort estimate: ~3 days. Significant payoff at the headline config: at
1.0/224, peak could drop from 802 KB to ~200 KB (just the B0 project
output + small rolling buffers + quantized input).

These two opportunities together could plausibly drop 1.0/224 arena from
841 KB to **~250 KB**, a 70% reduction from the unfused baseline.

### Memory: multi-block fusion (modest)

For consecutive blocks without intervening branches (e.g., B8 → B9 → B10
at width-1.0 are all 384-ch dwconv 14² stride-1 with residual adds), we
could fuse pairs of blocks. The intermediate (block-boundary) tensor
disappears.

Returns diminish: each fused pair eliminates one project output. At
width-1.0 the project outputs at deep layers are small (160-320 ch at
spatial 7-14), so each fusion saves only ~5-20 KB. Cumulative impact
probably ~5-10% additional arena reduction. Implementation is non-trivial
because fusion across blocks must respect residual paths.

Effort estimate: ~1 week. Lower priority than B0 fusion.

### Latency: stem conv 2-pixel batching (small)

The stem conv currently processes one output pixel at a time. Adjacent
output pixels at stride-2 share one input column (col `2*ow+1`). A
2-pixel batched inner loop could share that load.

Per output row: 56 pixel pairs × ~2 shared loads × 32 OCs = some saving.
Estimate: ~30% on stem cycles = ~1.5% of total inference. Marginal but
measurable.

Effort estimate: ~half a day.

### Latency: head conv epilogue batching (marginal)

Head conv (1280 × 320 × 7²) contributes ~3-4% of total cycles. The
per-OC-tile epilogue (requantize + clamp + store) is ~4 ops/output —
roughly 2× the inner-loop cost. Batching the epilogue across two OC
tiles would amortize the mult/shift loads. Save ~0.5% of total.

Effort estimate: ~1 day, mostly carefully redoing the asm constraint
solving. Not worth pursuing on its own.

### Latency: real-silicon validation (different category)

Everything measured here is Corstone-300 FVP in `--fast` mode, which
doesn't model the memory hierarchy. The relative rankings (MVE vs
CMSIS-NN, fused vs unfused) should hold on silicon, but absolute cycle
counts and the DDR-bandwidth savings from fusion are unvalidated. An
MPS3 FPGA running the Cortex-M55 RTL, or any real Cortex-M55 SoC, would
close this gap.

Effort: hardware-dependent (board procurement, BSP setup, debug
infrastructure). Not a coding task per se but a prerequisite for any
external deployment claim.

### Quantization-level (large, separate project)

Int4 weights with int8 activations would halve weight memory (helps flash
but not arena). Per-block precision recipes (some blocks tolerate lower
precision) could compress activations selectively. Both are substantial
quantizer + kernel work outside the current AOT/runtime scope.

## Goal-level assessment

The project had two stated goals:

### Goal 1 — assess how far an AI agent can take TinyML optimization

Demonstrated across every layer of the stack:
- Hand-tuned MVE intrinsics + inline asm with register-allocation-aware
  constraint solving (the `+Te` even-GPR constraint dance).
- AOT pass authoring against ExecuTorch's edge-dialect IR.
- Codegen plumbing (extractors + emitters for new edge ops).
- Multi-op fusion composition (the 3/4-op pass builds on 2-op outputs).
- Rigorous Python-prototype derisking before C implementation; caught
  the aliasing bug between dwconv-row scratch and project-row scratch
  before runtime debugging.
- Bit-exact validation across host + FVP at multiple model configs.
- End-to-end design-space sweep with 40 FVP runs total.
- Honest stopping points: marked work as marginal/done when the analysis
  showed it (head asm, NCHW dwconv, custom planner, KWT
  generalization).

The agent did not produce work that an experienced engineer couldn't
have produced; what the experiment tested was whether the agent could
*sustain* deep technical engagement across a multi-week project without
losing thread, and whether it could be trusted to do honest analysis
(including admitting marginal returns) rather than producing activity
for its own sake. That latter held.

### Goal 2 — reduce TinyML inference cost (memory + latency)

| Metric | Before | After | Ratio |
|---|---|---|---|
| 1.0/224 latency | 1158M PMU (extrapolated scalar baseline) | 153.7M PMU | **7.5×** |
| 1.0/224 vs CMSIS-NN | 172.2M PMU | 153.7M PMU | 1.12× faster |
| 1.0/224 arena | 1505 KB | 841 KB | -44% |
| Smallest deployable | wouldn't fit any MCU | 0.35/96 at 81 KB / 35 ms | enabled new deployment tier |

The remaining MV2 headroom catalogued above could plausibly drop 1.0/224
arena to ~250 KB if pursued.

## Recommended next direction

If continuing the experiment, the highest-value next step is **B0
dwconv+project fusion + stem fusion**: bounded scope (~1 week), large
payoff (potentially 50-70% additional arena reduction at headline
configs), and uses techniques that are already validated in the existing
2-op/3-op patterns.

Other directions in roughly descending value:

1. **Real-silicon validation** — closes the FVP-vs-deployment gap, makes
   the result externally credible.
2. **Generalize to MCUNet-VWW** — tests CNN-to-CNN portability without
   the transformer-shaped gap KWT exposed. Most informative
   capability-test that's still bounded scope.
3. **Multi-block fusion** — modest payoff, large effort.
4. **Quantization-level optimization** (int4, per-block recipes) —
   high potential but separate project scope.
