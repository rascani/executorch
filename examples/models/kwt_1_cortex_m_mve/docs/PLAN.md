# KWT-1 / Transformer Support on cortex_m + Standalone MVE — Implementation Plan

Authored 2026-05-15 as the kickoff plan for the transformer-class
model support effort, following the MV2 fusion experiment (Phase G)
reaching a stopping point.

## Executive Summary

KWT-1 is a 12-layer pure transformer encoder over MFCC features
(d_model=64, nhead=1, dim_ff=256, seq_len=99, ~607K params, 35
classes). The existing standalone MV2 pipeline (Phase G) is
**CNN-shaped end-to-end**: every fused op composes from {stem,
conv1x1, dwconv3x3, add, avgpool, gemv}. KWT-1 reuses **none** of
those — its inner block is `LN → QK^T → softmax → AV → linear →
residual → LN → linear → GELU → linear → residual`.

The existing probe (`examples/models/kwt_1/export_kwt_1_cortex_m.py`)
already proves the *lowering* path completes; what's missing is the
kernel set on **both** paths (cortex_m backend kernels for the
runtime path, standalone MVE C kernels for the bare-metal path)
plus the AOT plumbing that prepares quant params, plans memory, and
fuses ops.

State check on what cortex_m **already has** (good news):
- `cortex_m::quantized_batch_matmul` (wraps CMSIS-NN
  `arm_batch_matmul_s8`) — ready.
- `cortex_m::softmax` (wraps CMSIS-NN softmax_s8) — ready.
- `cortex_m::quantized_linear`, `quantized_add`,
  `quantize_per_tensor`, `dequantize_per_tensor` — ready.
- `convert_to_cortex_m_pass.py` already rewrites
  `aten.bmm.default` → `cortex_m.quantized_batch_matmul` and
  `aten.linear.default` → `cortex_m.quantized_linear`.

What's **missing** on the cortex_m backend:
- LayerNorm (no op, no quantizer pattern, no convert-pass rewrite).
- GELU (no op, no quantizer pattern; cortex_m's
  activation_fusion_pass folds clamp-style activations into
  preceding ops — GELU is not handled).
- Quantizer coverage for the transformer pattern (Linear→Add→
  LayerNorm residuals, QK^T scaling, AV path, GELU).

What's missing on the **standalone path** (the harder side):
- Everything. No transformer kernels (LN, GELU, BMM, softmax), no
  extractor/emitter for these ops, no MVE-tuned softmax/exp
  helpers, no float-stays-resident fallback path (today the
  standalone path is pure int8 after the stem qin).

The plan rolls out in phases that **each end at a runnable,
bit-exact-validated checkpoint**, mirroring MV2's A→B→F→G shape.
Each phase ships an AOT pass + a hand-written C kernel (scalar
first, then MVE) + a kernel-only unit test before the next phase
composes on top. Memory-first ordering throughout — kernels that
materialize the (S×S) attention score matrix are designed for
streaming from day one.

---

## Phase 0 — Scaffolding & validation harness (no kernels yet)

Goal: stand up the project tree, the FVP host-scalar build, and
the bit-exact validation strategy *before* writing any kernels.
End state: an empty runner that loads a fixture and asserts
equality on a known tensor — green CI smoke.

- New directory `examples/models/kwt_1_cortex_m_mve/` paralleling
  `mv2_cortex_m_mve/` with subtree `{tools/, src/, include/,
  fvp/, generated/, docs/}`.
- Copy & adapt: `fvp/toolchain-arm-none-eabi.cmake`,
  `src/runner_fvp.cpp`, `src/runner_host.c`, `src/arena.c` from
  MV2 verbatim (no transformer logic in these).
- Pick a pretrained checkpoint. The arxiv KWT-1 reference points
  at training on the Google Speech Commands v2 dataset (35-class).
  Concrete options to land in this phase:
  1. The author's open-source release
     (`ARM-software/keyword-transformer`) — train recipe +
     Pareto weights.
  2. Reproduce a quick training run on Speech Commands v2 (8-16
     hrs on a single GPU; the model is tiny).
  Decision: ship a small training script
  (`tools/train_kwt_1.py`) AND wire it to accept the upstream
  Arm-software checkpoint if the file is present. Validation runs
  against a pinned 50-sample fixture so plan phases without a
  trained net can still test bit-exactness.
- MFCC frontend decision: **MFCC done host-side; the model graph
  starts at the post-MFCC `(1, 1, 40, 98)` tensor**. Rationale:
  MFCC is a stable, well-tested DSP block; today it isn't
  expressible as int8-quantizable PyTorch ops cleanly, and forcing
  it into the graph would block the project on a DSP problem
  orthogonal to the transformer kernels. The runner harness gets
  a tiny `mfcc.c` (CMSIS-DSP `arm_rfft_q31` + Mel filterbank LUT +
  log + DCT) that runs once on raw int16 PCM and produces the
  (40, 98) float32 input. Phase Z (stretch) folds MFCC into the
  model graph for true raw-audio-in deployment.
- Validation strategy follows MV2 exactly:
  - **Reference**: Python forward through the quantized exported
    program (via `cortex_m::*` python `impl` registrations —
    these already exist and are bit-exact). Dump (input, output)
    fixture as int8 tensors.
  - **Host scalar build**:
    `cmake -DMV2_BUILD_HOST=ON`-equivalent runs the same kernel C
    files compiled for x86, reads the fixture, asserts
    `max int8 diff = 0` against the Python reference. This is
    the **primary bit-exactness gate**; FVP is for cycle
    measurement only.
  - **FVP MVE build**: same kernels, MVE codepaths active, run
    on Corstone-300 — must also produce `max int8 diff = 0`
    against the same fixture.
- Add `backends/cortex_m/test/models/test_kwt_1_standalone_mve.py`
  mirroring `test_mv2_standalone_mve.py`. In Phase 0 it just
  builds the empty runner and round-trips a memcpy through the
  arena.

**Checkpoint**: pytest green on host & FVP. No transformer math yet.

---

## Phase 1 — LayerNorm (the keystone)

The most numerics-sensitive op in the set. LN is small in op-count
(~2% of cycles in TinyML transformers) but, because every
transformer block starts and ends with one, every downstream op's
bit-exactness depends on LN being right. Land it first, in
isolation, on a one-block test model.

### 1A — LayerNorm AOT (cortex_m backend)
- New op `cortex_m::quantized_layer_norm` and
  `cortex_m::quantized_layer_norm.out` in
  `backends/cortex_m/ops/operators.py` and `operators.yaml`.
  Signature: `(Tensor input, int input_zero_point,
  float input_scale, Tensor weight, Tensor bias, float eps,
  int output_zero_point, float output_scale) -> Tensor`.
  Weight/bias are float32 baked-in constants (LN's γ, β); they
  sit in `.rodata`.
- **Numerics decision: LN stays float internally**. The op
  dequantizes int8 input to float32 in a small row-streaming
  loop, applies LN in float, requantizes to int8. Memory-first
  justification: an int8 LN with the rounding compensation needed
  to match PyTorch's LN to ±0 is fragile; the float math costs
  ~6×N extra ops per row (N=64 for KWT-1, seq_len=99 → 6,336
  extra FMAs/block, ~20K cycles vectorized — under 1% of
  inference budget). Bit-exactness is then easy (compare against
  torch's `aten.native_layer_norm` in float and matching the
  requantize against the python reference).
- Python `impl` runs torch's `aten.native_layer_norm` after
  dequant, then requantizes. This is the bit-exact reference.
- Quantizer extension: `CortexMQuantizer` needs an LN pattern. LN
  sits between two int8 tensors but the inner math is float; the
  quantizer treats the LN as a "boundary" op with input observer
  (from upstream Linear) and output observer (feeding downstream
  Linear). Add `LayerNormPattern` to
  `backends/cortex_m/quantizer/quantization_configs.py` returning
  per-tensor int8 specs on both sides.
- Convert pass: add `_get_layer_norm_replacement` to
  `convert_to_cortex_m_pass.py` that pattern-matches
  `dequant → aten.native_layer_norm → quant` and folds the quant
  params into the new op.

### 1B — LayerNorm runtime: cortex_m backend kernel
- New `backends/cortex_m/ops/op_quantized_layer_norm.cpp`.
  Implementation: per-row dequant to a small scratch float buffer
  (size = embed_dim, 64 floats = 256 B), Welford mean/variance,
  multiply-add against γ/β, requantize to int8. Pure C++ scalar —
  the cortex_m backend mirrors CMSIS-NN style; the standalone
  path is where the MVE work happens.
- Unit test `backends/cortex_m/test/ops/test_layer_norm.py` —
  random inputs across (B, S, D) shapes, assert bit-exact against
  the python impl.

### 1C — LayerNorm runtime: standalone MVE kernel
- Add to `examples/models/kwt_1_cortex_m_mve/src/kwt_1_kernels.c`
  (new file):
  `static void layer_norm_s8(const int8_t* input,
  int8_t* output, const LayerNormParams* p)`. Params struct in
  `kwt_1_layer_params.h` holds: input_zp, input_scale (float),
  embed_dim, output_zp, output_scale (float), eps (float),
  γ_ptr, β_ptr.
- **MVE shape**: process 4 channels at a time with `vldrbq_s32`
  (load 4 int8s, sign-extend), float-convert via
  `vcvtq_f32_s32`, accumulate sum and sum-of-squares in two
  `float32x4_t` accumulators per row. Mean/variance reduce with
  `vaddvq_f32`. Rsqrt either via Newton iteration (one
  `vrsqrteq_f32` + one `vrsqrtsq_f32` step ≈ 22-bit precision,
  sufficient for int8) or via `1.0f / sqrtf(variance + eps)`
  scalar (cheap because it's once per row). γ/β load as
  `float32x4_t` and apply with one FMA. Requantize: float×scale
  → int32 → narrow saturating store. ~80 MVE ops per 64-channel
  row.
- New extractor `extract_quantized_layer_norm` and emitter for
  the LayerNormParams struct.

### 1D — Test model & validation
- New `examples/models/kwt_1_cortex_m_mve/tools/test_models.py`
  with a `OneBlockKWT` module: just LN + a frozen Linear (so the
  export pipeline has a real edge to land at). Export, dump
  artifacts, build host + FVP, assert bit-exact.

**Checkpoint**: one-block test model runs end-to-end, host scalar
= FVP MVE = python reference, max int8 diff = 0.

---

## Phase 2 — GELU

GELU is **pointwise**, **stateless**, and (unlike LN) has no
parameters. Lands quickly as a pure activation. Critical numeric
decision: GELU's tanh-approximation form vs the erf form.
PyTorch defaults to erf; the `approximate='tanh'` form is much
cheaper on MVE and matches what TFLite/CMSIS-NN style int8
transformers ship.

### 2A — AOT
- New op `cortex_m::quantized_gelu` with int8 in / int8 out +
  input/output quant params, plus a `tanh: bool` flag
  (default True).
- Python `impl` runs the tanh-approx GELU after dequant.
- Quantizer pattern in `quantization_configs.py`: GELU is treated
  as a stateless activation, observed input + observed output.
- Convert pass: match `dequant → aten.gelu (approx='tanh') →
  quant`. Replace `aten.gelu` defaulting to erf form with
  tanh-approx via a small pre-pass
  `decompose_gelu_to_tanh_pass.py` (mirrors the existing
  `decompose_hardswish_pass.py`).

### 2B — Runtime: cortex_m kernel
- New `op_quantized_gelu.cpp` — scalar-only on cortex_m backend;
  LUT-driven int8 GELU is the natural form (256 entries, computed
  once at AOT, baked into the op). Sub-100 line C++.

### 2C — Runtime: standalone MVE
- **LUT-based**, not float. Build a 256-byte int8 LUT at AOT time
  encoding `quantize(gelu(dequantize(x)))` for x in
  `[-128, 127]`. The LUT is per-(input_zp, input_scale,
  output_zp, output_scale) — bake it per-op into `.rodata`.
- MVE kernel: 16-wide `vldrbq_s8` input load, scatter-gather LUT
  lookup. MVE has `vldrbq_gather_offset_s8` for 8-bit gather —
  exactly the right primitive. One instruction per 16 elements.
  This is faster than computing GELU in MVE float.
- Memory cost: 256 B per distinct (zp, scale) tuple per op. With
  12 layers × 2 GELU activations per FFN block but identical
  scales across blocks, deduplication keeps total LUT memory to
  ~512 B - 3 KB.

### 2D — Test
- Extend `OneBlockKWT` to include the GELU activation. Validate
  as before.

**Checkpoint**: LN + GELU compose. Bit-exact host & FVP.

---

## Phase 3 — Batched matmul (QK^T and AV)

The CMSIS-NN `arm_batch_matmul_s8` already exists and the cortex_m
op (`quantized_batch_matmul`) already wraps it. So **Phase 3
cortex_m work is zero net new code** for the runtime side — it's
just plumbing.

### 3A — AOT (mostly already done)
- `convert_to_cortex_m_pass.py` already rewrites
  `aten.bmm.default`. Verify it fires on both QK^T and AV
  positions when `nn.TransformerEncoderLayer` is decomposed (the
  export needs `preserve_ops=[aten.linear.default]` — already in
  the probe).
- **Subtle**: KWT-1 with `nhead=1` means the QK^T is
  `(B, S, d) @ (B, d, S) = (B, S, S)`. With nhead>1, attention
  reshapes to `(B, nhead, S, d_head)` and the QK^T is per-head.
  The convert pass and BMM op need to handle the 3-D input as
  `(B*nhead, S, d_head)` — KWT-1's nhead=1 makes this trivial;
  conformer stretch (later) will need a careful look.
- Quantizer extension: BMM in attention needs to be observed at
  the input side **with a known input_scale tied to the
  softmax_input requirement**. Today, the existing
  `arm_batch_matmul_s8` produces int8 output; for the QK^T path we
  want the BMM output to feed softmax with a specific output scale
  matching CMSIS-NN softmax's expectations. Add
  `AttentionBmmPattern` to the quantizer that pins the QK^T
  output scale.

### 3B — Runtime: cortex_m
- The existing CMSIS-NN-backed `op_quantized_batch_matmul.cpp`
  should just work for KWT-1's shapes. **Verify** for the
  (1, 99, 64) × (1, 64, 99) → (1, 99, 99) and the
  (1, 99, 99) × (1, 99, 64) → (1, 99, 64) shapes. The Python
  impl already handles arbitrary 3-D shapes.

### 3C — Runtime: standalone MVE BMM
- This is the meaty kernel. New `batch_matmul_s8` in the
  standalone runtime. Two flavors needed because the access
  patterns differ:
  - **Case 1 (QK^T)**: `lhs (1, S, d) × rhs_T (1, S, d) →
    out (1, S, S)`. d=64 is the inner reduction. With d=64 = 4×
    MVE int8 reduction width (16), each `vmladavaq_s8`
    accumulator covers 16 elements; 4 vmladava's per output
    element with a `wlstp.8 lr, #64` tail-predicated loop,
    exactly like the conv1x1 inner loop from MV2 step 11.
  - **Case 2 (AV)**: `lhs (1, S, S) × rhs (1, S, d) →
    out (1, S, d)`. Here S=99 is the inner reduction. Same
    `wlstp.8` pattern but with tail-predicated inner of length 99
    (= 6× 16 + 3-elem tail).
- Memory-first: do **NOT** materialize the full (S, S) score
  matrix in the arena. Stream row-by-row. For KWT-1, S=99 so the
  full matrix is 99² = 9.8 KB — trivial. **But for conformer
  stretch S could hit 1000+** (1 MB matrix). Build the streaming
  path now even though it's overkill for KWT.
- Streaming design: produce 1 row of QK^T at a time → apply
  softmax to that row (Phase 4) → use the softmax row as the lhs
  row of AV → write 1 row of AV out. This is fused attention;
  commit it as a separate fusion phase (Phase 5).

### 3D — Test
- `TwoLinearBmmAttention` test model: `Linear(d→3d) → split to
  (q,k,v) → bmm(q, k.T) / sqrt(d) → softmax → bmm(_, v) →
  Linear(d→d)`. Validate. This is a no-LN, no-residual attention
  head — minimal test of BMM + softmax composition.

**Checkpoint**: attention head runs bit-exact on both runtimes.

---

## Phase 4 — Softmax (transformer-tuned)

cortex_m already has `cortex_m::softmax` wrapping CMSIS-NN's
`arm_softmax_s8`. For the cortex_m runtime path, that's done.
**For standalone, softmax needs new MVE code.**

### 4A — Standalone softmax MVE
- **Shape**: attention softmax operates on rows of length S (99
  for KWT-1). Each row is independent → trivial parallelism axis.
- **Algorithm**: standard max-subtract + exp + normalize, in
  fixed-point. CMSIS-NN's reference does this with
  `arm_nn_exp_q15` fixed-point exp. Mirror that, MVE-vectorized:
  1. Max reduction across the row: `vldrbq_s8` + `vmaxq_s8`
     chain, final `vmaxvq_s8`. ~99 / 16 = 7 vector ops.
  2. Subtract max + multiply by `(input_scale * log2e_q15)` for
     fixed-point exp input. MVE `vsubq_s8` + `vmulhq_s32` style.
  3. Fixed-point `exp_q15` per element — this can be a
     polynomial (Padé-style) in 4-5 MVE ops per 16 elements.
  4. Sum reduction with `vaddvaq_s32`.
  5. Reciprocal of sum, multiply each element, narrow-saturating
     store at output_zp = -128 (CMSIS-NN convention, already
     encoded in `CMSIS_SOFTMAX_ZERO_POINT` in the cortex_m op).
- **Reusable softmax helper for FFN/output?** KWT-1's
  classification head is `linear → ?`. The probe has no softmax
  at the output (the head is just a Linear over the CLS token).
  So softmax appears **only inside attention** for KWT-1, and the
  helper exists in service of attention. But making it
  shape-agnostic (1-D row softmax with arbitrary length) means
  it'd serve any future model with a logits softmax.

### 4B — Tests
- Standalone unit test directly on the softmax kernel (raw int8
  rows, no model).
- Integration test: the `TwoLinearBmmAttention` model from
  Phase 3 now passes through standalone softmax instead of
  falling back to the python ref.

**Checkpoint**: attention head with standalone MVE softmax is
bit-exact vs. CMSIS-NN softmax on FVP.

---

## Phase 5 — Fused attention (memory-first attention block)

The (S×S) score tensor doesn't need to materialize. Fuse QK^T +
softmax + AV into a single op that streams.

### 5A — AOT fusion pass
- New `backends/cortex_m/passes/attention_fusion_pass.py`.
  Pattern: `bmm(q, k_T) → mul(_, scale) → softmax → bmm(_, v)`.
  Emit `cortex_m::quantized_fused_attention` taking q, k_T, v,
  scale-as-int (folded into the bmm output requantize), and all
  quant params.
- **Memory accounting**: at KWT-1 d=64, S=99 the saving is
  `99×99 = 9.8 KB` per layer × 12 layers — if these all co-live
  in the arena worst case, that's 117 KB. The exir greedy planner
  won't keep them all live at once, but each block has at least
  one score matrix live; the post-fusion arena drops the score
  from peak by ~10 KB.

### 5B — Standalone runtime: fused attention kernel
- One C function
  `attention_fused_s8(q, k_T, v, out, params)`. Inner loop: for
  each of S query rows, compute that row's S scores (S × d MACs,
  using the QK^T MVE inner from Phase 3), apply softmax to that
  row in-place to a small scratch (`int8_t scratch[MAX_S]` —
  99 B in DTCM), then compute that row of AV using the scratch as
  the contraction operand. Output is S×d streamed directly to the
  arena slot.
- Scratch size: `MAX_S` = 99 for KWT, conformer stretch will push
  this up.

### 5C — cortex_m backend: skip fused attention for now
- The cortex_m runtime composes the three ops via the executorch
  runtime — already works after Phase 4. The fused op only ships
  on standalone. (This matches MV2 — fusion ops live on
  standalone; cortex_m path uses unfused ops via CMSIS-NN
  dispatch.)

**Checkpoint**: end-to-end attention with score matrix never
materialized in arena.

---

## Phase 6 — Full one-encoder-block KWT end-to-end

Compose: LN → attention (fused) → residual → LN → FFN(Linear →
GELU → Linear) → residual.

### 6A — Encoder block AOT fusion
- Optional second fusion: `linear → gelu → linear` (FFN body)
  into `quantized_ffn`. The d_ff=256 intermediate is 99×256 =
  25 KB per block — non-trivial. Streaming FFN means
  materializing only a row of d_ff at a time → 256 B scratch
  instead of 25 KB. Lands as
  `backends/cortex_m/passes/ffn_fusion_pass.py` + standalone
  kernel `ffn_fused_s8`.
- Block-level fusion (LN + attention + residual + LN + FFN +
  residual) is **deferred to Phase 7** if needed.

### 6B — Pretrained weights & accuracy validation
- Train (or load) a real KWT-1 checkpoint on Speech Commands v2.
  Top-1 target: ~95% on the standard 35-class eval split (the
  paper claims 97.7%; lower is fine — bit-exactness is the gate,
  accuracy is the sanity check).
- Validate: int8-quantized python forward = standalone host
  scalar = standalone FVP MVE, bit-exact on a 50-sample fixture;
  top-1 within 1.5% of the FP32 reference.

### 6C — cortex_m runtime parity
- At this point everything for the cortex_m runtime path also
  runs: ExecuTorch dispatches LN (new cortex_m op), GELU (new),
  softmax (existing CMSIS-NN), BMM (existing CMSIS-NN), Linear
  (existing CMSIS-NN). Validate top-1 matches standalone.

**Checkpoint**: full KWT-1 runs end-to-end, both runtimes,
bit-exact. This is the analog of MV2's Phase A.

---

## Phase 7 — Kernel tuning pass (MV2 steps 1-20 analog)

Memory-first means measure memory first; latency tuning comes
after the memory floor is reached. So Phase 7 is:

### 7A — Memory pareto sweep
- Sweep `(d_model ∈ {32, 64, 128}, depth ∈ {6, 8, 12})` ×
  `(seq_len ∈ {49, 99})` — the standard KWT-S / KWT-M / KWT-L
  axes plus a half-sequence-length variant for sub-49-frame
  inference budgets. 18 configs.
- Report arena, scratch, cycles per the MV2 BENCHMARK.md format.

### 7B — Latency tuning (only if needed)
- Identify the hot kernels via FVP cycle breakdown. Expected
  ranking from a-priori arithmetic:
  1. BMM (~40% — d=64 × S=99 × S=99 = 625K MACs × 2 (QK + AV) ×
     12 layers = 15 M MACs, plus FFN linears at d=64, d_ff=256:
     99 × 64 × 256 × 2 × 12 = 39 M MACs → FFN dominates).
  2. FFN linears (~50%).
  3. LN + GELU + softmax (~7% combined).
  4. Patch embed + head + class-token machinery (~3%).
- The FFN linears are exactly the same primitive as MV2's 1×1
  conv inner kernel (matmul, no spatial). The existing MV2
  step-11+ `wlstp.8` 2-pixel × 4-OC inner asm transfers directly.
  **Reuse, don't rewrite.**
- BMM kernels are also matmul-shape; same inner loop, different
  lhs/rhs strides. The contraction extents (d=64, S=99) are
  friendly to MVE.
- Tune iteratively, MV2-style: each commit changes one thing,
  measures, documents in `docs/BENCHMARK.md`.

**Checkpoint**: KWT-1 latency at acceptable cycles/inference
(target: under 50 M PMU at 1.0/full-seq).

---

## Phase 8 — Conformer stretch

Conformer = (FFN/2 → MHSA → Conv-module → FFN/2 → LN) blocks. The
new pieces beyond the KWT transformer kernel set:
- **Multi-head attention with nhead>1**: forces a reshape and
  per-head BMM. The standalone BMM already handles `(B, S, d)`
  3-D shapes; per-head means batching as `(nhead, S, d_head)`.
  Sequencing: extend the BMM extractor + emitter to accept
  `(B, H, S, D)` 4-D shapes (or fold heads into batch).
- **Conv module**: 1-D pointwise conv → 1-D depthwise conv →
  BatchNorm → activation → 1-D pointwise conv. **All composable
  from the existing MV2 CNN kernel set** treated as `out_w=1` or
  `out_h=1` degenerate 2-D convs. The MV2 conv2d_s8 and
  dwconv2d_s8 kernels already handle arbitrary spatial dims;
  pass `out_h=1, kernel_h=1` for 1-D conv. Marginal kernel work
  expected: maybe a fast path for stride-1 1-D depthwise to
  avoid the `out_h=1` outer-loop overhead.
- **GLU (in some conformer variants)**: `split → sigmoid(half) *
  other half`. Sigmoid is a sibling of GELU — same LUT shape.
  Add as needed.
- **Sequence lengths**: real conformers (Conformer-S) hit
  S=400-1000. The streaming attention kernel from Phase 5 is the
  load-bearing piece. Validate the streaming kernel on S=1000
  with the 1 MB score matrix never materialized.

Gap: transformer kernel set complete + 1-D conv overlay + GLU +
multi-head BMM reshape = conformer-runnable. Expected effort ~2-3
phases on top of Phase 7.

---

## Quantization story (consolidated)

The cortex_m PT2E flow already covers Linear, BMM, Softmax, Add,
Conv, AvgPool. Missing: LayerNorm pattern, GELU pattern,
attention-specific BMM constraints.

Three concrete `CortexMQuantizer` extensions needed:
1. **`LayerNormPattern`** (`quantization_configs.py` +
   `quantizer.py`): matches `aten.native_layer_norm.default`.
   Pins input + output observers as int8 per-tensor. LN's γ, β
   stay float32 (baked into `.rodata`, not quantized).
2. **`GELUPattern`**: matches `aten.gelu.default` (any approx).
   Stateless activation observer pattern.
3. **`AttentionBmmPattern`**: the existing BMM pattern is
   shape-agnostic, but the QK^T output scale should be tied to
   softmax's expected input range to avoid clipping after
   softmax's log-domain math. Add a pattern variant that, when
   the BMM output feeds softmax, propagates a wider output range.

**LayerNorm precision decision**: dequant → float32 LN →
requant. Rationale: int8 LayerNorm is well-known to be the most
numerics-fragile op in TinyML transformers; the rounding
interactions with subsequent quantize-add residual paths cause
cascading drift. Float-internal LN costs ~6×D additional ops per
row (4K MVE ops total across all 12 layers for KWT-1, under 1%
of inference cycles). Bit-exactness becomes trivial: the C float
math matches the python `aten.native_layer_norm` to machine
precision; the requantize step is the only int8-boundary.

---

## Memory-first kernel design notes

- **Attention scores**: never materialized (Phase 5 fused
  attention streams them per row).
- **FFN intermediate**: never materialized (Phase 6 fused FFN
  streams per row).
- **Patch embed output (B, S=99, D=64) = 6.3 KB**: stays in
  arena — small enough.
- **CLS token concat**: today `torch.cat([cls, x], dim=1)`
  produces (B, 100, 64). Treat the concat as an arena layout
  choice: pre-allocate the (100, 64) slot and write CLS to
  slot[0] (a memcpy from rodata) and patch embed output to
  slot[1:]. Saves the materialized concat copy.
- **Pos embed add**: pointwise int8 add fuses into the patch
  embed Linear via an AOT pass (`PatchEmbedPosFusionPass`).
  Trivial.
- **Worst-case live set (no fusion)** for KWT-1: roughly 2 × 6.3
  KB (residual) + 9.8 KB (scores) + 25 KB (FFN intermediate) =
  ~47 KB peak per block. After Phase 5 + Phase 6 streaming:
  2 × 6.3 KB = ~13 KB peak. The headline arena should land in
  the 20-40 KB range — comfortably in any Cortex-M55 SRAM tier.

---

## Validation strategy (consolidated)

Three identical-output gates, every phase:
1. **Python int8 reference**: torch quantized forward through
   registered `cortex_m::*` python impls. Pickle int8 input +
   int8 output as the fixture.
2. **Standalone host scalar (x86)**: same C kernels,
   `-DKWT_HOST=1`, run against the fixture in CTest.
   `max int8 diff == 0` required.
3. **Standalone FVP MVE (Cortex-M55)**: same C kernels, MVE
   codepaths active. `max int8 diff == 0` required.

Plus, the cortex_m backend runtime path runs the same `.pte`
through ExecuTorch on FVP — its output must also match the
fixture bit-for-bit.

PMU cycle counter (`ARM_PMU_Get_CCNTR`) used for all cycle
measurements (DWT is broken on this FVP, per the BENCHMARK.md
calibration finding).

---

## Standalone vs cortex_m parity

Mirroring MV2: **both paths ship, but standalone is the
optimization frontier**. Per-phase split:

| Phase | cortex_m backend | Standalone |
|---|---|---|
| 0 Scaffold | — | Build harness |
| 1 LN | Scalar C++ kernel + AOT pass + quantizer pattern | MVE C kernel |
| 2 GELU | Scalar LUT C++ kernel + AOT pass + quantizer pattern | MVE LUT gather kernel |
| 3 BMM | Already exists | New MVE BMM kernel |
| 4 Softmax | Already exists | New MVE softmax kernel |
| 5 Fused attention | — (composed at runtime) | Fused kernel |
| 6 End-to-end | Validate dispatch works | Validate matches |
| 7 Tuning | — | All the latency wins live here |

The cortex_m path is the "reference, slow but always-works"
path; the standalone path is "memory-tight, bit-exact,
hand-tuned".

---

## Risks and unknowns

1. **PyTorch `nn.TransformerEncoderLayer` decomposition
   surprises**. The Phase 0 probe shows the lowering completes;
   what's *not* yet verified is whether every PT2E export run
   produces the same op decomposition across torch versions.
   KWT-1's `norm_first=True` plus the fused multi-head attention
   path can dispatch to different aten ops depending on torch
   internals (`aten.scaled_dot_product_attention.default` vs
   decomposed). Mitigation: pin torch version in the project's
   `requirements.txt`, capture the exact aten graph in
   `docs/EXPORT_GRAPH.md`, and add an export-time assert that
   fails fast if the graph shape drifts.
2. **Quantizer pattern matching ordering**. `prepare_pt2e` walks
   patterns in a defined order; LayerNorm + Add residual is a
   notorious interaction (the residual's observer needs to feed
   the LN's input observer). Pattern conflicts between
   `LayerNormPattern`, `quantized_op_fusion_pass`, and the
   residual Add pattern could leave observers in the wrong
   spots. Mitigation: write quantizer pattern tests *first* on
   tiny models, before any C code.
3. **Softmax numerics drift between CMSIS-NN s8 softmax and
   MVE-vectorized fixed-point exp**. CMSIS-NN's
   `arm_softmax_s8` uses a specific fixed-point exp polynomial;
   the standalone MVE softmax has to match it bit-for-bit,
   otherwise the cortex_m path and standalone path produce
   different outputs. Mitigation: write the standalone softmax
   as a faithful MVE-vectorized port of `arm_softmax_s8`'s
   scalar code, not as a from-scratch implementation. Diff-test
   row-by-row against CMSIS-NN.
4. **GELU LUT scale dependence**. Each GELU op needs a LUT
   specialized to its input/output (zp, scale) tuple. If the
   AOT-determined scales differ across the 12 FFN blocks, we
   get 12 LUTs (~3 KB rodata, fine) — but if they happen to be
   identical (because the post-LN scale is fixed) we get 1 LUT.
   Mitigation: at AOT emit time, hash the LUT contents and
   dedupe. Plumbing only, no risk to correctness.
5. **`arm_batch_matmul_s8` shape compatibility**. CMSIS-NN's BMM
   expects specific dim ordering and may have undocumented shape
   limits. The existing op wrap is generic-looking but only
   tested on the shapes used by other backends. The (1, 99, 99)
   × (1, 99, 64) shape of AV is uncommon (square LHS); validate
   before assuming it's correct. Mitigation: extend
   `backends/cortex_m/test/ops/test_batch_matmul.py` to cover
   the exact KWT-1 shapes in Phase 0.
6. **MFCC frontend bit-exactness across implementations**.
   Different FFT/Mel-filterbank implementations produce
   different MFCCs at the LSB. If the model graph starts at
   post-MFCC features, training and inference must agree on
   MFCC config to within int8 quantization noise. Mitigation:
   ship one canonical MFCC config (window, hop, n_mels, log
   floor) in `tools/mfcc_config.py`, used by both training and
   the runner.
7. **Conformer reach**. KWT-1's nhead=1 hides the multi-head
   reshape problem. Conformer needs nhead>1, which means the
   BMM kernel must accept `(B*H, S, d_head)` shapes — possible,
   but the per-head softmax must also batch correctly. The
   conformer stretch is a separate ~2-week effort *after* KWT
   is solid, not bundled in.
8. **Training data availability**. Speech Commands v2 is freely
   downloadable (Google) and the upstream KWT-1 repo has a
   reference recipe. The risk is the recipe drifting from
   current torchaudio. Mitigation: fall back to a pretrained
   Arm-software checkpoint if available; otherwise budget half
   a day for the training run.
9. **Standalone MVE softmax accuracy at long sequences**.
   Conformer's S=1000 sequences mean the softmax sum is the sum
   of 1000 fixed-point exp results; saturation in the sum
   accumulator is plausible. Mitigation: use int32 accumulator
   with a max-subtract pre-pass (standard); document the max
   safe sequence length.
10. **AOT planner not picking up streaming buffers correctly**.
    The existing exir greedy planner allocates output tensors
    per-op; the fused attention op produces a streamed output
    without ever needing the (S, S) score tensor in the
    planner's view. This is exactly the same shape of problem
    that MV2's Phase A/B/F/G solved — copy the technique.

---

## Critical Files for Implementation

- `/home/rja/executorch/backends/cortex_m/ops/operators.py` —
  declare new ops (`quantized_layer_norm`, `quantized_gelu`,
  `quantized_fused_attention`, `quantized_ffn`), python `impl`s
  as bit-exact references, fake meta kernels.
- `/home/rja/executorch/backends/cortex_m/passes/convert_to_cortex_m_pass.py`
  — add `_get_layer_norm_replacement`, `_get_gelu_replacement`,
  and the attention/FFN fusion pattern dispatchers; this is the
  single file that decides what enters the cortex_m op namespace.
- `/home/rja/executorch/backends/cortex_m/quantizer/quantization_configs.py`
  — new `LayerNormPattern`, `GELUPattern`, and
  `AttentionBmmPattern` entries; pins quant specs for every
  transformer op so PT2E observers wire up correctly.
- `/home/rja/executorch/examples/models/kwt_1_cortex_m_mve/src/kwt_1_kernels.c`
  (new) — all hand-written standalone MVE kernels (LayerNorm,
  GELU LUT, BMM, softmax, fused attention, fused FFN). Analog of
  `mv2_inference.c`.
- `/home/rja/executorch/examples/models/kwt_1_cortex_m_mve/tools/extractors.py`
  and `emitters.py` (new) — extract Phase-1+ ops from the
  exported graph and emit the `kwt_1_layer_params.h` / weight
  rodata blobs that the C kernels read. Analog of the MV2 tools.
