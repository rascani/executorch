# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).
"""Load pretrained ARM-software/keyword-transformer KWT-1 weights into
this directory's PyTorch `KWT1` module.

The ARM-software repo ships an .h5 Keras checkpoint trained on Speech
Commands v2 (35 keywords).  See:
  https://github.com/ARM-software/keyword-transformer  (Apache 2.0)

Architectural caveats — the cortex_m + standalone path implements a
subset of the published KWT-1 architecture today (see ../README.md
"Architectural caveats" section).  In particular:

  * The published KWT-1 uses a learned [CLS] token + final slice; this
    port substitutes a mean-pool over the sequence dim, which the
    cortex_m backend lowers cleanly.  Encoder weights load 1:1, but
    the classification head (`head.weight`, `head.bias`) cannot be
    transferred as-is — it was trained on CLS-pooled features.  A
    practical recipe: load the encoder weights, then fine-tune just
    the head on Speech Commands v2 (cheap — single Linear).

  * Pretrained pos-encoding weights aren't loaded today either; the
    standalone path's positional-encoding code path is wired
    (AddLayer.{self,other}_const, see kwt_1_kernels.c) but disabled
    because PT2E's annotator emits a warning when an Add has a
    parameter input.  When that warning is resolved upstream this
    loader will additionally populate `model.pos`.

  * Pretrained final LayerNorm weights aren't loaded today; the
    CortexMQuantizer doesn't currently annotate a LN that lives
    between the encoder output and a float reduction (mean), so we
    skip it.

Weight key mapping table (Keras → PyTorch):

    keras name (under model.weights)                pytorch param
    --------------------------------                 -------------
    embedding/kernel:0          (40, 64)             embed.weight.T  (transposed)
    embedding/bias:0            (64,)                embed.bias
    pos_embedding/embeddings:0  (1, 99, 64)          model.pos (currently skipped — drop the CLS row to use first 98)
    transformer_block_{i}_layernorm_0/gamma:0  (64,)  encoder.blocks[i].ln1.weight
    transformer_block_{i}_layernorm_0/beta:0   (64,)  encoder.blocks[i].ln1.bias
    transformer_block_{i}_attention/query/kernel:0  (64, 64)  encoder.blocks[i].lq.weight.T
    transformer_block_{i}_attention/query/bias:0    (64,)     encoder.blocks[i].lq.bias
    transformer_block_{i}_attention/key/kernel:0    (64, 64)  encoder.blocks[i].lk.weight.T
    transformer_block_{i}_attention/key/bias:0      (64,)     encoder.blocks[i].lk.bias
    transformer_block_{i}_attention/value/kernel:0  (64, 64)  encoder.blocks[i].lv.weight.T
    transformer_block_{i}_attention/value/bias:0    (64,)     encoder.blocks[i].lv.bias
    transformer_block_{i}_attention/output/kernel:0 (64, 64)  encoder.blocks[i].lo.weight.T
    transformer_block_{i}_attention/output/bias:0   (64,)     encoder.blocks[i].lo.bias
    transformer_block_{i}_layernorm_1/gamma:0  (64,)  encoder.blocks[i].ln2.weight
    transformer_block_{i}_layernorm_1/beta:0   (64,)  encoder.blocks[i].ln2.bias
    transformer_block_{i}_ffn_0/kernel:0  (64, 128)   encoder.blocks[i].ff1.weight.T
    transformer_block_{i}_ffn_0/bias:0    (128,)      encoder.blocks[i].ff1.bias
    transformer_block_{i}_ffn_1/kernel:0  (128, 64)   encoder.blocks[i].ff2.weight.T
    transformer_block_{i}_ffn_1/bias:0    (64,)       encoder.blocks[i].ff2.bias
    layernorm_final/gamma:0  (64,)        (skipped — see caveat above)
    layernorm_final/beta:0   (64,)        (skipped)
    classifier/kernel:0  (64, 35)         head.weight.T  (NOT bit-equivalent — see caveat)
    classifier/bias:0    (35,)            head.bias

The exact key strings depend on the .h5 saver — keras's HDF5 format
nests groups under `model_weights/<layer_name>/<weight_name>`.  This
loader expects a flat {key -> tensor} dict already extracted from
the .h5; pair it with a small `h5py` reader (kept out of this file
to avoid an h5py dependency for users that don't need it).

Usage:

    from .dump_kwt_1_artifacts import KWT1
    from .load_arm_kwt1 import load_arm_kwt1, h5_to_dict

    model = KWT1()
    weight_dict = h5_to_dict("kws_model_kwt1_best.h5")
    n_loaded, n_skipped = load_arm_kwt1(model, weight_dict)
    print(f"loaded {n_loaded} tensors, skipped {n_skipped}")
"""

from __future__ import annotations

from typing import Optional

import torch


def h5_to_dict(path: str) -> dict[str, torch.Tensor]:
    """Walk a Keras .h5 file and flatten its weights into
    `{slash/separated/key: torch.Tensor}`.  Lazy import h5py so the
    rest of the dumper stays dependency-free.

    This is the only piece that needs the .h5 file present on disk;
    everything below operates on the resulting dict.
    """
    try:
        import h5py  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "load_arm_kwt1.h5_to_dict requires h5py; install with `pip install h5py`."
        ) from e
    out: dict[str, torch.Tensor] = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = torch.from_numpy(obj[...])
        f.visititems(visit)
    return out


def _find_key(weights: dict, *substrings: str) -> Optional[str]:
    """Return the first weight key that contains every substring, or
    None if no key matches.  Keras key paths vary by save mode
    (`model_weights/transformer_block_0/...` vs flat); the substring
    match lets us be permissive."""
    for key in weights:
        if all(s in key for s in substrings):
            return key
    return None


def _load_linear(
    layer, weights: dict, kernel_substrings, bias_substrings,
    transpose: bool = True,
) -> bool:
    """Find the kernel + bias and assign them to `layer` (a nn.Linear).
    Keras Dense layers store weights as (in_features, out_features);
    PyTorch Linear stores them as (out_features, in_features) — so
    we transpose by default."""
    kkey = _find_key(weights, *kernel_substrings)
    bkey = _find_key(weights, *bias_substrings)
    if kkey is None or bkey is None:
        return False
    k = weights[kkey]
    b = weights[bkey]
    if transpose:
        k = k.t().contiguous()
    with torch.no_grad():
        layer.weight.copy_(k.to(layer.weight.dtype))
        layer.bias.copy_(b.to(layer.bias.dtype))
    return True


def _load_layernorm(layer, weights: dict, gamma_substrings, beta_substrings) -> bool:
    gk = _find_key(weights, *gamma_substrings)
    bk = _find_key(weights, *beta_substrings)
    if gk is None or bk is None:
        return False
    with torch.no_grad():
        layer.weight.copy_(weights[gk].to(layer.weight.dtype))
        layer.bias.copy_(weights[bk].to(layer.bias.dtype))
    return True


def load_arm_kwt1(model, weights: dict[str, torch.Tensor]) -> tuple[int, int]:
    """Populate as many of `model`'s parameters as possible from an
    ARM-software-format weight dict.  Returns (n_loaded, n_skipped).

    Skipped: any param that the architectural caveats above flag as
    not transferrable (pos-encoding, final LN, head when present).
    The caller should pair this with a small fine-tune on the head to
    recover most of the published accuracy."""
    n_loaded = 0
    n_skipped = 0

    # Input embedding.
    if _load_linear(
        model.embed, weights,
        kernel_substrings=("embedding", "kernel"),
        bias_substrings=("embedding", "bias"),
    ):
        n_loaded += 2
    else:
        n_skipped += 2

    # Per-block encoder weights.
    for i, blk in enumerate(model.encoder.blocks):
        tag = f"transformer_block_{i}"

        for ln_attr, ln_idx in (("ln1", "layernorm_0"), ("ln2", "layernorm_1")):
            if _load_layernorm(
                getattr(blk, ln_attr), weights,
                gamma_substrings=(tag, ln_idx, "gamma"),
                beta_substrings=(tag, ln_idx, "beta"),
            ):
                n_loaded += 2
            else:
                n_skipped += 2

        for lin_attr, attn_name in (
            ("lq", "query"), ("lk", "key"), ("lv", "value"), ("lo", "output"),
        ):
            if _load_linear(
                getattr(blk, lin_attr), weights,
                kernel_substrings=(tag, "attention", attn_name, "kernel"),
                bias_substrings=(tag, "attention", attn_name, "bias"),
            ):
                n_loaded += 2
            else:
                n_skipped += 2

        for ff_attr, ff_idx in (("ff1", "ffn_0"), ("ff2", "ffn_1")):
            if _load_linear(
                getattr(blk, ff_attr), weights,
                kernel_substrings=(tag, ff_idx, "kernel"),
                bias_substrings=(tag, ff_idx, "bias"),
            ):
                n_loaded += 2
            else:
                n_skipped += 2

    # Skipped tensors per the caveats — counted for transparency.
    if model.pos is not None:
        n_skipped += 1
    if model.final_ln is not None:
        n_skipped += 2
    # Head intentionally not loaded (trained against CLS-pooled features).
    n_skipped += 2

    return n_loaded, n_skipped
