# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).

"""Emit generated/ artifacts for the standalone KWT-1 build.

Phase 0 scope (this file): emit only `kwt_1_arena.h` (sizes) and
`input_fixture.h` (a deterministic 40x98 MFCC-shaped float fixture
that the Phase 0 round-trip stub kernel consumes).  No weights, no
LayerParams, no AOT lowering yet — that lands in Phase 1+ once
LayerNorm is the first real op.

The Phase 0 stub kernel reduces the 40x98 input to 35 int8 outputs
via a deterministic per-column-mod position-folded average; the
fixture + arena sizes here have to match the macros the C code
expects (KWT_1_INPUT_NUM_ELEMENTS, KWT_1_OUTPUT_NUM_ELEMENTS,
KWT_1_ARENA_BYTES, KWT_1_SCRATCH_BYTES).
"""

from __future__ import annotations

import argparse
import pathlib

# KWT-1 canonical input shape (post-MFCC): (1, 1, 40, 98)
INPUT_NUM_ELEMENTS = 40 * 98  # = 3920
OUTPUT_NUM_ELEMENTS = 35  # Speech Commands v2 class count


def emit_arena_h(out_dir: pathlib.Path) -> None:
    """Phase 0: arena holds the int8-quantized input.  Sized for the
    largest live tensor we'll need to round-trip — at Phase 0 that's
    just the input.  Phase 1+ will replace this with the AOT memory
    planner's verdict."""
    arena_bytes = INPUT_NUM_ELEMENTS  # int8 per element

    text = (
        "#ifndef KWT_1_ARENA_H_\n"
        "#define KWT_1_ARENA_H_\n"
        "\n"
        "#include <stdint.h>\n"
        "\n"
        f"#define KWT_1_INPUT_NUM_ELEMENTS {INPUT_NUM_ELEMENTS}u\n"
        f"#define KWT_1_OUTPUT_NUM_ELEMENTS {OUTPUT_NUM_ELEMENTS}u\n"
        f"#define KWT_1_ARENA_BYTES {arena_bytes}u\n"
        "#define KWT_1_SCRATCH_BYTES 0u\n"
        "\n"
        "#endif\n"
    )
    (out_dir / "kwt_1_arena.h").write_text(text)


def emit_input_fixture_h(out_dir: pathlib.Path) -> None:
    """Phase 0: deterministic fixture so the round-trip stub has a
    known output.  Uses a sinusoidal pattern across the 40x98 grid;
    nothing about it resembles real MFCC features.  Phase 1+ will
    replace this with a fixed Speech Commands v2 sample after MFCC
    preprocessing."""
    import math

    values = []
    for r in range(40):
        for c in range(98):
            v = math.sin(0.1 * r) * math.cos(0.07 * c) + 0.01 * (r - c)
            values.append(v)
    assert len(values) == INPUT_NUM_ELEMENTS

    lines = [
        "#ifndef KWT_1_INPUT_FIXTURE_H_",
        "#define KWT_1_INPUT_FIXTURE_H_",
        "",
        "#include <stdint.h>",
        "",
        f"const float kwt_1_fixture_input[{INPUT_NUM_ELEMENTS}] = {{",
    ]
    def _float_lit(v: float) -> str:
        # `:.9g` can produce bare integer like "0" or "1e-05"; force a decimal
        # so the C++ lexer accepts the 'f' suffix (e.g. "0" -> "0.0f", not "0f").
        s = f"{v:.9g}"
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s + "f"

    for i in range(0, len(values), 6):
        chunk = values[i : i + 6]
        lines.append("  " + ", ".join(_float_lit(v) for v in chunk) + ",")
    lines.append("};")
    lines.append("")
    lines.append("#endif")
    lines.append("")

    (out_dir / "input_fixture.h").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "out_dir",
        type=pathlib.Path,
        help="Output directory; will be created if absent.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    emit_arena_h(args.out_dir)
    emit_input_fixture_h(args.out_dir)
    print(f"Phase 0 artifacts written to {args.out_dir}")
    print(f"  KWT_1_INPUT_NUM_ELEMENTS  = {INPUT_NUM_ELEMENTS}")
    print(f"  KWT_1_OUTPUT_NUM_ELEMENTS = {OUTPUT_NUM_ELEMENTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
