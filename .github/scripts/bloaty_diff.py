#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bloaty binary-size reports for CI."""

import argparse
import csv
import io
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from github_utils import (  # noqa: E402
    gh_fetch_json_list,
    gh_fetch_url,
    gh_post_pr_comment,
    GITHUB_API_URL,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BLOATY_CONFIG = REPO_ROOT / "test" / "bloaty" / "executorch.bloaty"
BLOATY_CMD = shlex.split(os.environ.get("BLOATY", "bloaty"))

# Buckets considered "ExecuTorch source code" for the summary table. Everything
# else (stdlib, libc, startup, metadata, other) is shown separately.
EXECUTORCH_SOURCE_BUCKETS = [
    "runtime",
    "extension",
    "backends",
    "kernels",
    "cmsis_nn",
    "tokenizers",
    "flatbuffer",
]


def _run(cmd: List[str]) -> str:
    """Run a subprocess; on failure include stderr in the exception."""
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"command failed (exit {e.returncode}): {' '.join(cmd)}\n"
            f"stderr:\n{e.stderr}"
        ) from e


def run_bloaty(elf: Path, data_sources: str) -> List[Dict[str, object]]:
    # -n 0 defeats bloaty's default 20-row truncation. -s vm sorts by VM size
    # (bytes claimed in flash + RAM after load), which is what matters for
    # embedded targets — .bss claims RAM at runtime but has filesize 0.
    cmd = [
        *BLOATY_CMD,
        "-c",
        str(BLOATY_CONFIG),
        "-d",
        data_sources,
        "-n",
        "0",
        "--csv",
        "-s",
        "vm",
        str(elf),
    ]
    out = _run(cmd)
    reader = csv.DictReader(io.StringIO(out))
    rows: List[Dict[str, object]] = []
    for row in reader:
        parsed: Dict[str, object] = {}
        for k in reader.fieldnames or []:
            if k in ("vmsize", "filesize"):
                parsed[k] = int(row[k])
            else:
                parsed[k] = row[k]
        rows.append(parsed)
    return rows


def bloaty_text(
    elf: Path,
    data_sources: str,
    top_n: int,
    source_filter: Optional[str] = None,
) -> str:
    cmd = [
        *BLOATY_CMD,
        "-c",
        str(BLOATY_CONFIG),
        "-d",
        data_sources,
        "-n",
        str(top_n),
        "-s",
        "vm",
    ]
    if source_filter is not None:
        cmd += ["--source-filter", source_filter]
    cmd.append(str(elf))
    return _run(cmd)


def strip_copy(elf: Path, strip_tool: str) -> Path:
    stripped = elf.with_suffix(elf.suffix + ".stripped")
    _run([strip_tool, "-o", str(stripped), str(elf)])
    return stripped


@dataclass
class BinaryReport:
    job: str
    binary_name: str
    head_sha: str
    stripped_head: int
    segments_head: List[Dict[str, object]] = field(default_factory=list)
    sections_head: List[Dict[str, object]] = field(default_factory=list)
    groups_head: List[Dict[str, object]] = field(default_factory=list)
    groups_head_stripped: List[Dict[str, object]] = field(default_factory=list)
    symbols_head: List[Dict[str, object]] = field(default_factory=list)
    # Base (merge-base) measurements — populated only when --base was supplied
    # to `measure`. Used by `post` to compute per-bucket deltas.
    base_sha: Optional[str] = None
    base_build_failed: bool = False
    stripped_base: Optional[int] = None
    segments_base: List[Dict[str, object]] = field(default_factory=list)
    sections_base: List[Dict[str, object]] = field(default_factory=list)
    groups_base: List[Dict[str, object]] = field(default_factory=list)
    groups_base_stripped: List[Dict[str, object]] = field(default_factory=list)
    symbols_base: List[Dict[str, object]] = field(default_factory=list)


def atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)


def render_table(rows: List[Dict[str, object]], key: str) -> str:
    if not rows:
        return "_(no data)_"
    out = ["| {} | vmsize | filesize |".format(key), "|---|---:|---:|"]
    for r in sorted(rows, key=lambda x: -int(x["vmsize"])):
        if r[key] == "TOTAL":
            continue
        out.append(f"| `{r[key]}` | {r['vmsize']:,} | {r['filesize']:,} |")
    return "\n".join(out)


def render_step_summary(
    report: BinaryReport, full_text: str, head_only_text: str
) -> str:
    et_rows = [
        r
        for r in report.groups_head
        if r.get("executorch") in EXECUTORCH_SOURCE_BUCKETS
    ]
    et_total = sum(int(r["vmsize"]) for r in et_rows)
    lines = [
        f"## Bloaty: `{report.job}` / `{report.binary_name}`",
        "",
        f"- head sha: `{report.head_sha}`",
        f"- stripped head vm size: **{report.stripped_head:,} bytes**",
        f"- ExecuTorch source total (unstripped, bucketed, vm): **{et_total:,} bytes**",
        "",
        "### Per-bucket sizes (unstripped, all buckets)",
        "",
        render_table(report.groups_head, "executorch"),
        "",
        "<details><summary>Full bloaty output</summary>",
        "",
        "```",
        full_text.rstrip(),
        "```",
        "",
        "</details>",
        "",
        "<details><summary>Top ExecuTorch source symbols</summary>",
        "",
        "```",
        head_only_text.rstrip(),
        "```",
        "",
        "</details>",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Post: sticky PR comment with per-bucket deltas
# ---------------------------------------------------------------------------

STICKY_MARKER = "<!-- executorch-ci-comment kind=bloaty-binary-size -->"

# Only consider these author logins as candidates for editing — defends against
# a malicious user comment that includes the sticky marker.
ALLOWED_COMMENT_AUTHORS = {
    "github-actions[bot]",
    "pytorch-bot[bot]",
    "facebook-github-bot",
}

# Hide rows whose absolute vmsize delta is below this.
NOISE_FLOOR_BYTES = 16

# Cap per-table row count so an enormous Δ (e.g. a refactor touching 1000s of
# symbols) doesn't blow through the 60 KB body limit before we get to other
# tables.
DELTA_TABLE_ROW_CAP = 50

# GitHub comment body hard limit is 65,536; leave headroom for trailing
# truncation markers we may add.
MAX_COMMENT_BODY = 60_000


def _md_cell(s: object) -> str:
    """Escape backticks, pipes, and newlines in a markdown table cell. Bloaty
    can return C++ demangled symbols containing all three; without escaping,
    these break the code-span (backtick) or table column (pipe) and could let
    arbitrary markdown through."""
    out = str(s)
    out = out.replace("`", r"\`").replace("|", r"\|")
    out = out.replace("\r", "").replace("\n", " ")
    return out


def _delta_table(
    head_rows: List[Dict[str, object]],
    base_rows: List[Dict[str, object]],
    key: str,
    floor: int = NOISE_FLOOR_BYTES,
    row_cap: int = DELTA_TABLE_ROW_CAP,
) -> Tuple[str, int]:
    """Render a markdown table of per-row Δ vmsize. Returns (table, max_abs_delta)."""
    head_by_key = {
        str(r.get(key)): int(r["vmsize"])
        for r in head_rows
        if r.get(key) and r.get(key) != "TOTAL"
    }
    base_by_key = {
        str(r.get(key)): int(r["vmsize"])
        for r in base_rows
        if r.get(key) and r.get(key) != "TOTAL"
    }
    all_keys = set(head_by_key) | set(base_by_key)
    deltas = []
    for k in all_keys:
        h = head_by_key.get(k, 0)
        b = base_by_key.get(k, 0)
        d = h - b
        if abs(d) >= floor:
            deltas.append((k, h, b, d))
    if not deltas:
        return ("_(no rows above noise floor)_", 0)
    deltas.sort(key=lambda x: -abs(x[3]))
    total = len(deltas)
    truncated = total > row_cap
    deltas = deltas[:row_cap]
    out = [
        f"| {key} | head (vm) | base (vm) | Δ (vm) |",
        "|---|---:|---:|---:|",
    ]
    for k, h, b, d in deltas:
        sign = "+" if d > 0 else ""
        out.append(f"| `{_md_cell(k)}` | {h:,} | {b:,} | {sign}{d:,} |")
    if truncated:
        out.append(f"| _…{total - row_cap} more rows omitted_ |  |  |  |")
    return "\n".join(out), max(abs(d) for _, _, _, d in deltas)


def _render_comment(reports: List[BinaryReport]) -> Tuple[str, bool]:
    """Render the sticky PR comment body. Returns (body, any_change_above_noise)."""
    lines = [STICKY_MARKER, "", "## Binary size report (bloaty)", ""]

    lines += [
        "| job / binary | head (vm) | base (vm) | Δ stripped (vm) | note |",
        "|---|---:|---:|---:|---|",
    ]
    any_change = False
    for r in reports:
        note = ""
        if r.base_build_failed:
            note = "base build failed; head-only"
            base_str = "—"
            delta_str = "—"
        elif r.stripped_base is None:
            note = "no base measurement"
            base_str = "—"
            delta_str = "—"
        else:
            delta = r.stripped_head - r.stripped_base
            base_str = f"{r.stripped_base:,}"
            sign = "+" if delta > 0 else ""
            delta_str = f"{sign}{delta:,}"
            if abs(delta) >= NOISE_FLOOR_BYTES:
                any_change = True
        lines.append(
            f"| `{_md_cell(r.job)} / {_md_cell(r.binary_name)}` "
            f"| {r.stripped_head:,} | {base_str} | "
            f"{delta_str} | {note} |"
        )

    for r in reports:
        lines += [
            "",
            f"<details><summary><code>{_md_cell(r.job)} / "
            f"{_md_cell(r.binary_name)}</code> details</summary>",
            "",
        ]
        if r.base_build_failed:
            lines += ["_Base build failed — showing head-only data._", ""]
            lines += ["**Per-bucket (stripped, head only):**", ""]
            lines += [
                render_table(
                    [
                        x
                        for x in r.groups_head_stripped
                        if x.get("executorch") in EXECUTORCH_SOURCE_BUCKETS
                    ],
                    "executorch",
                )
            ]
        elif r.stripped_base is None:
            lines += ["_No base measurement provided._", ""]
        else:
            sym_tbl, sym_max = _delta_table(
                r.symbols_head, r.symbols_base, "shortsymbols"
            )
            lines += ["**Top symbol Δ:**", "", sym_tbl, ""]
            grp_tbl, grp_max = _delta_table(
                r.groups_head_stripped, r.groups_base_stripped, "executorch"
            )
            lines += ["**Per-bucket Δ (stripped):**", "", grp_tbl, ""]
            sec_tbl, _ = _delta_table(r.sections_head, r.sections_base, "sections")
            lines += ["**By section Δ:**", "", sec_tbl, ""]
            seg_tbl, _ = _delta_table(r.segments_head, r.segments_base, "segments")
            lines += ["**By segment Δ:**", "", seg_tbl]
            if max(sym_max, grp_max) >= NOISE_FLOOR_BYTES:
                any_change = True
        lines.append("</details>")

    body = "\n".join(lines)
    if len(body) > MAX_COMMENT_BODY:
        cut = body.rfind("\n\n", 0, MAX_COMMENT_BODY)
        body = body[: cut if cut > 0 else MAX_COMMENT_BODY]
        # Close any open <details> the truncation may have cut into, so the
        # rest of the comment doesn't render as a collapsed-broken block.
        body += "\n\n</details>\n\n_…truncated to fit GitHub's 65 KB comment limit._\n"
    return body, any_change


def _render_noise_floor_comment() -> str:
    return (
        f"{STICKY_MARKER}\n\n"
        "## Binary size report (bloaty)\n\n"
        f"All measured binaries within ±{NOISE_FLOOR_BYTES} bytes of base — "
        "no notable change.\n"
    )


def _find_sticky_comment(org: str, repo: str, pr_num: int) -> Optional[int]:
    """Return the comment ID of the existing sticky comment, or None.

    Paginates through all comments — long PRs routinely exceed 100, and the
    sticky must remain findable across the full history.
    """
    page = 1
    while True:
        url = (
            f"{GITHUB_API_URL}/repos/{org}/{repo}/issues/{pr_num}/comments"
            f"?per_page=100&page={page}"
        )
        comments = gh_fetch_json_list(url)
        if not comments:
            return None
        for c in comments:
            body = c.get("body") or ""
            author = (c.get("user") or {}).get("login", "")
            if STICKY_MARKER in body and author in ALLOWED_COMMENT_AUTHORS:
                return int(c["id"])
        if len(comments) < 100:
            return None
        page += 1


def _patch_comment(org: str, repo: str, comment_id: int, body: str) -> None:
    url = f"{GITHUB_API_URL}/repos/{org}/{repo}/issues/comments/{comment_id}"
    gh_fetch_url(url, data={"body": body}, method="PATCH", reader=lambda x: x.read())


def cmd_post(args: argparse.Namespace) -> int:
    in_dir = Path(args.in_dir).resolve()
    if not in_dir.is_dir():
        print(f"--in-dir does not exist: {in_dir}", file=sys.stderr)
        return 1

    reports: List[BinaryReport] = []
    pr_num: Optional[int] = args.pr_num
    for meta in sorted(in_dir.rglob("metadata.json")):
        with open(meta) as f:
            data = json.load(f)
        # Drop unknown fields so older artifacts don't blow up if schema grows
        known = {f.name for f in BinaryReport.__dataclass_fields__.values()}
        reports.append(BinaryReport(**{k: v for k, v in data.items() if k in known}))
        if pr_num is None:
            pr_file = meta.parent / "pr_number.txt"
            if pr_file.exists():
                txt = pr_file.read_text().strip()
                if txt.isdigit():
                    pr_num = int(txt)

    if not reports:
        print(f"no metadata.json found under {in_dir}", file=sys.stderr)
        return 1

    body, any_change = _render_comment(reports)
    # Use the friendly "within noise floor" body only when we actually have base
    # measurements to compare against. With no base data (e.g. push event), the
    # absence of changes is meaningless, not noise.
    have_base = any(r.stripped_base is not None for r in reports)
    if have_base and not any_change and not any(r.base_build_failed for r in reports):
        body = _render_noise_floor_comment()

    if args.dry_run:
        print(body)
        return 0

    if pr_num is None:
        print(
            "no PR number found (pass --pr-num or include pr_number.txt in an artifact)",
            file=sys.stderr,
        )
        return 1

    existing = _find_sticky_comment(args.org, args.repo, pr_num)
    if existing is not None:
        print(
            f"patching existing comment {existing} on {args.org}/{args.repo}#{pr_num}"
        )
        _patch_comment(args.org, args.repo, existing, body)
    else:
        print(f"posting new comment on {args.org}/{args.repo}#{pr_num}")
        gh_post_pr_comment(args.org, args.repo, pr_num, body)
    return 0


# ---------------------------------------------------------------------------
# Measure
# ---------------------------------------------------------------------------


def _measure_one(elf: Path, strip_tool: str):
    stripped = strip_copy(elf, strip_tool)
    try:
        groups_stripped = run_bloaty(stripped, "executorch")
    finally:
        stripped.unlink(missing_ok=True)
    # VM size of the stripped binary — flash + RAM bytes the loader claims.
    # .bss adds to vm but not file, so this differs from `ls -la` for any
    # binary with statically-allocated buffers.
    stripped_size = sum(
        int(r["vmsize"]) for r in groups_stripped if r.get("executorch") != "TOTAL"
    )
    segments = run_bloaty(elf, "segments")
    sections = run_bloaty(elf, "sections")
    groups = run_bloaty(elf, "executorch")
    symbols = run_bloaty(elf, "shortsymbols")
    return stripped_size, groups_stripped, segments, sections, groups, symbols


def cmd_measure(args: argparse.Namespace) -> int:
    head = Path(args.head).resolve()
    if not head.exists():
        print(f"head ELF does not exist: {head}", file=sys.stderr)
        return 1

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    (
        stripped_head,
        groups_head_stripped,
        segments_head,
        sections_head,
        groups_head,
        symbols_head,
    ) = _measure_one(head, args.strip_tool)

    report = BinaryReport(
        job=args.job,
        binary_name=args.binary_name,
        head_sha=args.head_sha,
        stripped_head=stripped_head,
        segments_head=segments_head,
        sections_head=sections_head,
        groups_head=groups_head,
        groups_head_stripped=groups_head_stripped,
        symbols_head=symbols_head,
    )

    if args.base_build_failed:
        report.base_build_failed = True
        if args.base_sha:
            report.base_sha = args.base_sha
    elif args.base:
        base = Path(args.base).resolve()
        if not base.exists():
            print(
                f"base ELF does not exist: {base}; marking base_build_failed",
                file=sys.stderr,
            )
            report.base_build_failed = True
            if args.base_sha:
                report.base_sha = args.base_sha
        else:
            (
                stripped_base,
                groups_base_stripped,
                segments_base,
                sections_base,
                groups_base,
                symbols_base,
            ) = _measure_one(base, args.strip_tool)
            report.base_sha = args.base_sha
            report.stripped_base = stripped_base
            report.segments_base = segments_base
            report.sections_base = sections_base
            report.groups_base = groups_base
            report.groups_base_stripped = groups_base_stripped
            report.symbols_base = symbols_base

    atomic_write(out_dir / "metadata.json", json.dumps(asdict(report), indent=2))

    # executorch first → groups all symbols by bucket; sections then symbols
    # show what's inside each. Skipping `segments` (uninformative at this level).
    full_text = bloaty_text(head, "executorch,sections,shortsymbols", top_n=30)
    # Filter the head-only top-symbols dump to ExecuTorch source buckets only,
    # so stdlib / libc / startup / metadata / other don't crowd it out.
    head_only_text = bloaty_text(
        head,
        "executorch,shortsymbols",
        top_n=30,
        source_filter="|".join(EXECUTORCH_SOURCE_BUCKETS),
    )
    atomic_write(out_dir / "full.txt", full_text)
    atomic_write(out_dir / "head_only.txt", head_only_text)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(render_step_summary(report, full_text, head_only_text))

    print(f"wrote {out_dir / 'metadata.json'}")
    print(f"stripped head vm size: {stripped_head:,} bytes")
    if report.stripped_base is not None:
        delta = stripped_head - report.stripped_base
        print(
            f"stripped base vm size: {report.stripped_base:,} bytes "
            f"(delta {delta:+,})"
        )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_measure = sub.add_parser("measure", help="Measure an ELF with bloaty")
    p_measure.add_argument(
        "--head", required=True, help="Path to head (unstripped) ELF"
    )
    p_measure.add_argument("--job", required=True, help="CI job identifier")
    p_measure.add_argument(
        "--binary-name", required=True, help="Binary name (e.g. size_test)"
    )
    p_measure.add_argument(
        "--head-sha", required=True, help="Git SHA of the head commit"
    )
    p_measure.add_argument(
        "--strip-tool", default="strip", help="Strip tool (e.g. arm-none-eabi-strip)"
    )
    p_measure.add_argument("--out", required=True, help="Output directory")
    p_measure.add_argument(
        "--base",
        default=None,
        help="Path to merge-base (unstripped) ELF. Optional.",
    )
    p_measure.add_argument(
        "--base-sha",
        default=None,
        help="Git SHA of the merge-base commit. Optional.",
    )
    p_measure.add_argument(
        "--base-build-failed",
        action="store_true",
        help="Set when the base build itself failed; records the failure so "
        "the PR comment can render '(base build failed, head-only)'.",
    )
    p_measure.set_defaults(func=cmd_measure)

    p_post = sub.add_parser(
        "post", help="Render & upsert the sticky PR comment from artifacts"
    )
    p_post.add_argument(
        "--in-dir",
        required=True,
        help="Directory containing per-job subdirectories with metadata.json + "
        "(optionally) pr_number.txt",
    )
    p_post.add_argument(
        "--org", default="pytorch", help="GitHub org (default: pytorch)"
    )
    p_post.add_argument(
        "--repo", default="executorch", help="GitHub repo (default: executorch)"
    )
    p_post.add_argument(
        "--pr-num",
        type=int,
        default=None,
        help="Pull request number. If omitted, read from pr_number.txt in an artifact.",
    )
    p_post.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the comment body to stdout instead of posting.",
    )
    p_post.set_defaults(func=cmd_post)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
