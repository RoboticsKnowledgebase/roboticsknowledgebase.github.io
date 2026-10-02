#!/usr/bin/env python3
"""CI image validation for changed markdown files only."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
MD_EXTS = {".md", ".markdown"}
FENCE_RE = re.compile(r"^```")


def _clean_target(target: str) -> str:
    return target.strip().split()[0].strip("<>").split("#", 1)[0]


def _iter_markdown(root: Path, inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in inputs:
        p = (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
        if p.is_file() and p.suffix.lower() in MD_EXTS:
            out.append(p)
    return sorted(set(out))


def _mask_fenced_code_blocks(text: str) -> str:
    lines = text.splitlines(keepends=True)
    masked: list[str] = []
    in_fence = False
    for line in lines:
        if FENCE_RE.match(line.strip()):
            in_fence = not in_fence
            masked.append("\n")
            continue
        masked.append("\n" if in_fence else line)
    return "".join(masked)




def _candidate_targets(base: Path) -> list[Path]:
    s = str(base)
    cands = [base]
    if not base.suffix:
        cands.append(Path(s + ".md"))
        cands.append(Path(s + ".markdown"))
    cands.append(base / "index.md")
    cands.append(base / "index.markdown")
    return cands


def _resolve_target(raw: str, md: Path, root: Path) -> Path | None:
    cleaned = unquote(raw)
    base = (root / cleaned.lstrip("/")) if cleaned.startswith("/") else (md.parent / cleaned)
    for cand in _candidate_targets(base):
        if cand.exists():
            return cand
    return None

def main() -> int:
    parser = argparse.ArgumentParser(description="Validate image references in changed markdown")
    parser.add_argument("paths", nargs="*", help="Changed file paths")
    parser.add_argument("--root", default=".", help="Repository root")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    md_files = _iter_markdown(root, args.paths)
    if not md_files:
        print("Image CI validation skipped (no markdown files changed).")
        return 0

    issues: list[str] = []
    for md in md_files:
        text = _mask_fenced_code_blocks(md.read_text(encoding="utf-8"))
        for m in IMG_RE.finditer(text):
            raw = _clean_target(m.group(1))
            if not raw or raw.startswith(("http://", "https://", "data:")):
                continue
            target = _resolve_target(raw, md, root)
            if target is None:
                line = text.count("\n", 0, m.start()) + 1
                issues.append(f"{md}:{line} missing image: {raw}")

    if issues:
        print("Image CI validation failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print(f"Image CI validation passed ({len(md_files)} markdown file(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
