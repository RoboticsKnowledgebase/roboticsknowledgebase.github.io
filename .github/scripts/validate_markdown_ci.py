#!/usr/bin/env python3
"""CI markdown validation for changed markdown files."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

MD_EXTS = {".md", ".markdown"}
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
FENCE_RE = re.compile(r"^```")
INLINE_MATH_RE = re.compile(r"\$(?:\\.|[^\n$])+\$")
BLOCK_MATH_RE = re.compile(r"\$\$(?:.|\n)*?\$\$", re.MULTILINE)
VALID_MD_BASENAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*\.md$")


def _clean_target(target: str) -> str:
    return target.strip().split()[0].strip("<>").split("#", 1)[0]


def _is_external(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:", "tel:"))


def _iter_markdown_paths(root: Path, inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in inputs:
        p = (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
        if p.is_file() and p.suffix.lower() in MD_EXTS:
            out.append(p)
    return sorted(set(out))


def _validate_front_matter(lines: list[str], path: Path) -> list[str]:
    issues = []
    if lines and lines[0].strip() == "---":
        try:
            end_idx = next(i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---")
            if end_idx == 1:
                issues.append(f"{path}:1 empty front matter block")
        except StopIteration:
            issues.append(f"{path}:1 missing closing front matter delimiter '---'")
    return issues


def _validate_filename(path: Path, root: Path) -> list[str]:
    rel = path.relative_to(root)
    name = path.name
    if name in {"index.md", "__all_subsections.md"}:
        return []
    if name.lower() != name:
        return [f"{path}:1 invalid markdown filename casing: {rel}"]
    if not VALID_MD_BASENAME_RE.match(name):
        return [f"{path}:1 invalid markdown filename format (use kebab-case): {rel}"]
    return []


def _candidate_targets(base: Path) -> list[Path]:
    s = str(base)
    cands = [base]
    if not base.suffix:
        cands.append(Path(s + ".md"))
        cands.append(Path(s + ".markdown"))
    cands.append(base / "index.md")
    cands.append(base / "index.markdown")
    return cands


def _resolve_target(raw: str, path: Path, root: Path) -> Path | None:
    cleaned = unquote(raw)
    base = (root / cleaned.lstrip("/")) if cleaned.startswith("/") else (path.parent / cleaned)
    for cand in _candidate_targets(base):
        if cand.exists():
            return cand
    return None


def _validate_links(text: str, path: Path, root: Path) -> list[str]:
    issues = []
    for regex, kind in ((LINK_RE, "link"), (IMAGE_RE, "image")):
        for m in regex.finditer(text):
            raw = _clean_target(m.group(1))
            if not raw or raw.startswith("#") or _is_external(raw):
                continue
            if _resolve_target(raw, path, root) is None:
                line = text.count("\n", 0, m.start()) + 1
                issues.append(f"{path}:{line} broken {kind} target: {raw}")
    return issues


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


def _mask_math_spans(text: str) -> str:
    def _blank(match: re.Match[str]) -> str:
        return "".join("\n" if ch == "\n" else " " for ch in match.group(0))

    out = BLOCK_MATH_RE.sub(_blank, text)
    out = INLINE_MATH_RE.sub(_blank, out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate changed markdown files")
    parser.add_argument("paths", nargs="*", help="Changed file paths")
    args = parser.parse_args()

    root = Path('.').resolve()
    md_files = _iter_markdown_paths(root, args.paths)
    if not md_files:
        print("Markdown CI validation skipped (no markdown files changed).")
        return 0

    issues: list[str] = []
    failed_files: set[Path] = set()
    for path in md_files:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            issues.append(f"{path}:1 unreadable markdown: {exc}")
            failed_files.add(path)
            continue
        path_issues = []
        path_issues.extend(_validate_filename(path, root))
        path_issues.extend(_validate_front_matter(text.splitlines(), path))
        scan_text = _mask_math_spans(_mask_fenced_code_blocks(text))
        path_issues.extend(_validate_links(scan_text, path, root))
        if path_issues:
            issues.extend(path_issues)
            failed_files.add(path)

    passed = len(md_files) - len(failed_files)
    failed = len(failed_files)
    if issues:
        print("Markdown validation failed:")
        for issue in issues:
            print(f"- {issue}")
        print(f"\nSummary: {passed} file(s) passed, {failed} file(s) failed.")
        return 1

    print(f"Markdown validation passed ({len(md_files)} file(s)).")
    print(f"Summary: {passed} file(s) passed, {failed} file(s) failed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
