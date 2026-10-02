#!/usr/bin/env python3
"""CI orchestration for incoming-change validation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CI_SCRIPTS = ROOT / ".github" / "scripts"
ALLOWED_PREFIXES = ("wiki/", "docs/", "assets/")


def _git_changed(base: str, head: str) -> list[str]:
    proc = subprocess.run(["git", "diff", "--name-only", f"{base}...{head}"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd, cwd=ROOT).returncode


def _in_scope(path: str) -> bool:
    return path.startswith(ALLOWED_PREFIXES)


def _all_files_in_scope() -> list[str]:
    files: list[str] = []
    for prefix in ALLOWED_PREFIXES:
        base = ROOT / prefix.rstrip("/")
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                files.append(str(path.relative_to(ROOT)))
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate changed files in CI")
    parser.add_argument("--base", default="origin/master")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.all:
        changed = _all_files_in_scope()
    elif args.paths:
        changed = [p for p in args.paths if _in_scope(p)]
    else:
        changed = [p for p in _git_changed(args.base, args.head) if _in_scope(p)]

    md_files = [p for p in changed if p.lower().endswith((".md", ".markdown"))]
    yaml_files = [p for p in changed if p.lower().endswith((".yml", ".yaml"))]

    status = 0

    if md_files:
        status |= _run([sys.executable, str(CI_SCRIPTS / "validate_markdown_ci.py"), *md_files])
        status |= _run([sys.executable, str(CI_SCRIPTS / "validate_images_ci.py"), *md_files])
    else:
        print("Skipping markdown/image validation (no markdown files changed).")

    if yaml_files:
        status |= _run([sys.executable, str(CI_SCRIPTS / "validate_yaml_ci.py"), *yaml_files])
    else:
        print("Skipping YAML validation (no yaml files changed).")

    if status != 0:
        print("\nIncoming change validation failed.")
        return 1

    print("\nIncoming change validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
