#!/usr/bin/env python3
"""CI YAML validation for changed yaml files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

YAML_EXTS = {".yml", ".yaml"}


def _iter_yaml_paths(root: Path, inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in inputs:
        p = (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
        if p.is_file() and p.suffix.lower() in YAML_EXTS:
            out.append(p)
    return sorted(set(out))


def _validate_yaml(path: Path) -> str | None:
    cmd = [
        "ruby",
        "-e",
        "require 'yaml'; YAML.safe_load(File.read(ARGV[0]), permitted_classes: [], aliases: true)",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip().splitlines()[-1]
        return f"{path}: {err}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate changed yaml files")
    parser.add_argument("paths", nargs="*", help="Changed file paths")
    args = parser.parse_args()

    root = Path('.').resolve()
    yaml_files = _iter_yaml_paths(root, args.paths)
    if not yaml_files:
        print("YAML CI validation skipped (no yaml files changed).")
        return 0

    issues: list[str] = []
    for path in yaml_files:
        err = _validate_yaml(path)
        if err:
            issues.append(err)

    if issues:
        print("YAML validation failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print(f"YAML validation passed ({len(yaml_files)} file(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
