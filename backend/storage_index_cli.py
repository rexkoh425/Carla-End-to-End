#!/usr/bin/env python3
"""
Prebuild the Storage index without starting the backend server.

This script is stdlib-only and safe to run on the host. It mirrors the backend's
indexer: reads Storage/.storageignore (glob patterns), walks Storage/, and writes
an index mapping filenames to repo-relative paths.

Usage (from repo root):
  python backend/storage_index_cli.py
  python backend/storage_index_cli.py --storage-root Storage --output Storage/storage_index.json
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.json"


def load_storage_config() -> Dict[str, str]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_ignore(path: Path) -> List[str]:
    if not path.exists():
        return []
    patterns: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line.replace("\\", "/").lstrip("/"))
    return patterns


def should_ignore(rel_to_storage: Path, patterns: Sequence[str]) -> bool:
    if not patterns:
        return False
    rel_posix = rel_to_storage.as_posix()
    top = rel_posix.split("/", 1)[0]
    for pat in patterns:
        if not pat:
            continue
        if fnmatch.fnmatch(rel_posix, pat) or fnmatch.fnmatch(top, pat):
            return True
    return False


def build_index(storage_root: Path, ignore_patterns: Sequence[str], alias: str) -> Dict[str, List[str]]:
    alias = alias.strip() or storage_root.name or "Storage"
    index: Dict[str, List[str]] = {}
    if not storage_root.exists():
        raise SystemExit(f"Storage root not found: {storage_root}")

    prefix = Path(alias)
    for path in storage_root.rglob("*"):
        if not path.is_file():
            continue
        rel_storage = path.relative_to(storage_root)
        if should_ignore(rel_storage, ignore_patterns):
            continue
        rel_index = (prefix / rel_storage).as_posix()
        index.setdefault(path.name, []).append(rel_index)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Prebuild Storage index.")
    config = load_storage_config()
    env_storage_root = os.environ.get("STORAGE_ROOT")
    cfg_storage_root = config.get("storage_root")
    base_storage = env_storage_root or cfg_storage_root or (ROOT_DIR / "Storage")
    default_storage = Path(base_storage).expanduser()
    if not default_storage.is_absolute():
        default_storage = (ROOT_DIR / default_storage).resolve()
    parser.add_argument("--storage-root", type=Path, default=default_storage, help="Path to Storage directory.")
    parser.add_argument(
        "--ignore-file",
        type=Path,
        default=None,
        help="Path to .storageignore (defaults to <storage-root>/.storageignore).",
    )
    parser.add_argument(
        "--storage-alias",
        type=str,
        default=os.environ.get("STORAGE_ALIAS") or config.get("storage_alias") or "Storage",
        help="Name prefix used inside the index (defaults to 'Storage' unless STORAGE_ALIAS is set).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for index JSON (defaults to env STORAGE_INDEX_PATH or Storage/storage_index.json).",
    )
    args = parser.parse_args()

    storage_root = args.storage_root.resolve()
    ignore_file = (args.ignore_file or storage_root / ".storageignore").resolve()
    storage_alias = (args.storage_alias or "Storage").strip() or "Storage"
    output = args.output
    if output is None:
        env_output = os.environ.get("STORAGE_INDEX_PATH")
        cfg_output = config.get("storage_index_path")
        if env_output:
            output = Path(env_output).expanduser()
        elif cfg_output:
            output = Path(cfg_output).expanduser()
        else:
            output = storage_root / "storage_index.json"
    output = output.resolve()

    ignore_patterns = load_ignore(ignore_file)
    print(f"Building index from: {storage_root}")
    print(f"Index prefix: {storage_alias}")
    print(f"Ignore patterns: {', '.join(ignore_patterns) if ignore_patterns else 'none'}")
    index = build_index(storage_root, ignore_patterns, storage_alias)
    total = sum(len(v) for v in index.values())

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index), encoding="utf-8")
    print(f"Indexed {total} files across {len(index)} unique names.")
    print(f"Wrote index to: {output}")


if __name__ == "__main__":
    main()
