#!/usr/bin/env python3
"""
Build a storage index using the storage path defined in config.json.

Reads config.json at the repo root to locate:
  - storage_root: the base directory that holds your data (e.g., D:/Datasets)
  - storage_index_path: where to write the index JSON (defaults to <storage_root>/storage_index.json)
  - storage_alias: optional display prefix (defaults to "Storage")

The index maps filename -> list of paths prefixed with the storage alias (e.g., Storage/...). This keeps the JSON usable both on host (alias resolves to storage_root) and inside Docker (alias is mounted at /Storage).
Ignore patterns (one per line, glob syntax) can be placed in <storage_root>/.storageignore.

Usage (from repo root):
  python utils/storage_index_builder.py
"""

from __future__ import annotations

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
    norm = pat.strip()
    if not norm:
      continue
    if fnmatch.fnmatch(rel_posix, norm) or fnmatch.fnmatch(top, norm):
      return True
  return False


def build_index(storage_root: Path, ignore_patterns: Sequence[str], alias: str) -> Dict[str, List[str]]:
  index: Dict[str, List[str]] = {}
  if not storage_root.exists():
    raise SystemExit(f"Storage root not found: {storage_root}")

  prefix = Path(alias or "Storage")
  for path in storage_root.rglob("*"):
    if not path.is_file():
      continue
    rel_storage = path.relative_to(storage_root)
    if should_ignore(rel_storage, ignore_patterns):
      continue
    logical_entry = (prefix / rel_storage).as_posix()
    index.setdefault(path.name, []).append(logical_entry)
  return index


def main() -> None:
  cfg = load_storage_config()
  env_root = os.environ.get("STORAGE_ROOT")
  storage_root = Path(env_root or cfg.get("storage_root") or (ROOT_DIR / "Storage")).expanduser()
  if not storage_root.is_absolute():
    storage_root = (ROOT_DIR / storage_root).resolve()

  storage_alias = (cfg.get("storage_alias") or "Storage").strip() or "Storage"

  env_index = os.environ.get("STORAGE_INDEX_PATH")
  cfg_index = cfg.get("storage_index_path")
  output = Path(env_index or cfg_index or (storage_root / "storage_index.json")).expanduser()
  if not output.is_absolute():
    output = (ROOT_DIR / output).resolve()

  ignore_file = (storage_root / ".storageignore").resolve()
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
