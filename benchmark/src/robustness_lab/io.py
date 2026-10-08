"""Serialization, hashing, and atomic artifact helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


def to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_plain(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_data(path: str) -> Dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        if source.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.safe_load(handle)
        else:
            value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("expected a mapping in %s" % source)
    return value


def atomic_write_text(path: str, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=str(destination.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def dump_json(path: str, value: Any) -> None:
    atomic_write_text(path, json.dumps(to_plain(value), indent=2, sort_keys=True) + "\n")


def dump_yaml(path: str, value: Any) -> None:
    atomic_write_text(path, yaml.safe_dump(to_plain(value), sort_keys=False))


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_data(value: Any) -> str:
    payload = json.dumps(to_plain(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_tree(path: str, suffixes: tuple = (".py",)) -> str:
    """Hash stable relative paths plus file bytes for a source tree."""
    root = Path(path).resolve()
    digest = hashlib.sha256()
    files = [item for item in root.rglob("*") if item.is_file() and item.suffix in suffixes and "__pycache__" not in item.parts]
    for item in sorted(files, key=lambda value: str(value.relative_to(root))):
        digest.update(str(item.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def runtime_fingerprint() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    content = "".join(json.dumps(to_plain(row), sort_keys=True) + "\n" for row in rows)
    atomic_write_text(path, content)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        atomic_write_text(path, "")
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=str(destination.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_telemetry(path_without_suffix: str, rows: List[Dict[str, Any]]) -> Dict[str, str]:
    """Write Parquet when pyarrow is installed and always write JSONL evidence."""
    artifacts = {}
    jsonl_path = path_without_suffix + ".jsonl"
    write_jsonl(jsonl_path, rows)
    artifacts["telemetry_jsonl"] = jsonl_path
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet_path = path_without_suffix + ".parquet"
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, parquet_path, compression="zstd")
        artifacts["telemetry_parquet"] = parquet_path
    except (ImportError, ModuleNotFoundError):
        artifacts["telemetry_parquet_status"] = "skipped: install robustness-lab[analytics]"
    return artifacts
