from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable


MANIFEST_SCHEMA_VERSION = 1
STAGE1_PARTITIONS = frozenset({"train_core", "dev", "stage1_test"})
ALL_PARTITIONS = STAGE1_PARTITIONS | {"final_system_test"}


def _canonical_payload(manifest: dict[str, Any]) -> bytes:
    payload = {key: value for key, value in manifest.items() if key != "checksum_sha256"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def manifest_checksum(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_payload(manifest)).hexdigest()


def _stratified_holdout(
    scene_names: Iterable[str],
    holdout_fraction: float,
    metadata: dict[str, dict[str, Any]],
    seed: int,
) -> tuple[list[str], list[str]]:
    names = sorted(set(scene_names))
    if len(names) < 2:
        raise ValueError("Each official split needs at least two scenes.")
    holdout_count = min(max(int(round(len(names) * holdout_fraction)), 1), len(names) - 1)
    strata: dict[tuple[str, ...], list[str]] = {}
    for name in names:
        row = metadata.get(name, {})
        key = (
            str(row.get("location", "unknown")),
            str(row.get("lighting", "unknown")),
            str(row.get("weather", "unknown")),
        )
        strata.setdefault(key, []).append(name)
    rng = random.Random(seed)
    shuffled: dict[tuple[str, ...], list[str]] = {}
    quotas: dict[tuple[str, ...], int] = {}
    fractions = []
    for key in sorted(strata):
        values = sorted(strata[key])
        rng.shuffle(values)
        shuffled[key] = values
        exact = len(values) * holdout_fraction
        quotas[key] = min(int(math.floor(exact)), len(values))
        fractions.append((exact - math.floor(exact), rng.random(), key))
    remaining = holdout_count - sum(quotas.values())
    for _, _, key in sorted(fractions, reverse=True):
        if remaining <= 0:
            break
        if quotas[key] < len(shuffled[key]):
            quotas[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("Unable to allocate the requested stratified holdout size.")
    holdout = {name for key, values in shuffled.items() for name in values[: quotas[key]]}
    return sorted(set(names) - holdout), sorted(holdout)


def create_split_manifest(
    train_scene_names: Iterable[str],
    val_scene_names: Iterable[str],
    scene_metadata: dict[str, dict[str, Any]] | None = None,
    seed: int = 42,
    version: str = "v1.0-trainval",
    excluded_scenes: dict[str, str] | None = None,
) -> dict[str, Any]:
    metadata = dict(scene_metadata or {})
    train_names = sorted(set(train_scene_names))
    val_names = sorted(set(val_scene_names))
    overlap = set(train_names) & set(val_names)
    if overlap:
        raise ValueError(f"Official train/val scene lists overlap: {sorted(overlap)[:3]}")
    train_core, dev = _stratified_holdout(train_names, 0.15, metadata, seed)
    stage1_test, final_system_test = _stratified_holdout(val_names, 0.50, metadata, seed + 1)
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": "nuScenes",
        "version": version,
        "seed": int(seed),
        "experimental_unit": "scene",
        "partitions": {
            "train_core": train_core,
            "dev": dev,
            "stage1_test": stage1_test,
            "final_system_test": final_system_test,
        },
        "scene_metadata": {name: metadata.get(name, {}) for name in sorted(train_names + val_names)},
        "sealed_partitions": ["final_system_test"],
        "excluded_scenes": dict(sorted((excluded_scenes or {}).items())),
    }
    manifest["checksum_sha256"] = manifest_checksum(manifest)
    validate_split_manifest(manifest)
    return manifest


def validate_split_manifest(manifest: dict[str, Any]) -> None:
    if int(manifest.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported comparison split manifest schema.")
    if manifest.get("experimental_unit") != "scene":
        raise ValueError("Comparison split manifest must use scene as the experimental unit.")
    partitions = manifest.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != ALL_PARTITIONS:
        raise ValueError(f"Split manifest must define exactly {sorted(ALL_PARTITIONS)}.")
    seen: set[str] = set()
    for name in sorted(ALL_PARTITIONS):
        values = partitions[name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Partition {name!r} must be a non-empty scene list.")
        if len(values) != len(set(values)):
            raise ValueError(f"Partition {name!r} contains duplicate scenes.")
        overlap = seen & set(values)
        if overlap:
            raise ValueError(f"Split partitions overlap at scenes: {sorted(overlap)[:3]}")
        seen.update(values)
    expected = str(manifest.get("checksum_sha256", ""))
    actual = manifest_checksum(manifest)
    if expected != actual:
        raise ValueError(f"Split manifest checksum mismatch: expected {expected}, computed {actual}.")


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_split_manifest(manifest)
    return manifest


def partition_scene_names(
    path: str | Path,
    partition: str,
    *,
    allow_final_system_test: bool = False,
    expected_version: str | None = None,
) -> list[str]:
    if partition not in ALL_PARTITIONS:
        raise ValueError(f"Unknown comparison partition {partition!r}.")
    if partition == "final_system_test" and not allow_final_system_test:
        raise PermissionError("final_system_test is sealed until the system is frozen.")
    manifest = load_split_manifest(path)
    if expected_version and manifest.get("version") != expected_version:
        raise ValueError(
            f"Split manifest version {manifest.get('version')!r} does not match {expected_version!r}."
        )
    return list(manifest["partitions"][partition])
