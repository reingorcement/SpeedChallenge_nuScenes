from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch


GEOMETRY_CACHE_VERSION = 1
GEOMETRY_CACHE_FIELDS = (
    "velocity",
    "covariance",
    "condition_number",
    "residual_rms",
    "valid",
    "support_count",
)


def geometry_cache_fingerprint(contract: dict[str, Any]) -> str:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path | None) -> str:
    if not path:
        return "none"
    source = Path(path)
    if not source.is_file():
        return "missing"
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SceneGeometryCache:
    def __init__(
        self,
        root: str | Path,
        expected_contract: dict[str, Any],
        required_tokens: Iterable[str],
    ) -> None:
        self.root = Path(root)
        required_tokens = [str(token) for token in required_tokens]
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Geometry cache manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("version", -1)) != GEOMETRY_CACHE_VERSION:
            raise ValueError("Geometry cache version does not match the V2 reader.")
        expected_fingerprint = geometry_cache_fingerprint(expected_contract)
        if manifest.get("fingerprint") != expected_fingerprint:
            raise ValueError(
                "Geometry cache fingerprint mismatch; regenerate it for the resolved config."
            )
        if manifest.get("contract") != expected_contract:
            raise ValueError("Geometry cache contract does not match the resolved config.")

        self._token_to_location: dict[str, tuple[str, int]] = {}
        self._scene_token_counts: dict[str, int] = {}
        for scene_name, tokens in manifest.get("scenes", {}).items():
            self._scene_token_counts[str(scene_name)] = len(tokens)
            for index, token in enumerate(tokens):
                self._token_to_location[str(token)] = (str(scene_name), index)
        missing = [token for token in required_tokens if token not in self._token_to_location]
        if missing:
            raise KeyError(
                f"Geometry cache is missing {len(missing)} required tokens, e.g. {missing[0]}."
            )
        required_scenes = {
            self._token_to_location[token][0] for token in required_tokens
        }
        missing_scene_files = [
            scene for scene in sorted(required_scenes) if not (self.root / f"{scene}.pt").is_file()
        ]
        if missing_scene_files:
            raise FileNotFoundError(
                f"Geometry cache is missing scene file: {missing_scene_files[0]}.pt"
            )
        self._scene_packages: dict[str, dict[str, torch.Tensor]] = {}

    def _load_scene(self, scene_name: str) -> dict[str, torch.Tensor]:
        package = self._scene_packages.get(scene_name)
        if package is not None:
            return package
        path = self.root / f"{scene_name}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Geometry scene cache not found: {path}")
        raw = torch.load(path, map_location="cpu")
        if not isinstance(raw, dict):
            raise ValueError(f"Geometry scene cache must be a dictionary: {path}")
        package = {field: torch.as_tensor(raw[field]) for field in GEOMETRY_CACHE_FIELDS}
        count = int(package["velocity"].shape[0])
        if count != self._scene_token_counts[scene_name]:
            raise ValueError(
                f"Geometry scene count mismatch for {scene_name}: "
                f"{count} != {self._scene_token_counts[scene_name]}"
            )
        expected_shapes = {
            "velocity": (count, 2),
            "covariance": (count, 3, 3),
            "condition_number": (count,),
            "residual_rms": (count,),
            "valid": (count,),
            "support_count": (count,),
        }
        for field, shape in expected_shapes.items():
            if tuple(package[field].shape) != shape:
                raise ValueError(f"Invalid {field} shape in {path}: {tuple(package[field].shape)}")
        if package["valid"].dtype != torch.bool:
            raise TypeError(f"Geometry valid mask must be boolean: {path}")
        self._scene_packages[scene_name] = package
        return package

    def get(self, token: str) -> dict[str, torch.Tensor]:
        if token not in self._token_to_location:
            raise KeyError(f"Geometry token is absent from cache: {token}")
        scene_name, index = self._token_to_location[token]
        package = self._load_scene(scene_name)
        return {field: package[field][index] for field in GEOMETRY_CACHE_FIELDS}
