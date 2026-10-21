"""Read-only verification that both source projects remain byte-identical."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "verification/source_baseline.json").read_text())
    errors = []
    for name, record in baseline.items():
        path = root.parent / name
        if not path.is_file() or path.stat().st_size != record["size"]:
            errors.append(name)
            continue
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != record["sha256"]:
            errors.append(name)
    actual = {p.relative_to(root.parent).as_posix() for folder in ["PureVision_V3", "补偿速度估计"]
              for p in (root.parent / folder).rglob("*") if p.is_file()}
    errors.extend("added:" + name for name in sorted(actual - set(baseline)))
    for p in (root / "purevision_v4").rglob("*.py"):
        for node in ast.walk(ast.parse(p.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            if any(name.split(".")[0] in {"model", "dataset", "roll_velocity", "utils"} for name in names):
                errors.append(f"unisolated import: {p}:{node.lineno}")
    errors.extend("symlink:" + str(p) for p in root.rglob("*") if p.is_symlink())
    result = dict(source_files=len(baseline), unchanged=not errors, errors=errors)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
