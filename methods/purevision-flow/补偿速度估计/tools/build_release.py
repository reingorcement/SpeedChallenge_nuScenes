"""Build a transferable archive without caches, environments or older smoke runs."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roll_velocity.io import sha256, write_json


def selected(path):
    relative = path.relative_to(ROOT)
    if any(part in {"__pycache__", ".venv", ".git", "runs"} for part in relative.parts):
        return False
    if relative.parts[:2] == ("verification", "smoke"):
        return False
    if path.name == "package_manifest.json" or path.suffix in {".zip", ".partial", ".pyc", ".sha256"}:
        return False
    if path.suffix == ".pt":
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="roll_velocity_server_20261020.zip")
    args = parser.parse_args()
    if Path(args.name).name != args.name or not args.name.endswith(".zip"):
        raise ValueError("Archive name must be a filename ending in .zip")
    archive = ROOT / args.name
    if archive.exists():
        raise ValueError(f"Archive already exists; choose a new --name: {archive}")
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and selected(p))
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), release="F_RC server 1.0.0",
                    excluded=["generated .pt flow caches", "runs", ".venv", "__pycache__", "superseded verification/smoke",
                              "release archives", "archive checksums", "manifest itself"],
                    files=[dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha256(p)) for p in files])
    write_json(ROOT / "package_manifest.json", manifest)
    files.append(ROOT / "package_manifest.json")
    temporary = archive.with_suffix(".zip.partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as output:
        for p in files:
            output.write(p, "roll_velocity_server/" + p.relative_to(ROOT).as_posix())
    temporary.replace(archive)
    with zipfile.ZipFile(archive) as check:
        bad = check.testzip()
        if bad:
            raise RuntimeError(f"Archive CRC failure: {bad}")
        if len(check.namelist()) != len(files):
            raise RuntimeError("Archive entry count mismatch")
    checksum = sha256(archive)
    archive.with_suffix(".zip.sha256").write_text(checksum + "  " + archive.name + "\n", encoding="ascii")
    print(f"Files: {len(files)}; ZIP: {archive.stat().st_size / 1024 ** 2:.2f} MiB; CRC: passed")
    print(f"SHA256: {checksum}")


if __name__ == "__main__":
    main()
