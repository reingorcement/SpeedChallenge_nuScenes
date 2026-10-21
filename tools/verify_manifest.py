"""Verify files shipped in the current branch, without importing project code."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "FILE_MANIFEST.json").read_text(encoding="utf-8"))
errors = []
for row in manifest["files"]:
    path = root / row["path"]
    if not path.is_file() or path.is_symlink():
        errors.append(row["path"])
        continue
    data = path.read_bytes()
    if "sha256" in row:
        digest = hashlib.sha256(data).hexdigest()
        expected = row["sha256"]
    else:
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        expected = row["git_blob"]
    if digest != expected:
        errors.append(row["path"])
if errors:
    raise SystemExit("MISMATCH:\n" + "\n".join(errors))
print(f"PASS: {len(manifest['files'])} delivery files verified")
