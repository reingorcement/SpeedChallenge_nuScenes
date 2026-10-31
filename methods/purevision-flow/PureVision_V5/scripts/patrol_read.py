"""Deterministic bounded file reads invoked by Astra, with verifiable per-file receipts."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path


def read_targets(request):
    root = Path(request["root"]).resolve()
    for target in request["targets"]:
        path = Path(target["path"])
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or resolved.name in {"network.env", "network.json"}:
            raise ValueError("Read target is outside task artifacts or is a private control file")
        with path.open("rb") as stream:
            info = os.fstat(stream.fileno())
            mode = target.get("mode", "head")
            if mode == "tail":
                stream.seek(max(0, info.st_size - 24000))
            blob = stream.read() if mode == "json_summary" else stream.read(24000)
        receipt = dict(request_id=request["request_id"], path=str(path), run_id=target.get("run_id"),
                       status="read_success", bytes_read=len(blob), source_size=info.st_size,
                       read_scope="full_json_summarized" if mode == "json_summary" else f"{mode}_at_most_24000_bytes", sha256=hashlib.sha256(blob).hexdigest())
        if not blob:
            raise ValueError(f"Empty required patrol target: {path}")
        print("V5_FILE_READ_RECEIPT " + json.dumps(receipt), flush=True)
        content = blob.decode("utf-8", errors="replace")
        if mode == "json_summary":
            content = json.loads(content)
            scope=request.get("scope_run_ids")
            if scope is not None:content["jobs"]=[j for j in content.get("jobs",[]) if j["job_id"] in scope]
            content["jobs"] = [{k: j.get(k) for k in ("job_id", "kind", "status", "phase", "process", "child_process",
                "epoch", "global_step", "loss", "validation", "validation_epoch_zero_based", "checkpoint_age_s", "progress", "progress_path",
                "progress_report_status", "progress_age_s", "processed_batches", "optimizer_updates", "amp_skips", "best_epe", "best_epoch", "notes")} for j in content.get("jobs", [])]
        print("V5_FILE_CONTENT " + json.dumps({"path": str(path), "content": content}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    read_targets(request)


if __name__ == "__main__":
    main()
