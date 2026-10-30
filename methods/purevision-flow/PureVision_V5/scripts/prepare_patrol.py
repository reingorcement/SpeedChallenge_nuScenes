"""Select exact task files for an independent, bounded AI read; never include environment values."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import secrets
import shlex
import hashlib
import subprocess


def prepare(root, prefix, snapshot, *, startup=None):
    targets = []
    def add(path, run_id=None, mode="head"):
        path = Path(path)
        if path.is_file():
            targets.append(dict(path=str(path.resolve()), run_id=run_id, mode=mode))
    add(prefix.with_suffix(".snapshot.json"), mode="json_summary")
    if startup:
        proof = json.loads(startup.read_text())
        run_id = proof["run_id"]
        if startup.resolve() != (root / "monitoring/startup" / run_id / "startup_proof.json").resolve():
            raise ValueError("Startup proof path/run identity mismatch")
        add(startup, run_id)
        for name in ("run.json", "status.json", "metrics.jsonl"):
            add(root / "runs" / run_id / name, run_id)
        add(root / "logs" / f"{run_id}.log", run_id, "tail")
        if len(targets) != 6:
            raise ValueError("Startup patrol is missing required target files")
    else:
        for job in snapshot.get("jobs", []):
            if job["status"] in {"completed", "failed", "interrupted", "cancelled", "blocked"}:
                continue
            run_id = job["job_id"]
            if job.get("run_dir"):
                add(Path(job["run_dir"]) / "run.json", run_id)
                add(Path(job["run_dir"]) / "status.json", run_id)
            if job.get("status_path"):
                add(job["status_path"], run_id)
            if job.get("log_path"):
                add(job["log_path"], run_id, "tail")
        # A no-job probe still verifies a real local task artifact, the host snapshot.
    targets = list({(t["path"], t["run_id"]): t for t in targets}.values())
    request = dict(schema="v5_patrol_read_request_v1", request_id=secrets.token_hex(16), root=str(root), targets=targets)
    helper = Path(__file__).with_name("patrol_read.py")
    repository = Path(__file__).resolve().parents[4]
    base = ["git", "-c", f"safe.directory={repository}", "-C", str(repository)]
    try:
        request["code"] = dict(commit=subprocess.check_output(base + ["rev-parse", "HEAD"], text=True).strip(),
                               dirty=bool(subprocess.check_output(base + ["status", "--porcelain", "--untracked-files=no"], text=True).strip()),
                               helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest())
    except (OSError, subprocess.SubprocessError):
        request["code"] = {"status": "unavailable"}
    command = ["/home/zhangyu/.conda/envs/speedest/bin/python", "-B", str(helper), "--request", str(prefix.with_suffix(".request.json"))]
    request["command"] = command
    prefix.with_suffix(".request.json").write_text(json.dumps(request, indent=2) + "\n")
    prompt = """你是本机 PureVision V5 的只读 Astra 巡视员。只做诊断；不得修改代码、参数、配置、作业状态，不得启动/停止/重启训练，不访问外网或凭据。
必须先执行下面这条完整命令（原样运行，不要重定向或包在其他Python代码中），它实际读取本次指定工件并输出文件内容和读取收据。不能仅复述提示词或把其他run的状态当作目标run通过。若命令失败或缺文件，报告失败，不写成功结束语。
%s
宿主机快照是PID/GPU/磁盘检查的依据；sandbox的/proc或GPU不可见不能推断宿主进程死亡。核查目标run ID、训练步数与验证指标、日志异常、checkpoint更新时间、GPU/磁盘；unknown必须保留为待核实，不能算终止。正常损失波动不触发修改。报告实际读取路径、进展、异常及人工定位建议。明确host确定性检查与本次AI诊断分别是什么；不声称修改了bug。
若目标包含startup_proof，交叉核对同run步骤1/2/3、首次验证、恢复状态与commit/contract；不要自行加载大checkpoint。
只把简短中文报告放在最终回答。只有命令成功且完成目标读取后，以独立一行 ASTRA_PATROL_OK 结束。
""" % shlex.join(command)
    if not any(t.get("run_id") for t in targets):
        prompt += "本次没有登记的活跃run，是无作业连接/读取探针；请明确该事实，不应视作某个训练run已经被巡视或通过启动检查。\n"
    prefix.with_suffix(".prompt.txt").write_text(prompt)
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--startup", type=Path)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.prefix, json.loads(args.prefix.with_suffix(".snapshot.json").read_text()), startup=args.startup)


if __name__ == "__main__":
    main()
