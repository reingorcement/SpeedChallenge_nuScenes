"""Summarize fixed-weight tests; no training or checkpoint writing."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_v3_paired_bootstrap import bootstrap, paired_scene_deltas

MODES = ("rgb_only", "flow_geometry_only", "fusion")


def summarize(output: Path) -> None:
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("test_only") is not True:
        raise ValueError("Expected test-only protocol.")
    rows, paired = [], {}
    reference = output / "fusion/predictions.csv"
    for mode in MODES:
        deltas = paired_scene_deltas(reference, output / mode / "predictions.csv")
        metrics = json.loads((output / mode / "metrics.json").read_text(encoding="utf-8"))
        if metrics["sideslip_min_speed_mps"] != protocol["beta_min_speed_mps"]:
            raise ValueError("Sideslip evaluation thresholds differ.")
        rows.append({"mode": mode, **metrics})
        if mode != "fusion" and not protocol["partial_test"]:
            paired["fusion_minus_" + mode] = bootstrap([deltas], samples=10000, random_seed=42)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if paired:
        (output / "paired_velocity_bootstrap.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    lines = [
        "# V3 已有权重测试", "",
        "部分样本功能测试，不能作为全量精度结论。" if protocol["partial_test"] else "stage1_test 场景测试；三组 token、scene 和真值一致。",
        "本次无训练、无微调、无权重更新。", "RGB 来源：" + protocol["rgb_source"],
        "光流组为固定光流几何求解，关闭 RGB 可靠性、修正及回退；不是独立训练的光流神经网络。",
        "固定 checkpoint 的分支测试不能等同于三个分别训练模型的消融；场景 bootstrap 不衡量训练种子稳定性。",
        "侧偏角单位 deg、速度单位 m/s；ego 参考点是否为质心尚未确认。", "",
        "| 模式 | vx MAE | vy MAE | EPE | β MAE | β RMSE |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cells = [row["mode"]] + [f"{row[key]:.4f}" for key in (
            "mae_vx_mps", "mae_vy_mps", "epe_mps", "sideslip_mae_deg", "sideslip_rmse_deg",
        )]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "光流求解无效时保留原求解器的零速度输出并计入误差，不丢弃失败样本。",
              "geometry_valid_rate 仅对光流和融合组有意义；RGB 组该字段为占位。",
              "侧偏角只对真值速度达到门槛的共同样本统计；预测方向未定义时计 180° 惩罚并单列数量。",
              "完整测试的速度场景置信区间见 paired_velocity_bootstrap.json；未计算侧偏角置信区间。"]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    summarize(parser.parse_args().output_dir)
