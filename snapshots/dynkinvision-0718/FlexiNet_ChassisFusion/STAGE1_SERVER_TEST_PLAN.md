# nuScenes 第一阶段服务器训练、测试与消融方案

## 结论

第一阶段训练结束后必须测试，而且不能继续使用早停所用的 `dev` 数据冒充测试集。本方案使用一个服务器命令完成数据隔离、训练、独立测试、消融、场景级统计和报告生成：

```bash
bash scripts/run_v2_server_pipeline.sh
```

脚本最终退出码为 `0` 表示全部验收门槛通过；退出码为 `2` 表示训练和报告已经完成，但模型没有通过第一阶段验收。发生数据、CUDA 或训练错误时返回其他非零退出码。

## 数据划分

固定随机种子为 `42`，实验独立单位为 scene：

| nuScenes 官方数据 | 子集 | 用途 | 第一阶段是否可见 |
|---|---|---|---|
| train | `train_core` 85% | 训练、运动学统计、场景加权 | 是 |
| train | `dev` 15% | 早停、选 checkpoint | 是 |
| val | `stage1_test` 50% | 第一阶段独立测试 | 仅训练结束后 |
| val | `final_system_test` 50% | CarSim 第二阶段完成后的最终测试 | 否，代码强制封存 |

精确 scene 清单、分层元数据和 SHA-256 校验和写入 `logs/v2_split_manifest.json`。RAFT 只预计算 `train_core/dev/stage1_test`，CAN 间隔与归一化统计只使用 `train_core`。

## 自动执行内容

1. 数据与 CAN 字段审计。
2. RAFT 光流预计算。
3. 几何分支审计。
4. 三个配对 seed `42/3407/2026` 的 visual、kinematic pretrain、joint 训练。
5. 在 `stage1_test` 上分别评估 geometry、visual、joint。
6. 自动执行关闭运动学修正、关闭视觉残差、关闭轮速尺度、关闭偏航补偿、CAN 缺失、CAN 过期、图像打乱、光流打乱、CAN 打乱等消融或反事实测试。
7. 单独统计几何失效、轮速回退、转弯、高侧向速度和轮速不一致样本。
8. 以 scene 为单位进行配对 bootstrap，生成 95% 置信区间。

## 验收门槛

- Geometry：有效率不低于 60%，有效样本条件数中位数低于 $10^5$，直行样本纵向速度符号一致率不低于 95%。
- Visual：EPE 不得比 geometry 恶化超过 2%，且 $v_x$ 或 $v_y$ 的 MAE 至少一项改善。
- Joint：三 seed 平均 EPE 不得比 visual 恶化超过 2%；至少 2 个 seed 在转弯或高侧向样本的 $v_y$ MAE 上改善。
- 所有模型输出必须有限；运动学修正严格满足 $|\Delta v_x|\leq1.0\,\mathrm{m/s}$、$|\Delta v_y|\leq0.5\,\mathrm{m/s}$。

这些门槛是工程验收条件，不替代不确定性分析。报告同时给出 joint 相对 visual 的 scene 配对差值及 95% bootstrap 置信区间。

## 输出

最终结果位于 `outputs/stage1_test_report/`：

- `stage1_report.md`：结论、核心指标、验收结果、置信区间。
- `stage1_report.json`：完整机器可读结果。
- `per_scene_metrics.csv`：每个模型、seed、scene 的指标。
- `ablation_metrics.csv`：全部消融结果及相对基线变化。
- `bad_cases.csv`：按三 seed 平均误差排序的 100 个最差样本。

逐样本预测和模型内部量位于 `outputs/stage1_test/*/predictions.csv`，包含几何速度、条件数、支持点、fallback、视觉残差、运动学门控、不确定性和修正量。

## 与第二阶段的关系

第一阶段通过后，visual 与弱运动学编码器才可作为 CarSim 动力学教师/学生阶段的冻结基线。第二阶段开发过程中仍不得读取 `final_system_test`；只有 CarSim 分支、融合规则和所有超参数全部冻结后，才允许进行一次最终系统测试。
