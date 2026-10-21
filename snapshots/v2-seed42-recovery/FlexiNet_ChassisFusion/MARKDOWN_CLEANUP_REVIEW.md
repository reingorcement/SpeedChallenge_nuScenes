# Markdown 清理标注

> 本文件是临时删除清单。完成清理后，本文件本身也可以删除。

## 判定前提

- 当前研究主线按 `model/v2/`、`train_v2.py` 和 `configs/velocity_xy_v2_nuscenes.yaml` 判断。
- `model/fusion_model.py`、`model/physics_velocity.py` 和旧配置仍作为对照基线存在。
- “建议删除”表示内容已失真、已被新文档覆盖或只是已完成任务的过程文件。
- “建议归档”表示不是当前说明，但仍有历史实验或旧基线复现价值。
- 本次只标注，没有删除或改写任何原 Markdown。

## 一、建议删除

以下 9 份文件已经不适合作为当前项目文档，且没有代码依赖。

| 文件 | 标记 | 原因 |
|---|---|---|
| `当前算法不足.md` | `[建议删除]` | 以旧标量速度代码为“当前算法”，其中多处描述已被二维速度和 V2 实现推翻，标题尤其容易误导。 |
| `视觉主导弱模型速度估计技术路线讨论稿.md` | `[建议删除]` | 文件自己标明是未形成决策的讨论稿；已被根目录修订版 V2、`V2_IMPLEMENTATION_REPORT.md` 和架构文档覆盖。 |
| `PROJECT_LEARNING_PLAN.md` | `[建议删除]` | 学习路径仍围绕旧 `64x64`、标量 `predicted_speed` 和旧 README，已经不符合当前 V2 主线。 |
| `AI_CODE_CONTEXT.md` | `[建议删除]` | 自称“当前分支优先上下文”，但仍把旧 `train.py`、非因果默认配置和旧融合模型作为主线，会直接误导后续 AI。 |
| `execution_plan.md` | `[建议删除]` | VKD gated refactor 的已完成实施计划，不是当前设计文档；对应代码和历史报告已经存在。 |
| `gpu_training_optimization_plan.md` | `[建议删除]` | 已完成的一次性任务计划，最终建议已单独写入 recommendations 文件。 |
| `gpu_training_optimization_notes.md` | `[建议删除]` | GPU 优化过程笔记，内容已被 recommendations 汇总，且参数针对旧 VKD/64x64 路线。 |
| `task_plan.md` | `[建议删除]` | Codex 过程状态、错误日志和阶段勾选记录；最终技术结论已经写入两份 V2 报告。后续如需新计划应重新创建。 |
| `notes.md` | `[建议删除]` | Codex 中间研究笔记，内容已汇总到 V2 实施报告和架构文档，不应作为用户文档长期保留。 |

### 建议删除顺序

先删除明显失真的说明，再删除过程文件：

```text
当前算法不足.md
视觉主导弱模型速度估计技术路线讨论稿.md
PROJECT_LEARNING_PLAN.md
AI_CODE_CONTEXT.md
execution_plan.md
gpu_training_optimization_plan.md
gpu_training_optimization_notes.md
task_plan.md
notes.md
```

## 二、建议归档，不建议直接删除

这些文件不再代表 V2 主线，但仍能解释旧配置、旧实验或历史决策。建议移动到 `docs/archive/legacy_vkd/` 或类似目录。

| 文件 | 标记 | 原因 |
|---|---|---|
| `速度XY改造使用说明.md` | `[建议归档]` | 对旧二维速度基线仍有复现价值，但没有覆盖 `model/v2` 和 `train_v2.py`。 |
| `服务器全量nuScenes_VKD运行说明.md` | `[建议归档]` | 是旧 `velocity_xy_vkd_gated.yaml` 的服务器说明，不是当前 V2 服务器流程。 |
| `implementation_report.md` | `[建议归档]` | 记录旧因果物理估计器的实现与验证，仍可解释 `model/physics_velocity.py` 的来源。 |
| `gpu_training_optimization_recommendations.md` | `[建议归档]` | 建议面向旧 64x64 VKD 配置，不能直接套用于当前 288x512 V2，但旧基线复现时仍有参考价值。 |
| `REPORT_FOR_ADVISOR.md` | `[建议归档]` | 老师汇报材料基于旧标量速度任务和旧指标，当前不能继续使用，但属于历史汇报记录。 |
| `analysis_reports/vkd_0704/REPORT_VKD_0704.md` | `[建议归档]` | 是旧 VKD 实验结果证据，不是当前算法说明；建议和对应 CSV/图像一起归档。 |

## 三、建议保留

| 文件 | 标记 | 原因 |
|---|---|---|
| `V2_NUSCENES_CARSIM_ARCHITECTURE.md` | `[保留]` | 当前最完整的 nuScenes 能力边界、V2 结构和 CarSim 路线说明。 |
| `V2_IMPLEMENTATION_REPORT.md` | `[保留]` | 当前 V2 已实现内容、参数、服务器入口、验收门槛和剩余缺陷的实施记录。 |
| `运动学动力学特征提取器数据接口说明.md` | `[保留]` | 对后续弱运动学分支和 CarSim 动力学学生接口设计仍有直接价值。后续需要补 V2 接口差异。 |
| `物理混合改造使用说明.md` | `[保留旧基线]` | 与仍存在的 `physics_residual_gated`、physics-only 配置直接对应，适合保留为对照基线说明。 |
| `eovm_ekf/README.md` | `[保留]` | 是独立 EOVM-EKF 子模块的最小使用与状态定义说明。 |
| `analysis/README.md` | `[保留]` | 与当前仍存在的分析脚本直接对应，删除后会降低分析工具可用性。 |

## 四、必须重写，但暂时不要删除

| 文件 | 标记 | 原因 |
|---|---|---|
| `README.md` | `[必须重写]` | 项目入口仍描述标量 `predicted_speed = base_speed + delta_v`、旧目录和旧默认参数。应改成“V2 主线 + legacy baselines”导航；在新 README 完成前不要删除。 |

建议新的 README 只承担导航职责：

1. 当前任务与输出参考点；
2. V2 主线入口；
3. legacy baseline 入口；
4. 数据审计、光流预计算、训练和测试命令；
5. 指向两份 V2 文档和归档目录。

## 五、自动生成报告：可删除，但不是无用文档

以下文件由分析脚本生成，不参与代码运行，并且有对应 JSON/CSV 或可重新运行脚本生成。若目的是减少仓库噪声，可以删除；若要保留某次实验快照，则应和同目录结果一起保留。

| 文件 | 标记 | 再生成来源 |
|---|---|---|
| `outputs/fair_rgb_flow_geom/eval_best/analysis/val_report.md` | `[可重新生成]` | `analysis/evaluate_checkpoint.py` |
| `outputs/fair_rgb_flow_geom/eval_last/analysis/val_report.md` | `[可重新生成]` | `analysis/evaluate_checkpoint.py` |
| `outputs/fair_rgb_flow_geom/eval_compare/comparison.md` | `[可重新生成]` | `analysis/compare_checkpoints.py` |

这两份 `val_report.md` 不是重复文件，哈希不同，分别对应 best 和 last 评估，不应以文件大小相同为理由只删除其中一个。

## 六、推荐的最终 Markdown 结构

```text
FlexiNet_ChassisFusion/
  README.md                              # 重写后的总入口
  V2_IMPLEMENTATION_REPORT.md            # 当前实现
  V2_NUSCENES_CARSIM_ARCHITECTURE.md     # 当前架构与数据路线
  运动学动力学特征提取器数据接口说明.md       # 后续分支接口
  物理混合改造使用说明.md                   # legacy physics baseline

  analysis/README.md
  eovm_ekf/README.md

  docs/archive/legacy_vkd/
    速度XY改造使用说明.md
    服务器全量nuScenes_VKD运行说明.md
    implementation_report.md
    gpu_training_optimization_recommendations.md
    REPORT_FOR_ADVISOR.md
    REPORT_VKD_0704.md
```

生成在 `outputs/` 下的 Markdown 不需要移动进文档目录，它们应继续跟随各自实验结果目录，或与整组生成结果一起清理。

## 七、最终清理建议

- 可以直接删除：9 份。
- 建议移动归档：6 份。
- 建议保留：6 份。
- 必须重写：1 份 `README.md`。
- 可随生成结果清理：3 份。
- 本清单：清理完成后删除。
