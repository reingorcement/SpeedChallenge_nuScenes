# V5 首轮数据与标签审计（2026-10-30）

本次已实际生成并核验全量 train_core/dev 清单。仅产生数据工件与数据检查结果；本页不报告模型训练精度。固定使用 `PLANAR_PROXY`，不将平面定位导数称为真实三维体轴速度。

## 已锁定的标签与输入

- 版本：`nuscenes_rear_axle_planar_nonuniform3_v1`。参考点为 **nuScenes 后轴中点**；[官方数据格式的 Sensor calibration](https://www.nuscenes.org/data-format) 将 ego frame 原点定义为后轴中点。因此直接对 ego_pose 位置求导，不施加相机杆臂补偿，也不称为质心。
- 坐标：当前相机时刻的 yaw 航向系，x 前、y 左，m/s。[官方 schema](https://www.nuscenes.org/nuscenes) 明确 ego_pose 定位只有世界 xy，z 恒为零。完整四元数不能恢复缺失的垂向运动，本版本只提取 yaw。
- `h-=t_i-t_(i-1)`、`h+=t_(i+1)-t_i`。世界 xy 导数为 `h+/(h-+h+) * (p_i-p_(i-1))/h- + h-/(h-+h+) * (p_(i+1)-p_i)/h+`，之后转到当前 yaw 系。两端帧没有标签；不进行额外平滑。
- 时间全部来自 CAM_FRONT 的 sample_data，不使用 2 Hz sample 关键帧间隔。每个样本是截至当前 t 的 5 张图像，4 对相邻前向流。未来 pose 和它的时间仅用于离线监督；导出的模型字段没有真实位姿/旋转/CAN/GNSS。
- 分离字段：推理为图像、pair_dt；监督为 target；scene/partition/log_token/token/filename 为索引和划分信息。数据载入器只从当前窗口提取帧与时间，不将完整序列 metadata 喂入模型。

旧 V3 在 `dataset/nuscenes_chassis.py:833` 与 `scripts/precompute_frames.py:145` 用两端位移除以总间隔，再表达于当前 yaw；前后间隔不等时导数对应时刻偏移。V5 使用已获同意的新版本，全六组采用相同标签；旧 head 精度不能直接列入新排名，也不能恢复旧优化器状态。

## 全量实测与门槛

数据根目录：`/mnt/sdb2/Workspace/zhangyu/dataset/nuscenes`。

| 分区 | 场景 | 图像帧 | 五帧窗口 | 标签状态 |
|---|---:|---:|---:|---|
| train_core | 582 | 135888 | 132978 | 已生成 |
| dev | 103 | 23995 | 23480 | 已生成 |
| stage1_test | 75 | 17466 | 17091 | 仅索引；target 全为空 |
| final_system_test | 75 | 17482 | 17107 | 仅索引；target 全为空 |

- 已逐项核验 **194831** 个原图路径存在、文件名/前后 token 链/场景归属、唯一 token、严格递增时间；元数据尺寸均为 1600×900。此处检查路径存在与元数据，不声称已解码所有 JPEG；实际帧缓存阶段负责图像解码检查。
- train/dev **159883** 个相机记录与对应 pose 时间完全一致（最大误差 0 微秒），159883 个 z 全为 0。非有限值、缺图、重复 token、跨 scene 链、时间倒序使准备失败。
- train/dev 相邻 Δt：159198 对；最小 0.043824 s、中位 0.100000 s、P99 0.102909 s、最大 0.300008 s。没有将采样率硬编码为 12 Hz。较长间隔仍保留真实 Δt；不按标签大小筛选样本。
- 前后间隔不对称的绝对值：中位 0.050000 s、P99 0.050388 s、最大 0.200016 s。新旧标签差异的 P95 为 `[0.067081, 0.024038]` m/s，最大 `[1.223092, 0.437431]` m/s，表明修正并非只有浮点差异。
- train/dev 的零侧向基线 MAE 分别为 **0.035626/0.035984 m/s**，RMSE 为 **0.049124/0.049530 m/s**。侧向信号很弱；必须同时报告这个基线，不能仅凭总 EPE 声称学到真实侧滑。

合成测试涵盖直线匀速、左右侧移、非均匀时间上的二次位置函数、无侧滑圆周运动、旋转杆臂、yaw 轴符号和非法时间。圆周例 `U=15 m/s,r=.4 rad/s,h-=.05 s,h+=.1 s`：旧 `vy=.149975 m/s`，新 `vy=-.000010 m/s`；2 m 前向相机杆臂带来的约 .8 m/s 侧向差异被明确区分，未错误地移到后轴标签。

一次有界敏感性检查只用 train_core 的 scene-0001（233 帧，seed42）。假设向 xy 独立加入 1 cm 高斯位置噪声，三点导数的 vx/vy 扰动 RMSE 为 `[.127070,.134436]` m/s，五点二次拟合为 `[.037226,.040486]` m/s；无人工噪声时五点与三点差异 RMSE 为 `[.033006,.006118]` m/s。**1 cm 是敏感性假设，不是实测定位误差，不据此声称真值误差已知。** 首轮锁定已确认的三点版本；额外平滑扩大监督时间范围且未取得真实速度核验，本轮不依赖该诊断调参。所有实验采用同一版本，侧向结论限于这一代理标签；后续若改变平滑，必须统一重建标签并另建实验版本。

## 划分、历史曝光与初始化

保留 V3 `logs/v2_split_manifest.json` 原有 582/103/75/75 划分。SHA-256 为 `d48ae71d98534c72d51d4629a6c1bcf775e1f5896ae5f324e2a262245e49e931`。四组 scene 完全不相交；41 个 dev log 同时在 train_core，17 个 final_system_test log 同时在 stage1_test。结论限于场景划分，不称新 log/新路线泛化。

官方 trainval 有 850 scenes，现有清单保留 835。以下 15 个场景均因历史缺少 CAN `vehicle_monitor,zoe_veh_info,steeranglefeedback` 被排除：

`scene-0161, scene-0162, scene-0163, scene-0164, scene-0165, scene-0166, scene-0167, scene-0168, scene-0170, scene-0171, scene-0172, scene-0173, scene-0174, scene-0175, scene-0176`。

V5 无 CAN 依赖；保留这些历史排除仅用于固定比较集合，不代表纯视觉需要 CAN。每场景原因逐项保存在外部 `audit.json.excluded_scenes`。

| 对象 | 可核查关系 | 本轮处理 |
|---|---|---|
| V3 direct seed42 | checkpoint config 为 causal T5、RGB72×128、epoch19，数据清单路径为 logs/v2_split_manifest.json，输出路径含 dev/direct_seed_42；`train_v3.py:409-410` 明确 train_core 训练、dev 选模；checkpoint SHA-256 `1d24368adb86643e68e28b0f69ea4ec658e008ff52b5fb7fc09d7bad06ac0e82` | 可作同一 RGB 特征初始化来源；全部新 head 用新标签训练；checkpoint 不包含实际 split 文件哈希，来源清单由交付文档与代码支持 |
| V3 fusion seed42 | 来源共享上述清单，融合初始化来自 direct；交付 VALIDATION.md 只报告 checkpoint 自带 best_epe，不是新测试 | 不把其几何速度融合当成本轮特征融合，不复用旧排名 |
| ImageXY | `train.py:370-423` 直接官方 train/val 训练/选模；实际旧 checkpoint 为离线 T13 灰度64，而不是本轮 causal T5；官方 val 涵盖两个75-scene留出集 | 首轮主比较不使用其旧权重或精度；可复用已核验的结构经验 |
| 历史 stage1_test | archive-legacy 的 `server_results/accelerated_20260719_143555/stage1_test_report_seed42_preliminary/stage1_report.md` 已记录75场景评价；7月20日 quick_existing_eval 也使用75场景 | 不能称从未查看过的全新测试集；这些报告是旧 V2/KD 等方法，不是当前 V3 direct 权重的重测证据 |
| final_system_test | 上述历史报告明示当时保持 sealed；但 ImageXY 曾用整个官方 val 选模，因此不能对所有历史分支作从未曝光保证 | 本轮 target 不生成、没有新评价；不使用 ImageXY 权重，从本轮模型冻结后再明确安排最终评价 |

只读检查没有发现与当前 V3 direct checkpoint 精确绑定的历史 stage1 预测工件；不将流水线中存在测试命令等同于测试实际完成。

## 工件与复现

工件位于 `/mnt/sdb2/Workspace/zhangyu/01_speedest_vision_artifacts/v5/data/nuscenes_proxy_v1/`：`dataset.json`、`audit.json`、835 份 `sequences/*.json`。不存入 Git。每份 sequence 的 SHA-256 已进入主清单；主清单还固定 label 契约/实现哈希、split 哈希。audit 保存六份原始元数据表的 SHA-256。

- label 契约 SHA-256：`19e5bb5e9d604836de5ca6e5aeaec4056602fe4364c0136ebb6616bbc0c3a6c6`。
- 当前 dataset.json SHA-256：`a74f7963f7c9e0b529fefef75f5be1c191a7a2fccbddd3678b82f825ee0de32a`。
- labels.py SHA-256：`381a758f59e6f59e2ecd8c97b39870eaf158a454fe9dd1a941bede10223148fe`。

从 PureVision_V5 执行（重新准备必须选新的输出目录，现有输出不会覆盖）：

```bash
/home/zhangyu/.conda/envs/speedest/bin/python -B prepare_data.py \
  --dataroot /mnt/sdb2/Workspace/zhangyu/dataset/nuscenes \
  --split-manifest ../PureVision_V3/logs/v2_split_manifest.json \
  --output /mnt/sdb2/Workspace/zhangyu/01_speedest_vision_artifacts/v5/data/nuscenes_proxy_v1
/home/zhangyu/.conda/envs/speedest/bin/python -B -m pytest -q tests/test_data.py
```

数据测试实际结果：**8 passed**，含小型完整导出、五帧因果边界、监督字段隔离、holdout 空标签和防覆盖检查。数据准备阶段未运行任何模型评价。
