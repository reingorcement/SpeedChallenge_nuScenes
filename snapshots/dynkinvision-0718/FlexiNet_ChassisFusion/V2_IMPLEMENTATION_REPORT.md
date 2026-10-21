# V2-nuScenes 视觉几何与弱运动学改造实施报告

## 1. 实施范围

当前 V2 仅使用 nuScenes 完成两个分支：

1. 视觉几何主线与有界视觉残差；
2. 严格因果的弱运动学有界修正。

本阶段没有接入 CarSim、动力学学生或可学习滑移分类头。旧 `FlexiNetChassisFusion`、旧训练入口和物理基线均保留，V2 使用独立的 `V2NuScenesVelocityModel` 与 `train_v2.py`。

模型默认输出 nuScenes `ego_pose` 参考点的 `ego_reference` 速度，不宣称车辆质心速度或完整动力学估计。

## 2. 当前算法

视觉分支为：

$$
\mathbf v_{\mathrm{visual}}
=
\begin{bmatrix}
s_w v_{x,\mathrm{geo}} \\
v_{y,\mathrm{geo}}
\end{bmatrix}
+g_{\mathrm{img}}\Delta\mathbf v_{\mathrm{img}}.
$$

弱运动学分支在视觉输出上进行第二次有界修正：

$$
\hat{\mathbf v}
=
\mathbf v_{\mathrm{visual}}
+g_{\mathrm{kin}}\Delta\mathbf v_{\mathrm{kin}}.
$$

其中：

- RAFT 光流先按可用 yaw rate 去除旋转分量，再由道路平面、相机标定和 Huber IRLS 求解平移速度；
- 四轮绝对速度均值只用于 `s_w\in[0.85,1.15]` 和几何失效时的 `[v_w,0]` 回退；
- FiLM 只接收几何协方差、条件数、残差和有效性四个诊断量，原始 CAN 不进入视觉残差头；
- `\Delta\mathbf v_{\mathrm{img}}` 限幅为 `[3.0,1.5] m/s`；
- `\Delta\mathbf v_{\mathrm{kin}}` 限幅为 `[1.0,0.5] m/s`，其门控为字段覆盖率与预测不确定性的单调函数。

模型支持 `geometry_only`、`visual` 和 `visual_kinematic` 三种模式。结构化输出包含三阶段速度、视觉和运动学残差、门控、运动学对数方差、几何条件数/残差/支持点、几何有效性和 `fallback_used`。

## 3. 数据契约

### 3.1 双分辨率

| 输入 | 形状 | 用途 |
|---|---|---|
| `images` | `[B,5,3,72,128]` | RGB FlexiNet 视觉特征 |
| `flow_v2` | `[B,2,288,512]` | 最新图像对的几何光流 |
| `flow_confidence` | `[B,1,288,512]` | 前后向一致性置信度 |
| `intrinsics` | `[B,3,3]` | 只缩放到几何分辨率 |
| `road_mask` | `[B,1,288,512]` | 当前静态道路 ROI |

灰度 FlexiNet 预训练首层按 `repeat(3)/3` 迁移到 RGB；加载时打印实际加载 tensor 数、参数数、适配 tensor 数和缺失项数量。

### 3.2 弱运动学序列

每个样本使用当前时刻及之前 `1 s / 20 Hz / K=21` 的查询点，任何查询点都不晚于标签时刻。每步包含 12 个值：

```text
yaw_rate, steering_command, steering_feedback,
longitudinal_accel, lateral_accel, throttle, brake,
wheel_residual_fl, wheel_residual_fr,
wheel_residual_rl, wheel_residual_rr,
wheel_speed_relative_std
```

每个值拼接逐字段有效掩码和归一化真实报文年龄，形成：

```text
kinematic_values : [B,21,12]
kinematic_valid  : [B,21,12]
kinematic_age    : [B,21,12]
GRU input        : [B,21,36]
```

缺失字段使用 `value=0`、`valid=False`、`age=1`。pose 派生速度、加速度和旋转率不能进入运动学输入。四轮共同缩放不会改变归一化轮速残差，绝对轮速均值保存在独立的确定性尺度/回退通道。

## 4. 代码改动

- `dataset/nuscenes_chassis.py`：增加真实因果 CAN 报文年龄查询，并保留旧接口。
- `dataset/v2_schema.py`：增加双分辨率 V2 适配器、21 步运动学序列、train-only 标准化、逐字段年龄阈值、序列一致光度增强和独立轮速通道。
- `model/v2/kinematics.py`：增加 `36 -> 64 -> 64 -> GRU(128)` 弱运动学编码器、异方差输出和单调门控。
- `model/v2/model.py`：增加三种模式、两级有界融合、轮速回退、RGB 权重迁移和分阶段冻结接口；几何求解在 AMP 下强制使用 FP32。
- `model/v2/losses.py`：增加相对冻结视觉输出的运动学残差监督、异方差 NLL 和运动学残差正则。
- `model/v2/types.py`：扩展结构化输出，不改变旧 `FlexiNetChassisFusion` 接口。
- `train_v2.py`：实现冻结 BatchNorm、零修正 epoch 0、分组 AdamW、warmup-cosine、有效 batch、早停、完整续训、约束式 checkpoint 选择及运行元数据记录。
- `scripts/audit_v2_data.py`：审计 CAN 场景覆盖、字段有限率及报文间隔。
- `scripts/prepare_v2_training_stats.py`：只用 nuScenes train split 拟合12维运动学均值/标准差、字段年龄阈值和场景采样权重。
- `scripts/evaluate_v2_counterfactual.py`：分别打乱图像、光流和 CAN，检查模态依赖。
- `scripts/compare_v2_stages.py` 与 `compare_v2_multiseed.py`：执行单 seed 和三 seed 的 EPE/分组 `vy` 声明门槛。

## 5. 分阶段训练参数

| 阶段 | 配置 | epoch | 可训练模块 | 学习率 |
|---|---|---:|---|---|
| 几何审计 | `velocity_xy_v2_geometry.yaml` | 0 | 无 | 无 |
| 视觉训练 | `velocity_xy_v2_visual.yaml` | 最多20 | 前5 epoch仅视觉头；之后加 `sfe/mfe/dig` | head `3e-4`，backbone `1e-5` |
| 运动学预训练 | `velocity_xy_v2_kinematic.yaml` | 最多15 | 仅弱运动学编码器，视觉模块保持 eval | `3e-4` |
| 联合微调 | `velocity_xy_v2_visual_kinematic.yaml` | 最多10 | 运动学、视觉残差头、视觉后层 | head/kin `1e-4`，backbone `5e-6` |

共同默认参数为 AdamW、weight decay `1e-4`、有效 batch size `8`、5% warmup、cosine decay 至0.1倍学习率、AMP、seed `42` 和梯度裁剪 `2.0`。bias 与 BatchNorm 不做 weight decay。联合阶段使用运动学模态 dropout `0.2` 和轮速尺度扰动 `±5%`。

服务器先用最大显存模式 smoke：batch `8` 失败时，只有日志明确包含 CUDA OOM 才回退 micro batch `4`，并用两步梯度累积保持有效 batch `8`；非 OOM 错误直接停止。visual/kinematic/joint 的 early-stopping patience 分别为 `5/4/4`。

`configs/velocity_xy_v2_nuscenes.yaml` 是联合阶段的兼容别名；正式流水线应使用四个阶段配置，不应直接跳到该别名。

## 6. 损失与评估

训练默认使用 `sigma_vx=5.0`、`sigma_vy=1.0`、`beta_weight=0.1`、`beta_min_speed=2.0 m/s`。确定性 geometry loss 在可训练阶段设为0，只在 geometry audit 中设为1并独立报告。其余损失包含速度 Smooth L1、真值侧偏角、视觉残差正则、运动学异方差 NLL 和运动学残差正则。运动学目标为：

$$
\mathbf y_{\mathrm{kin}}
=
\mathbf v_{\mathrm{target}}
-\operatorname{stopgrad}(\mathbf v_{\mathrm{visual}}).
$$

评估分别输出 geometry、visual 和 final 的 `MAE/RMSE vx/vy` 与 EPE，并输出 beta、geometry valid rate、条件数、几何残差、支持点、fallback rate、运动学门控和修正量。分组包括直行、转弯、高侧向速度和轮速不一致。

几何正式审计会在以下任一条件不满足时终止流水线：

- geometry valid rate `>= 60%`；
- 中位条件数 `< 1e5`；
- 直行样本纵向速度符号一致率 `>= 95%`。

visual checkpoint 按总体 EPE 选择。kinematic/joint checkpoint 先要求 EPE 不高于 visual 基线的 `102%`，再最小化转弯与高侧向 `vy MAE` 的均值。三 seed 最终声明要求平均 EPE 满足2%门槛，且至少两个 seed 的转弯或高侧向 `vy` 改善。

## 7. 服务器执行

在 `FlexiNet_ChassisFusion` 目录执行：

```bash
export NUSCENES_ROOT=/media/ps/AC78E69278E65B1A/zy/nuscenes
export NUSCENES_CAN_ROOT=${NUSCENES_ROOT}/can_bus
export V2_FLOW_ROOT=${NUSCENES_ROOT}/precomputed_flow_v2/CAM_FRONT
bash scripts/run_v2_server_pipeline.sh
```

脚本依次执行数据审计、RAFT 光流、train-only 运动学统计、最大显存 smoke、geometry audit、seed 42 四阶段训练、反事实检查、seed `3407/2026` 配对复现和三 seed 验收。每次运行保存解析后配置、代码提交、seed、micro/effective batch、累积步数、学习率和裁剪比例。

K-Dense `experimental-design` 与 `statistical-analysis` 已固定到提交 `3f825ca` 安装为 Codex skills；它们不进入 `speedXY_py310` 或服务器训练依赖。

本地不运行 nuScenes 预处理、RAFT、smoke 或正式训练。

## 8. 未解决风险

1. 当前只有 yaw rate，缺少可部署的完整三轴 IMU；俯仰/横滚旋转流仍可能污染平移解。
2. 相机高度 `1.51 m` 是待审计名义值，尺度误差会直接传递到速度。
3. 道路掩码仍是下半幅静态 ROI，动态车辆、路缘、坡度和非平面道路会产生系统误差。
4. 没有验证过的传感器到 CG 杆臂，结果只能解释为 `ego_reference`。
5. 真实报文年龄是记录时间戳差，不等于传感器采集到记录链路的端到端延迟。
6. 当前仅有即时轮速回退，没有跨样本的最近有效视觉状态传播。
7. 几何协方差和运动学不确定性尚未完成概率校准。
8. nuScenes 单数据集结果不能证明跨车辆、轮胎、相机或附着条件泛化。

## 9. 本地验证

本地验证只覆盖代码编译、合成小张量和回归测试。nuScenes 数据与服务器训练结果必须另行记录，不能用本地单元测试替代。

```text
92 passed
```

测试新增覆盖冻结 BatchNorm、解冻后统计更新、零初始化、train-only 标准化、序列一致增强、字段年龄解析、分组 weight decay、warmup-cosine、有效 batch、约束选模、完整续训和100步固定小批量过拟合。nuScenes 数据统计、显存 smoke 和正式训练仍只在服务器执行。
