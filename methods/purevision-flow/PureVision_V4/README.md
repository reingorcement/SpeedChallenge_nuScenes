# PureVision V4

纯视觉纵、侧向速度估计的隔离整合版。输入为 RGB、时间戳、固定相机标定和名义相机高度；推理不读取 IMU、CAN、真值速度或真实姿态。

V4 以有限帧间位姿求解和受约束视觉侧倾法向作为几何分支，保留 V3 的 RGB 特征及直接回归初始化，提供新的融合训练、推理、nuScenes 标签转换和统一评价入口。

**默认可直接运行 `F_RC` / `F_R`。新融合头需要通过 `train.py` 训练后使用。随包的 V3 权重是冻结特征提取与初始化权重，不是已经训练好的 V4 融合权重。**

## 隔离规则

- 本目录自包含代码、配置和权重；运行时无需导入相邻旧工程。
- 使用 `purevision_v4` 包名，原模型位于内部 `_legacy` 命名空间，不占用全局 `model`、`dataset`、`roll_velocity`。
- 原两个工程保持原样，权重为独立文件副本，没有软链接或硬链接。
- 输出只允许写入本目录的 `runs/`、`training/`、`prepared/`、`verification/`。输入目录与输出目录不能相互包含。
- 原工程 1252 个文件的大小及 SHA-256 记录在 `verification/source_baseline.json`。`tools/check_isolation.py` 只读核验原文件和新增文件，不修复、不覆盖。

## 环境与快速运行

当前机器已有可用环境，无需修改旧环境。以下命令均从本目录执行：

```bash
cd /mnt/sdb2/Workspace/zhangyu/01_speedest_vision/PureVision_V4
V4_PYTHON=/home/zhangyu/.conda/envs/speedest/bin/python

"$V4_PYTHON" -B check.py
"$V4_PYTHON" -B -m pytest -q

# 读取原样例，输出仅进入 V4；首次测试 8 帧。
"$V4_PYTHON" -B run.py \
  --input ../补偿速度估计/data/HW_DEV_SINE40_A12 \
  --output runs/a12_smoke --frames 8 --device cuda:0 \
  --methods F_RC F_R --save-features
```

去掉 `--frames 8` 可运行整段。已有输出需要用 `--resume` 恢复完全相同的任务；输入、代码、配置、权重或环境发生变化时应选择新输出目录。恢复会重放法向历史，复用校验后的原始光流。确认进程已结束后，才可手动移除被强制终止进程遗留的 `.running`。

所有权重从本目录加载，不在推理时下载。另一台机器应准备匹配的 PyTorch/torchvision，并安装 `requirements.txt`。代码按源目录运行；不要将输出指向旧工程或数据集。

## 方法与输出

| 方法 | 行为 |
|---|---|
| `F_RC` | 有限位姿求解，视觉侧倾法向，俯仰与高度固定；默认主方法 |
| `F_R` | 同一有限求解器，固定法向；默认对照 |
| `F_B` | 完整视觉法向；实验对照，不保证优于 F_RC |
| `fusion` | 新几何 + RGB 直接回归 + 新修正/门控/不确定度头；需要训练后的 V4 fusion 权重 |
| `rgb_only` | 训练后的 V4 RGB 直接分支；需要 V4 direct 或 fusion 权重 |
| `legacy_v3` | 原 V3 融合网络对照；保留旧标签口径标记 |
| `rgb_legacy` | 原融合 checkpoint 中的 RGB 直接分支；保留旧标签口径标记 |

`legacy_v3` 使用原 V3 的内参缩放及固定法向，输入光流来自当前运行的 RAFT 预处理；它不是对任意历史光流缓存逐位复现的承诺。旧模型的 yaw 平面训练目标与 V4 体轴标签不同，因此不能将旧模型输出直接当作 V4 同口径精度对照。旧项目中的独立 RGB 预训练权重也已复制到 `weights/v3_direct_seed42.pth`，默认特征链仍使用冻结的融合 checkpoint。

- `speeds.csv`：按方法输出速度、有效性、测量时刻、法向状态和耗时。速度 m/s，x 前、y 左、z 上；参考点由标定的 `reference_point` 指定。
- `pairs.csv`：每个源/目标帧对的法向、条件数、残差和完整三维体轴 twist；无效速度留空。
- `normals.json`：源帧法向、`update/hold/fixed_fallback`、原因和时间。
- `run.json`：输入/代码/权重指纹、坐标定义、环境、覆盖率、总耗时和结果哈希。
- `features/`：启用 `--save-features` 后保存冻结 RGB 描述子和 F_RC 几何诊断，供训练读取；没有真值。

前四帧为预热。几何完全失效时输出无效和空速度；训练后的融合分支在几何无效时显式退回 RGB。`geometry_valid` 与最终 `valid` 含义不同。

全幅无纹理输入返回 `textureless_input`，不会用 RGB 先验填成有效速度。路面候选还检查内部亮度变化，拒绝无纹理平面提供的伪光流支持。默认门槛为 8 位灰度标准差 1.0；该检查覆盖黑屏、白屏等明显失效输入，不代表已经解决所有低纹理或运动模糊情况。

几何主输出取最近三对有效速度的分量均值。`time_s` 是发布帧时刻，`effective_time_s` 是有效帧对中点均值；20 Hz、三对有效时相差 75 ms。新融合头以发布帧速度为训练目标，并读取几何测量延迟；这不等于已证明消除全部动态延迟。评价固定按发布帧时刻进行，不搜索有利时间偏移。

`inference_ms` 是对应方法计算时间；融合行包含其依赖的 F_RC 求解时间。`shared_ms` 包括图像、光流、法向和冻结 RGB 特征，`frame_total_ms` 是本帧全部所选方法及缓存写入的总时间。CUDA 计时显式同步。

## 自己的数据

准备 `calibration.json`、`frames.json` 和 RGB 图像。旧补偿版的标定与 `solver_time` 字段可直接读取，固定坐标转换会被规范化。新标定推荐：

```json
{
  "image_size": [800, 600],
  "K": [[400, 0, 400], [0, 400, 300], [0, 0, 1]],
  "height_m": 1.4,
  "plane_normal_camera": [0, -1, 0],
  "rotation_body_from_camera": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
  "camera_lever_body_m": [0.8, 0, 0.9],
  "reference_point": "static_cg"
}
```

这里的数值仅展示格式。`camera_lever_body_m` 是从输出参考点指向相机的杆臂，在车体轴中表达。相机光学轴为 x 右、y 下、z 前；平面满足 `nᵀX + height_m = 0`。图像须去畸变，K 对应原图。所有分辨率变换统一使用像素中心约定。

```json
[
  {"image_id": 0, "file": "rgb/000000.png", "time_s": 0.0},
  {"image_id": 1, "file": "rgb/000001.png", "time_s": 0.05}
]
```

至少五帧，ID/时间严格递增。图像路径默认相对输入目录；通过显式 `--image-root` 可以只读引用外部数据集，路径不能越出该根目录。输入中的标签字段不会进入模型。

RAFT 默认按原图比例计算高度，取 8 像素倍数且至少 128 像素，宽度取配置值；随后将光流及其分量转换到几何分辨率。不直接沿用两套旧缓存。缓存记录预处理版本、实际尺寸和权重。相机高度提供米制尺度，不能随意套用样例值。

## nuScenes 与训练

新的标签生成器使用完整三维位移导数和当前帧姿态，输出 nuScenes **ego origin** 的体轴速度。它不将 ego origin 称为质心。位姿只用于离线标签，生成的推理清单只有图像、时间和固定标定。原 `labels.json` 不覆盖、不自动沿用。

```bash
# 先用 --limit 1 检查每个分区的一段，再生成完整清单。
"$V4_PYTHON" -B export_nuscenes.py \
  --dataroot /mnt/sdb2/Workspace/zhangyu/dataset/nuscenes \
  --output prepared/nuscenes_v4 --camera-height 1.51 \
  --partitions train_core dev --limit 1

# 顺序生成每段法向与特征；原始图像不复制、不修改。
"$V4_PYTHON" -B run_batch.py --manifest prepared/nuscenes_v4/dataset.json \
  --output runs/nuscenes_features --save-features --evaluate --device cuda:0

# 第一阶段固定 backbone、描述子投影、可靠性网络和有限求解器。
"$V4_PYTHON" -B train.py --manifest runs/nuscenes_features/training_manifest.json \
  --output training/direct_seed42 --stage direct --device cuda:0

"$V4_PYTHON" -B train.py --manifest runs/nuscenes_features/training_manifest.json \
  --output training/fusion_seed42 --stage fusion \
  --init training/direct_seed42/best.pth --device cuda:0
```

训练支持批量、独立 dev 选优、早停、梯度检查、`best.pth/last.pth` 和恢复。续训写入新的 V4 训练目录，例如 `--resume training/fusion_seed42/last.pth --output training/fusion_resumed --epochs 30`。阶段变更使用 `--init`，同阶段续训使用 `--resume`。

目前训练的是冻结特征上的直接回归、修正、门控和不确定度头；OpenCV/SciPy 几何明确不可微，不声称对可靠性网络完成端到端几何反传。后续若微调特征提取器，必须重建特征缓存并更新权重契约。

自有数据可提供 `training_manifest_v1` 清单：

```json
{
  "schema": "training_manifest_v1",
  "train": [{"features": "../runs/train_case/features/manifest.json", "labels": "/data/labels/train.csv"}],
  "dev": [{"features": "../runs/dev_case/features/manifest.json", "labels": "/data/labels/dev.csv"}]
}
```

标签 CSV 为 `time,vx,vy`，可附 `vz,roll`，其中 roll 为 rad。每个 CSV 旁必须有同名 `.meta.json`：

```json
{"version":"body_reference_v1","axes":"forward_left_up","units":"m/s","reference_point":"static_cg","time":"output_frame"}
```

只应对确实符合该定义的标签声明这个契约；给旧 yaw 标签换一个名称不能完成三维转换。训练检查特征指纹、标签契约、时间匹配，以及 train/dev 场景和图像重叠，不能用同一序列相邻窗口充当独立验证集。

训练完成后：

```bash
"$V4_PYTHON" -B run.py --input /data/my_sequence --output runs/my_v4 \
  --methods F_RC F_R fusion rgb_only \
  --fusion-checkpoint training/fusion_seed42/best.pth --device cuda:0

"$V4_PYTHON" -B evaluate.py --predictions runs/my_v4/speeds.csv \
  --truth /data/labels/my_sequence.csv --output runs/my_v4/evaluation
```

推理会拒绝预处理、代码特征契约或参考点不匹配的融合权重。nuScenes ego-origin 模型与 static-CG 标签不能未经处理混用。

## 验证与适用范围

实际执行记录见 [验证记录](docs/VALIDATION.md)。测试覆盖几何运动、异常回退、时间/坐标契约、批处理、训练重载和代码隔离。`tools/check_isolation.py` 可重新核验原目录。

F_RC 的默认法向约束保留原已验证的相机坐标策略，固定俯仰和高度。有限位姿中联合估计三轴旋转，并只进行一次旋转处理和杆臂换算；这不表示已解决路面横坡、动态俯仰、高度变化或任意相机安装角的法向估计。全法向 F_B、跨相机泛化和实时性能仍应在目标数据上独立评价。

本次交付没有将小样本训练或旧开发集结果当成新模型的独立精度验收。
