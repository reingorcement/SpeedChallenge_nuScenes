# PureVision V3：纯视觉纵、侧向速度估计

将整个 `PureVision_V3` 目录复制到 Linux 服务器，在目录内运行下列命令。

模型使用前视 RGB 序列、RAFT 光流、帧间隔、相机标定和地面几何先验，输出车体坐标系下的纵向速度 `vx` 和左向侧向速度 `vy`，单位为 m/s。融合对象是 RGB 直接估计与光流几何估计；模型输入不含 CAN/底盘信号。nuScenes ego pose 用于生成监督标签。

## 1. 环境

优先激活服务器上原来运行该项目的 Python 环境。需要 Python 3.10 或以上、配套的 PyTorch/torchvision 和可用的 CUDA。

```bash
cd /path/to/PureVision_V3
python -m pip install -r requirements.txt
bash server.sh check
```

`check` 检查依赖、CUDA、随包权重的加载兼容性，并运行 V3 测试用例，无需 nuScenes 数据。依赖版本取自本地验证环境；PyTorch/torchvision 由服务器 CUDA 环境提供。本地验证版本是 `torch 2.10.0+cu128`、`torchvision 0.25.0+cu128`。

## 2. 设置数据路径

修改 `server.env` 中的 `NUSCENES_ROOT`。如果已有缓存位于其他目录，也修改 `V3_FLOW_ROOT` 和 `V3_FRAME_ROOT`。这三个路径也可以在 shell 中通过 `export` 设置。

```text
/data/nuscenes/
  v1.0-trainval/
  samples/
  sweeps/
  precomputed_flow_v2/CAM_FRONT/                 # RAFT large，288×512
  precomputed_frames/CAM_FRONT_72x128_rgb/       # RGB 帧缓存及标签
```

随包保留原始 `logs/v2_split_manifest.json` 数据划分。训练使用 `train_core`，开发验证使用 `dev`，独立测试使用 `stage1_test`。

已有兼容缓存时直接进入测试。需要生成缓存时运行：

```bash
bash server.sh prepare
```

预处理使用 GPU 计算 RAFT 光流，首次使用时 torchvision 会下载 RAFT 权重。离线服务器需提前准备对应 torch 权重缓存。图像和光流缓存保存在上述数据目录中。

## 3. 使用已有权重测试

已经带上 seed 42 的 RGB 和融合模型 `best.pth`、`last.pth`，默认使用融合模型的最佳权重。先跑两个 batch：

```bash
bash server.sh test-smoke
```

完整独立测试：

```bash
bash server.sh test
```

结果位于 `outputs/v3_test_<时间>/`，包含三组预测、误差指标、侧偏角指标及汇总：

| 组别 | 使用的估计方式 |
|---|---|
| `rgb_only` | 默认使用融合 checkpoint 内的 RGB 直接估计分支 |
| `flow_geometry_only` | 光流和静态标定驱动的几何求解 |
| `fusion` | 完整 V3 视觉融合模型 |

这三个结果用于比较已有模型的分支与完整推理行为。若希望 RGB 组使用随包的独立 RGB 预训练权重：

```bash
bash server.sh test --rgb_checkpoint checkpoints/pretrained/direct_seed_42/best.pth
```

使用其他融合权重可设置 `V3_CHECKPOINT`。设置 `V3_RUN_TAG` 可指定输出目录名；测试要求输出目录尚未存在，以免覆盖结果。

## 4. 从头训练

```bash
bash server.sh train
```

默认依次执行 seed 42、3407、2026 的 RGB 预训练、视觉融合训练、独立测试及统计分析。只跑 seed 42：

```bash
V3_SEEDS=42 V3_RUN_TAG=v3_seed42_server bash server.sh train
```

使用相同 `V3_RUN_TAG` 再次运行会续训未完成阶段，跳过达到训练轮数或已早停的阶段。新训练权重保存在 `checkpoints/<run-tag>/`，训练日志在 `logs/<run-tag>.log`。批量大小、worker 数量在 `server.env` 中设置。

后台执行：

```bash
mkdir -p logs
nohup env V3_RUN_TAG=v3_server_run bash server.sh train > logs/server-launch.log 2>&1 &
tail -f logs/server-launch.log
```

如需手动续训随包的融合模型，使用同结构配置和显式数据路径：

```bash
python train_v3.py --config configs/velocity_xy_v3_visual_only.yaml \
  --stage fusion --resume checkpoints/pretrained/fusion_seed_42/last.pth \
  --dataroot /data/nuscenes \
  --flow_root /data/nuscenes/precomputed_flow_v2/CAM_FRONT \
  --frame_cache_root /data/nuscenes/precomputed_frames/CAM_FRONT_72x128_rgb \
  --save_dir checkpoints/resumed_fusion --output_dir outputs/resumed_fusion
```

该随包 `last.pth` 已记录早停，通常直接使用 `best.pth` 测试即可。

## 5. 目录与整理范围

| 路径 | 用途 |
|---|---|
| `server.sh`、`server.env` | Linux 启动入口及路径、运行参数 |
| `train_v3.py` | V3 训练入口 |
| `model/v3/` | 纯视觉模型、几何求解和损失 |
| `dataset/` | nuScenes 图像、标签、划分及 V3 输入适配 |
| `scripts/` | 预处理、测试、统计分析、运行检查 |
| `configs/` | V3 正式配置及小规模检查配置 |
| `tests/` | 原有 V3 回归测试和冻结权重测试 |
| `checkpoints/pretrained/` | 已有 seed 42 权重 |
| `logs/v2_split_manifest.json` | 原实验的数据划分 |
| `docs/VALIDATION.md` | 本次整理的验证记录 |
| `MANIFEST.sha256` | 交付文件校验值 |

V3 复用了部分名字含 `v2`、`chassis` 的基础工具，这些必要依赖已包含在本目录中。`model/v2/` 只保留被 V3 引用的几何工具，包入口不再导入 V2 网络。可从任意位置运行本独立目录，无需旁边的其他项目。

包内包含当前 V3 代码及已有权重，数据集、图像/光流缓存由服务器提供。完整 nuScenes 训练和数据推理需要在服务器上执行；本地检查范围见验证记录。
