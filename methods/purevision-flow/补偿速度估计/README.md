# 侧倾补偿纯视觉速度估计：服务器运行版

推荐运行 **F_RC：有限帧间平面几何＋受约束视觉侧倾法向**。本包从 `02_透明底盘/02_Roll_VisualVelocity_AB_v2` 提取，采用第 20 阶段的像素中心修复接口，以及第 16 阶段的冻结几何求解器、网络和权重。整理日期：2026-10-20。

推理只使用 **RGB、时间戳和固定相机标定**。不需要 CarSim、CARLA、nuScenes SDK、IMU、轮速或真实侧倾角，也不需要重新训练。附带 RAFT 和可靠性网络权重，运行时不下载模型。

## 服务器上直接运行

建议新建 Python 3.10 环境，使用 NVIDIA GPU。已提供 Linux 脚本，路径相对包目录解析。将整个目录上传；压缩包解压后顶层目录名为 `roll_velocity_server`。

```bash
cd /your/path/roll_velocity_server

# 新建环境，安装固定版本，并执行 GPU 检查
bash scripts/setup_linux.sh

# 验证文件完整性和数学/输入测试
.venv/bin/python verify_package.py
.venv/bin/python -B -m unittest discover -s tests -v

# 8 张图的 RGB→光流→侧倾补偿→速度测试
bash scripts/smoke.sh

# 四段完整开发数据，共 960 张图；同时跑四种方法并生成评价图表
bash scripts/run_all.sh
```

安装默认 `torch==2.10.0`、`torchvision==0.25.0`、CUDA 12.8 wheel，与本次实际验证环境一致。这一组合见 [PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/)。已有可用环境时，可跳过安装，直接执行 `python check_environment.py`，并把后续命令中的 `.venv/bin/python` 换成自己的解释器。不要直接使用原工作区包含 `file:///home/...` 路径的整机依赖导出。

后台运行示例：

```bash
mkdir -p runs
nohup bash scripts/run_all.sh runs/development > runs/development.log 2>&1 &
tail -f runs/development.log
```

单独运行任意一段：

```bash
.venv/bin/python -u -B run.py \
  --input data/HW_DEV_SINE40_A12 \
  --output runs/a12 \
  --methods F_RC F_R
```

默认两组为 F_RC 主方法和 F_R 固定法向对照。完整消融使用：

```bash
.venv/bin/python -u -B run.py \
  --input data/HW_DEV_SINE40_A12 \
  --output runs/a12_ablation \
  --methods F_RC F_R F_B V3_matched_R
```

`--resume` 只恢复同一输出目录中输入、代码、参数、权重和环境均一致的运行。已有完整结果会校验后直接返回；中断运行复用已校验的 RAFT 光流，重新顺序计算法向状态和速度。更改方法或参数请换输出目录。进程被强制终止时可能留下 `.running`，确认对应进程已退出后再删除该锁文件。每个输出目录只允许一个进程写入。

`--device cuda:1` 可指定 GPU；`--device cpu` 可用于小规模检查，但 RAFT 会很慢。没有针对实时处理验收。完整运行会生成约 1.7 GiB 光流缓存，另需数据、依赖和结果空间。

## 输出看哪里

| 文件 | 内容 |
|---|---|
| `runs/a12/speeds.csv` | 主速度表，按 `method=F_RC` 取 `vx_mps` / `vy_mps` |
| `runs/a12/pairs.csv` | 每个窗口的三对运动、实际使用法向、求解有效性和残差 |
| `runs/a12/cache/region/normals.json` | 每个源帧的视觉法向、update/hold/fixed_fallback 及原因 |
| `runs/a12/cache/region/summary.json` | 法向更新、保持、回退计数 |
| `runs/a12/run.json` | 完整运行指纹、环境、输入哈希、方法、耗时和覆盖率 |
| `runs/development/macro_metrics.csv` | 批量评价：各工况等权 MAE、覆盖率及最差工况 P95 |
| `runs/development/<case>/evaluation/velocity.png` | 纵横向速度曲线 |

速度单位 m/s，`vx` 向前、`vy` 向左、`vz` 向上。主输出是按固定标定表示、换算到静态参考质心的车身体轴速度。`speed_mps=sqrt(vx²+vy²)`。相机光学坐标为 x 右、y 下、z 前。

前四帧为预热，`valid=False`、`reason=warmup`。求解失败时速度留空，不填零，不沿用旧速度假装有效。`valid_pairs` 表示参与聚合的 1–3 对；全部失败时整窗口无效。F_RC 的正常估计失败时，法向短时保持，随后回退固定标定；其速度求解仍可能有效，需结合 `normal_status` 阅读。

输出 i 使用图像 i−4…i、光流对 i−3→i−2、i−2→i−1、i−1→i，各对使用自己的源帧法向。主输出对有效帧对速度做分量均值。`time_s` 为输出帧时刻；`effective_time_s` 记录参与聚合帧对中点的平均时间。在 20 Hz、三对均有效时，后者约早 75 ms。评价始终按输出帧时刻对齐，不搜索最优时间偏移。`inference_ms` 仅是该方法求解阶段耗时，完整耗时见 `run.json`。

## 接入自己的图像

准备以下目录：

```text
my_sequence/
  calibration.json
  frames.json
  rgb/
    000000.png
    000001.png
    ...
```

`frames.json` 示例：

```json
[
  {"image_id": 0, "file": "rgb/000000.png", "solver_time": 0.0},
  {"image_id": 1, "file": "rgb/000001.png", "solver_time": 0.05}
]
```

至少五张连续图像，时间单位秒，严格递增。`solver_time` 在真实数据上就是采集时间戳的字段名，不要求仿真器。文件路径须在输入目录内。图像须已去畸变；用去畸变后的 K，保持标定分辨率和图像尺寸一致。

固定标定字段：

| 字段 | 定义 |
|---|---|
| `image_size` | `[原图宽, 原图高]`，像素 |
| `K` | 原图 3×3 针孔内参，零 skew；代码自动按像素中心缩放 |
| `height_m` | 相机到名义地面平面的垂直距离，正数、米；提供米制尺度 |
| `plane_normal_camera` | 相机坐标单位法向，满足 `nᵀX + height_m = 0` |
| `rotation_ego_from_camera` | 从相机光学轴到固定参考车辆坐标的旋转矩阵 |
| `translation_ego_from_camera` | 从参考质心指向相机的杆臂，在该固定车辆坐标表达，米 |
| `body_mount_rotation` | 原工程约定下，从相机前/左/上基底到车身体轴的固定旋转 |

原工程使用 `C_body = body_mount_rotation @ [[0,0,1],[-1,0,0],[0,-1,0]]`。最终从参考车辆坐标转到车身体轴的矩阵为 `C_body @ rotation_ego_from_camera.T`。若自己的外参已经直接从光学轴转到车身体轴，可令 `rotation_ego_from_camera=C_body`、`body_mount_rotation=C_body @ optical.T`，杆臂也用车身体轴表达。不要把动态姿态写进固定标定。

可参考 `data/HW_DEV_SINE40_A12/calibration.json` 的格式，实际高度、杆臂和外参须使用自己的标定值。单目速度尺度依赖高度；直接沿用示例的 1.4 m 会将高度误差带入速度。

```bash
.venv/bin/python -u -B run.py --input /data/my_sequence --output runs/my_sequence
```

按片段测试可加 `--start-frame 100 --frames 20`。这会重置法向历史并重新预热，不能要求与全程结果逐点相同。长序列的图像与流按滑动窗口加载，内存不随全部光流数量线性增长。

## 离线评价

推理入口没有标签参数。评价程序在推理结束后独立读取真值：

```bash
.venv/bin/python evaluate.py \
  --predictions runs/a12/speeds.csv \
  --truth evaluation/labels/HW_DEV_SINE40_A12/truth_20hz.csv \
  --output runs/a12/evaluation
```

标签最少包含 `time,vx,vy`，单位秒、m/s；可选 `roll` 单位 rad，用于划分侧倾子集。按时间戳在 1e-6 秒内匹配，不按行号连接。前四帧预热剔除；失效窗口保留在覆盖率分母中。分别报告各方法自身有效集和共同有效集，避免只报告有利帧。

目标子集：`|roll|≥1° 且 |vy|≥0.2 m/s`；安静子集：`|roll|<1° 且 |vy|<0.05 m/s`。这些分组只在评价阶段使用真实侧倾角。

## 为什么选择这版

详细依据见 [版本选择说明](docs/版本选择说明.md)。同一修复接口、同一开发道路的三段侧倾子集，原记录给出：

| 方法 | vx MAE (m/s) | vy MAE (m/s) |
|---|---:|---:|
| V3_matched_R：原微分求解器，共同路面支持 | 2.4621 | 0.1419 |
| F_R：有限求解，固定法向 | 0.0419 | 0.1264 |
| **F_RC：有限求解，视觉侧倾＋固定俯仰** | **0.0319** | **0.0683** |
| F_B：有限求解，完整视觉法向 | 0.1037 | 0.0733 |

上表是历史指标。**本次从 960 张 RGB 重新计算后，F_R→F_RC 的侧倾子集增益为纵向 24.7%、横向 47.0%**，对应 MAE 从 0.04299/0.12838 降至 0.03237/0.06809 m/s。历史缓存对应改善约 23.9%/46.0%。纵向从原微分模型的大幅改善主要来自有限帧间求解。完整开发数据、历史独立测试和本次重跑结果分别保存在 `evidence/`、`verification/`，不能把开发子集收益写成普遍保证。

当前方法适合平坦、纹理足够的路面和已有标定条件。它固定俯仰和高度，明显动态俯仰、坡道、道路横坡、遮挡和大侧倾均需另测。旧高速困难工况曾大面积回退，未证明稳定的独立侧倾收益。这里选的是现有代码中最适合继续做服务器验证的一版，没有完成实车或未见路线精度验收。

## 本包内容与验证

- `roll_velocity/`：整理后的光流、路面支持、法向、有限求解和推理流程。
- `model/`：原冻结 V3 网络，用其图像可靠性权重；主速度不采用旧学习修正/融合输出。
- `weights/`：两份原权重，共约 45 MiB。
- `data/`：四段完整开发 RGB、时间戳、固定标定，共 960 图、约 623 MiB。
- `evaluation/labels/`：独立的真值标签，仅供评价。
- `evidence/`：原开发指标、旧测试宏表和来源哈希。
- `verification/`：本机实际重跑结果与检查记录；不携带大体积流缓存。
- `package_manifest.json`：交付文件大小和 SHA-256；`verify_package.py` 用于传输校验。

本机验证使用 Windows、Python 3.10.20、RTX 5060 Laptop 8 GiB、torch 2.10.0+cu128。Linux 启动脚本已做语法检查；实际 Linux 服务器和其驱动仍需先执行上面的环境检查与冒烟命令。

具体检查与逐帧差异见 [验证报告](docs/验证报告.md)。历史光流固定时，四组共 3776 个输出与历史结果的最大差异小于 0.000054 m/s；重新计算的 RAFT 光流存在运行环境相关数值差异，因此不宣称跨环境逐位相同。
