# 独立包验证记录

整理日期：2026-10-20。

## 已完成

- 从当前 `v3-visual-only/FlexiNet_ChassisFusion` 文件提取 V3 训练、测试、预处理及递归导入依赖。
- 独立目录内运行原有测试：`38 passed in 10.52s`。
- 从项目外的工作目录调用 `scripts/check_package.py --cuda`，依赖导入和四个随包权重的严格加载全部通过。
- 36 个 Python 文件语法检查通过。
- `server.sh`、`server.env` 和 Linux 训练流水线的 Bash 语法检查通过，帮助入口可执行。
- Linux 脚本统一为 LF 换行。
- 四个权重文件的 SHA-256 与源文件逐一相同；原数据划分内容一致。

本地验证环境：Python 3.10.20、PyTorch 2.10.0+cu128、torchvision 0.25.0+cu128，CUDA 可用。

## 随包权重

| 文件 | 阶段 | 保存 epoch（从 0 开始） | best_epe（checkpoint 记录，m/s） |
|---|---|---:|---:|
| `checkpoints/pretrained/direct_seed_42/best.pth` | RGB 直接回归 | 19 | 0.391467 |
| `checkpoints/pretrained/direct_seed_42/last.pth` | RGB 直接回归 | 19 | 0.391467 |
| `checkpoints/pretrained/fusion_seed_42/best.pth` | 视觉融合 | 8 | 0.235022 |
| `checkpoints/pretrained/fusion_seed_42/last.pth` | 视觉融合，已早停 | 11 | 0.235022 |

上述数值读取自已有 checkpoint，并非本次重新测试所得。

## 整理时的改动

- 模型前向计算、损失、数据读取、训练和测试 Python 实现保持源文件内容。
- `model/v2/__init__.py` 精简为包说明，避免导入与 V3 无关的 V2 网络；被引用的几何工具保留。
- V3 YAML 中的数据路径统一为 `/data/nuscenes`；默认直接回归初始化权重改为包内已有权重。
- Linux 流水线改用原有 `checkpoint_complete` 辅助函数识别已早停阶段，并支持通过 `V3_SEEDS` 选择种子。
- 新增 `server.sh`、`server.env`、`scripts/check_package.py`、依赖文件和本说明。
- 仅保留 V3 配置和原有 V3 测试。保留的数据/几何工具中部分命名含 `v2`、`chassis`，V3 入口显式关闭 CAN 适配器。

## 服务器验证

本次未执行完整 nuScenes 数据加载、缓存生成、全量推理或训练。复制到服务器后，先设置 `server.env`，执行 `bash server.sh check`，再执行 `bash server.sh test-smoke` 验证实际数据、缓存与 GPU 推理链路。

`MANIFEST.sha256` 可在解压后的项目根目录用 `sha256sum -c MANIFEST.sha256` 验证传输完整性；修改配置前执行。
