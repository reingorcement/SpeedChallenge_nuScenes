# 光流纯视觉速度估计

三个实现保持独立目录、配置、权重和输出。各自进入目录运行，不把全局 `model`/`dataset` 包混在同一 Python 进程。

| 目录 | 内容 | 运行状态 |
|---|---|---|
| [PureVision_V3](PureVision_V3/README.md) | RGB direct、微分六自由度光流几何、视觉融合 | 附 direct/fusion 的 best/last 权重 |
| [补偿速度估计](补偿速度估计/README.md) | 有限位姿求解；F_R 固定法向、F_RC 受约束侧倾法向、F_B 完整视觉法向 | 附冻结网络和 RAFT 权重 |
| [PureVision_V4](PureVision_V4/README.md) | 隔离整合、三维坐标/杆臂处理、统一训练推理评价 | F_RC/F_R 可运行；新融合头仍需正式训练 |

以上方法推理不读取 CAN、IMU、真值速度或真实动态姿态；使用图像、图像派生光流、时间戳及固定相机/地面参数。V3 也估计旋转，不能将版本差异简称为“无旋转/有旋转”。

## 本次仓库打包范围

- V3/V4 按原交付清单复制，Python 算法代码、配置、权重不修改；保留原清单及关键验证记录。
- 补偿版保留代码、权重、标定/帧清单、标签及历史证据；不附 960 帧原始 RGB、压缩包和生成缓存。`data/` 下仅有元数据，不是完整可运行图像样例。原 README 的全量样例命令需要先补齐图像。
- 补偿版的 `package_manifest.json` 针对仓库实际文件重新生成；原完整包清单保存于 `evidence/original_package_manifest.json`。
- 原 README 中的本机路径、已有实验数值及原环境记录是历史信息。代码副本位置变更后，使用实际克隆目录和自己的 Python 环境。
- V4 的 `tools/check_isolation.py` 仅用于原始工作区的 1252 文件审计，要求完整旧项目（包括原数据/压缩包）；精简仓库不适用。仓库交付完整性通过 `tools/verify_manifest.py` 检查。

## 不依赖样例数据的检查

```bash
cd PureVision_V3
python -B scripts/check_package.py
python -B -m pytest -q -p no:cacheprovider tests

cd ../补偿速度估计
python -B verify_package.py
python -B -m unittest discover -s tests -v

cd ../PureVision_V4
python -B check.py
python -B -m pytest -q -p no:cacheprovider tests
```

在补偿版或 V4 中处理自备 RGB 序列：

```bash
python -B run.py --input /path/to/sequence --output runs/example --frames 8 --device cuda:0 --methods F_RC F_R
```

输入格式见各自 README。V4 的运行输出只允许在自身 `runs/`、`training/`、`prepared/`、`verification/` 内。现有历史样例和完整 nuScenes 数据不上传到方法分支。
