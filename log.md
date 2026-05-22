# 训练运行问题与修改记录

## 1. 服务器首次运行训练时报错

### 用户反馈的问题

在服务器环境 `zy_speedest` 中运行：

```bash
python train.py --config configs/default.yaml
```

程序已经识别到 CUDA：

```text
Using device: cuda
```

但随后在导入 `nuscenes-devkit` / `scikit-learn` 相关模块时失败，核心报错为：

```text
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6:
version `GLIBCXX_3.4.29' not found
```

### 问题原因

这不是 `FlexiNet_ChassisFusion` 训练代码本身的逻辑错误，而是服务器 Python 环境中的 C++ 运行库版本不匹配。

调用链大致为：

```text
train.py
-> dataset/nuscenes_chassis.py
-> from nuscenes.nuscenes import NuScenes
-> nuscenes-devkit
-> sklearn.metrics
-> scikit-learn 的 C++ 扩展
-> 需要 GLIBCXX_3.4.29
```

当前运行时加载到了系统路径下较旧的：

```text
/lib/x86_64-linux-gnu/libstdc++.so.6
```

而不是 conda 环境中较新的 `libstdc++`。

### 根据建议做出的修改

在服务器 conda 环境中安装更新的 C++ 运行库：

```bash
conda activate zy_speedest
conda install -c conda-forge -y "libstdcxx-ng>=12" "libgcc-ng>=12"
```

并让当前 shell 优先使用 conda 环境中的动态库：

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

然后验证：

```bash
python -c "import sklearn; from nuscenes.nuscenes import NuScenes; print('nuscenes/sklearn import ok')"
```

### 为什么这样修改

`scikit-learn` 的二进制扩展需要较新的 `GLIBCXX_3.4.29`。更新 `libstdcxx-ng` / `libgcc-ng` 并优先加载 conda 环境中的库，可以避免 Python 包链接到系统旧版 `libstdc++`，从而解决 `nuscenes-devkit` 导入失败的问题。

## 2. 训练已经启动，但速度过慢

### 用户反馈的问题

训练成功启动后，终端显示：

```text
Using device: cuda
Train scenes: 685 | Train samples: 151663
Val scenes: 150 | Val samples: 33148
Epoch 1/40
59/37916 [00:58<8:44:39, 1.20it/s]
```

根据这个速度估算：

```text
1 个 epoch 约 8 小时 44 分钟
40 个 epoch 约 14.5 天
```

### 监控结果

`nvidia-smi` 显示：

```text
GPU: NVIDIA GeForce RTX 5090
GPU-Util: 3%
Memory-Usage: 约 6983 MiB / 32607 MiB
```

`iostat -xz 1` 显示：

```text
sda %util: 98% ~ 99%
nvme0n1 %util: 0% ~ 1%
CPU idle: 约 97%
```

### 问题原因

瓶颈主要在数据读取，而不是 GPU 计算。

当前训练配置中：

```yaml
batch_size: 4
seq_len: 13
```

因此每个 step 需要读取：

```text
4 个样本 × 13 帧 = 52 张图片
```

一个 epoch 需要读取约：

```text
151663 个样本 × 13 帧 = 1,971,619 张图片
```

数据位于 `sda` 机械硬盘或较慢硬盘上，训练过程中需要大量读取小 JPEG 文件，导致 `sda` 的 I/O 利用率接近 100%。GPU 只能等待 dataloader 提供数据，因此 GPU 利用率只有约 3%。

## 3. 暂时不能迁移数据到 NVMe 时的临时优化

### 用户反馈的限制

当前暂时不能把 nuScenes 数据从 `sda` 迁移到 NVMe。

### 根据建议做出的修改

最终选择：

```yaml
train:
  batch_size: 16
```

并建议配套使用更适合机械硬盘的设置：

```yaml
train:
  batch_size: 16
  num_workers: 1
  epochs: 5
  use_amp: true
```

同时建议将 `train.py` 中训练集 dataloader 的随机打乱关闭：

```python
train_loader = DataLoader(
    train_set,
    batch_size=int(cfg["train"]["batch_size"]),
    shuffle=False,
    num_workers=int(cfg["train"]["num_workers"]),
)
```

原始设置是：

```python
shuffle=True
```

### 为什么选择 `batch_size: 16`

服务器 GPU 为 RTX 5090，显存约 32GB。当前训练时显存占用约 6.8GB，说明 `batch_size: 4` 明显偏保守。

增大 batch size 的主要收益是减少每个 epoch 的 step 数：

```text
batch_size 4:  37916 step / epoch
batch_size 8:  约 18958 step / epoch
batch_size 16: 约 9479 step / epoch
```

在数据读取受限的情况下，减少 step 数可以降低训练循环调度开销，并更充分利用 GPU 显存。

### 为什么建议 `num_workers: 1`

机械硬盘不擅长大量并发随机读取。`num_workers` 过高时，多个 dataloader worker 会同时从不同位置读取小图片文件，导致磁盘频繁寻道，可能让 `sda` 更容易达到 100% 利用率，速度反而更慢。

因此，在数据仍位于机械硬盘时，先使用：

```yaml
num_workers: 1
```

更稳妥。后续可以再对比：

```yaml
num_workers: 2
num_workers: 4
```

以实际 `it/s` 为准。

### 为什么建议 `shuffle=False`

当前数据集样本是连续帧滑动窗口，例如：

```text
样本 1: frame 1~13
样本 2: frame 2~14
样本 3: frame 3~15
```

相邻样本之间会共享大量图片。如果 `shuffle=False`，样本按顺序读取，操作系统文件缓存更容易复用最近读取过的图片。

如果 `shuffle=True`，dataloader 会随机跳转到不同 scene 和不同图片位置，机械硬盘需要频繁随机寻道，缓存命中率低，I/O 性能会明显下降。

因此，在暂时不能迁移数据到 NVMe 的情况下，关闭 shuffle 是一种临时的机械硬盘友好策略。代价是训练随机性降低，相邻 batch 的相关性变强，但可以先用于验证模型是否正常下降并获得初步 checkpoint。

### 为什么建议先把 `epochs` 改为 5

完整 `v1.0-trainval` 数据集已经包含：

```text
Train samples: 151663
Val samples: 33148
```

在当前 I/O 条件下直接跑 40 个 epoch 成本过高。先设置：

```yaml
epochs: 5
```

可以更快观察：

```text
train_loss 是否下降
val_loss 是否下降
val_mae / val_rmse 是否合理
模型是否优于 base_speed baseline
```

确认训练流程有效后，再决定是否增加 epoch 数或迁移数据到 NVMe 后进行正式训练。

## 4. 当前推荐运行配置

在暂时不能迁移数据的条件下，推荐临时配置为：

```yaml
train:
  batch_size: 16
  num_workers: 1
  epochs: 5
  learning_rate_head: 0.001
  learning_rate_backbone: 0.00003
  weight_decay: 0.0001
  use_amp: true
  seed: 42
  log_dir: runs/FlexiNet_ChassisFusion
  results_dir: results_git/FlexiNet_ChassisFusion
  save_dir: checkpoints
  loss: smoothl1
  grad_clip: 2.0
```

训练启动命令：

```bash
cd ~/WorkSpaces/zy/SpeedChallenge_nuScenes/FlexiNet_ChassisFusion
conda activate zy_speedest
python train.py --config configs/default.yaml
```

监控命令：

```bash
watch -n 1 nvidia-smi
iostat -xz 1
```

重点观察：

```text
GPU-Util 是否提高
sda %util 是否仍然接近 100%
训练速度 it/s 是否提高
显存是否足够
```

## 5. 后续优化方向

如果后续可以迁移数据，优先将以下目录复制到 NVMe：

```text
v1.0-trainval
can_bus
maps
samples/CAM_FRONT
sweeps/CAM_FRONT
```

并修改：

```yaml
data:
  dataroot: /path/to/nvme/nuscenes
  can_bus_root: /path/to/nvme/nuscenes/can_bus
```

如果迁移到 NVMe 后仍然较慢，再考虑代码层优化：

1. 为 CAN bus 时间戳查找增加二分索引，避免每个样本线性扫描 CAN 消息。
2. 在 dataset 中增加图片 tensor LRU 缓存，复用相邻滑动窗口中重复的帧。
3. 将图像序列预处理为 LMDB、WebDataset、HDF5 或内存映射格式，减少大量小文件读取开销。
4. 在 NVMe 环境下重新调大 `num_workers`，例如 8 或 12。

## 6. 已落实到代码中的临时优化

### 修改的文件

```text
FlexiNet_ChassisFusion/configs/default.yaml
FlexiNet_ChassisFusion/dataset/nuscenes_chassis.py
FlexiNet_ChassisFusion/train.py
FlexiNet_ChassisFusion/test.py
```

### 配置修改

将训练配置改为机械硬盘友好设置：

```yaml
data:
  image_cache_size: 4096

train:
  batch_size: 16
  num_workers: 1
  epochs: 5
  shuffle_train: false
```

含义：

```text
image_cache_size: 4096
```

表示每个 dataloader worker 最多缓存 4096 张已经完成 resize、灰度化、归一化后的图片 tensor。

```text
batch_size: 16
```

利用 RTX 5090 的显存余量，减少每个 epoch 的 step 数。

```text
num_workers: 1
```

避免机械硬盘被多个 worker 并发随机读取进一步拖慢。

```text
shuffle_train: false
```

让样本按 scene 内连续窗口顺序读取，提高操作系统文件缓存和自定义 LRU 图片缓存的命中率。

```text
epochs: 5
```

先做短周期验证，避免在 I/O 受限条件下一开始就运行 40 个 epoch。

### Dataset 修改

在 `NuScenesChassisFusionDataset` 中增加了图片 LRU 内存缓存：

```text
token -> transformed image tensor
```

相邻样本共享大量连续帧，例如：

```text
样本 1: frame 1~13
样本 2: frame 2~14
```

启用缓存后，样本 2 中的 frame 2~13 可以直接复用，减少重复打开 JPEG、重复解码和重复 transform。

### CAN bus 查找优化

将 CAN bus 消息按 `utime` 排序，并缓存时间戳列表。原来的查找方式是每次线性扫描：

```text
找出所有 utime <= 当前图像时间戳的记录，再取最大值
```

现在改为二分查找：

```text
bisect_right(timestamps, timestamp_us) - 1
```

这样每次 CAN 特征对齐从线性扫描变成对数时间查找，减少 CPU 侧开销。

### DataLoader 修改

`train.py` 和 `test.py` 中的 DataLoader 增加：

```python
pin_memory=device.type == "cuda"
persistent_workers=num_workers > 0
```

作用：

```text
pin_memory
```

加快 CPU 到 GPU 的 batch 拷贝。

```text
persistent_workers
```

让 dataloader worker 在多个 epoch 之间保持存活，避免每个 epoch 重新启动 worker，也有利于保留 worker 内部的图片缓存。

### 验证结果

本地已完成语法检查：

```bash
python -m py_compile FlexiNet_ChassisFusion/dataset/nuscenes_chassis.py FlexiNet_ChassisFusion/train.py FlexiNet_ChassisFusion/test.py
```

并在 `vision_est` 环境中完成 dataset 模块导入检查：

```bash
conda run -n vision_est python -c "from dataset.nuscenes_chassis import NuScenesChassisFusionDataset, NuScenesCanBusAdapter; print('dataset import ok')"
```

输出：

```text
dataset import ok
```
