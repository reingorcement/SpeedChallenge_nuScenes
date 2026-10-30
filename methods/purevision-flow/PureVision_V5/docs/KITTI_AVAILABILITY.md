# 本机 KITTI 可用性核验（2026-10-30）

检查位置：`/mnt/sdb2/Workspace/zhangyu/dataset/kitti`，约 78 GiB（`du -sh` 显示 78G）。
这里只读检查实际目录、全部 zip 中央目录、开发包说明、映射表和少量图像/标定/标签。
没有下载新数据，没有把物体检测标签转换成自车速度，没有改变当前 nuScenes R00—R05 实验。

## 实际已有内容

本机是 **KITTI Object Benchmark**，根目录 `readme.txt` 明确说明对象检测、三维检测和鸟瞰检测任务。

| 分区 | image_2 PNG | calib TXT | label_2 TXT | velodyne BIN |
|---|---:|---:|---:|---:|
| training | 7,481 | 7,481 | 7,481 | 7,481 |
| testing | 7,518 | 7,518 | 无 | 7,518 |

`label_2` 是类别、截断、遮挡、目标框、目标尺寸、相机坐标目标位置和朝向；没有自车 vx/vy。
`calib` 含 P0—P3、R0_rect、Tr_velo_to_cam、Tr_imu_to_velo 等固定标定；固定外参不提供车辆动态位姿或速度。
抽查图像：training/000000 为 1224×370，training/000001 与 testing/000000 为 1242×375。

本机全部子目录只有 `archives/cpp/mapping/matlab/training/testing` 及上述传感器子目录。
检查了 `data_object_image_2.zip`、`data_object_calib.zip`、`data_object_label_2.zip`、
`data_object_velodyne.zip`、`devkit_object.zip` 的全部条目；没有 OXTS、timestamps 或 ego-pose 条目。
没有发现 KITTI Raw 的完整 drive 目录，也没有 KITTI Odometry 的 sequences/poses/times 文件。

## 有可还原的连续图像，但不能按数字文件名拼帧

根目录开发包提供 `mapping/train_rand.txt` 和 `mapping/train_mapping.txt`。
按本机说明，`train_rand` 将 Object training 图像 ID 对应到 **1-based** 的 mapping 行；
mapping 行给出 Raw 日期、drive 和原始帧号。

检查得到：

- 两表均覆盖 7,481 条记录；随机排列恰好包含 1—7,481，无重复遗漏。
- 原始映射覆盖 141 个 drive。
- 同一 drive 内排序原始帧号，确有 **116 段至少五帧连续序列**，最长 174 帧。
- 按连续原始帧号可构造 **2,263 个五帧候选窗口**；这只是图像连续性候选数量，不是已通过时间或标签核验的训练样本数。
- 这些候选来自 **33 个 drive**；例如 drive_0101 有 375 个候选，drive_0014 有 184 个。
- 检测图像 000000 对应 `2011_09_28_drive_0106_sync` 的 raw frame 48；000001 对应
  `2011_09_26_drive_0101_sync` 的 raw frame 528。数字文件名相邻并不意味着连续采集。
- 只有 training 的 Raw 映射可用；当前 testing 不能凭文件名重建时序。

因此，“本机只有孤立图片，完全没有可恢复时间序列”过于绝对；准确阻碍是
**缺少可核验的实际图像时间戳和自车运动监督**。原始帧号不能替代时间戳，不能默认固定 10 Hz 后形成正式速度标签。

## 下一步接入条件

1. 优先针对已经存在五帧候选窗口的 Raw drives 补齐对应图像时间戳、OXTS/可信动态位姿及必要标定，
   保留来源与校验信息。现有 Object PNG 可在完成映射一致性核验后复用，避免重复存全部图像。
2. 明确速度是哪个参考点、哪个坐标系、哪个时刻的量；核对 OXTS 字段与外参含义，验证杆臂、符号、单位和时间配对。
   不能将物体坐标、静态 IMU 外参或相机轨迹差分直接称为真实车体 vx/vy。
3. 按完整 drive 划分 train/dev/test，先固定清单；不可随机切 Object 图像 ID 或重叠窗口。
   若与其他数据集联合训练，另立配置和结果表，不能静默改变已固定的 nuScenes 六组比较。
4. 当前 V5 RAFT 入口明确要求 16:9，KITTI 上述宽高比不满足。接入时需单独版本化尺寸变换、
   向量分量缩放和合法性掩码，并一致应用到各对照；不能直接把 KITTI 图像塞入当前缓存目录。
5. 接入后仍只允许图像、图像派生光流和时间进入推理。OXTS/动态位姿只用于离线监督与评价，
   新标签先通过直线、侧移、转弯和参考点转换检查，再决定其可声称的任务范围。

在缺失元数据补齐前，KITTI 不进入当前 vx/vy 主排名；nuScenes S2 训练与评价继续推进。

## 可复查命令

```bash
KITTI_ROOT=/mnt/sdb2/Workspace/zhangyu/dataset/kitti
find "$KITTI_ROOT" -maxdepth 3 -type d
sed -n '1,25p' "$KITTI_ROOT/readme.txt"
sed -n '/Mapping to Raw Data/,$p' "$KITTI_ROOT/readme.txt"
head -n 5 "$KITTI_ROOT/mapping/train_mapping.txt"
```

连续窗口统计定义：先用 `train_rand[id]-1` 取对应 mapping 行，按 drive 分组并去重 raw frame，
排序后把差值为 1 的帧连成段；每段长度 L 对应 `max(0,L-4)` 个 T=5 候选窗口。

核验文件 SHA-256：

- `readme.txt`: `00b967f8105a1b5161e90d24500fe0089715b8741eaecc42eb0065982ef10951`
- `mapping/train_rand.txt`: `0976b8b4f876f538010a9690249b60c0766d852e23acd5e07b4c3e0bd70f305f`
- `mapping/train_mapping.txt`: `ca9a80524adbfbebc3179987e70a8dab1488a88ba368e77715d561c901196a54`
