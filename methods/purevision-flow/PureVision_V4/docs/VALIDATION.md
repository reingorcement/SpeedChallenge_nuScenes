# V4 实际验证记录

日期：2026-10-21。所有生成文件位于 `PureVision_V4` 内，原工程和数据集只读。使用已有 `speedest` 环境，没有安装或升级依赖。

## 执行结果

| 检查 | 结果 | 记录 |
|---|---|---|
| 原目录完整性 | 原两个工程共 1252 个文件，大小和 SHA-256 全部一致，无新增文件 | `verification/isolation.json`、`source_baseline.json` |
| 单元及集成测试 | 39 passed，14.19 s | `verification/tests.log` |
| 原冻结权重加载 | V3 与 RAFT 严格加载通过；从 `/tmp` 调用本工程检查入口成功 | `verification/environment.json` |
| 四工况 RGB 推理 | 每段读取原序列第 100–131 帧，32 帧；F_RC/F_R 各 28/28 可评价窗口有效 | `verification/final_rgb/summary.json` |
| nuScenes 真图像推理 | scene-0001 前 8 帧，F_RC/F_R 各 4/4 可评价窗口有效 | `verification/final_nuscenes/` |
| nuScenes 新标签转换 | train_core/scene-0001 的 233 帧、dev/scene-0005 的 224 帧，完整三维体轴标签；原图片未复制或修改 | `verification/nuscenes_export/` |
| 同一历史光流回归 | F_RC/F_R/F_B 有效标志一致；最大纵横速度差小于 0.000014 m/s | `verification/fixed_flow_final/comparison.json` |
| 真实黑屏 + RAFT | 修复后 F_RC/F_R 最后一个窗口均无效，速度空值，原因 textureless_input | `verification/blank_fixed/` |
| 独立目录迁移 | 复制代码、配置、权重和 5 张图像，从 `/tmp` 启动；审计钩子禁止读取其他工程路径，全链成功 | `verification/portability.json`、`portability.log` |

迁移测试使用的临时工程副本已删除，保留执行日志。最终验证的原始光流缓存可以重新生成；速度、法向、运行指纹和比较结果保留。

## 测试覆盖

- 15 组完整运动投影：静止、仅旋转、直行、倒车、混合三轴运动、非零垂向速度、侧倾/俯仰、不同 dt 和非零杆臂。验证参考点速度与角速度符号、尺度以及协方差。
- 历史有限求解器的加噪观测回归、像素中心缩放、旧标定到新统一标定的等价性。
- 点数不足、错误 dt、数值异常、大重投影残差、非法协方差、无效帧对剔除及 75 ms 测量时刻计算。
- 单应性对应点触发真实法向更新，源帧法向路由、保持、时间间断回退。
- 完整三维 pose 标签转换，非均匀采样，输入字段白名单、路径边界与原目录写入拒绝。
- 批量几何与梯度边界，融合各头有限梯度，几何失效时精确 RGB 回退。
- 直接头训练 → 融合训练 → checkpoint 保存/加载 → 续训，以及训练后的头重新接入图像推理。这里使用合成标签验证功能，不是模型精度实验。
- train/dev 重叠拒绝、旧标签契约拒绝、全失效工况纳入覆盖率、输出恢复及输入变更拒绝。
- 黑、白、纯红图像无纹理判定；独立命名空间导入。

## 固定光流回归

使用原补偿包 `verification/smoke/cache/flow` 中 7 对光流和原 8 张 RGB，每个方法 4 个可评价窗口；重新计算路面支持、法向、冻结图像权重和几何速度。

| 方法 | 最大 vx 差 (m/s) | 最大 vy 差 (m/s) |
|---|---:|---:|
| F_RC | 0.00001101 | 0.00000670 |
| F_R | 0.00001303 | 0.00000469 |
| F_B | 0.00000917 | 0.00000605 |

误差小于设定的 `1e-4 m/s` 容差。该检查支持整合后几何链与原方法一致，不是独立精度改善的证据。重算 RAFT 会受 GPU/运行环境数值差异影响，不应套用此固定光流容差。

可复做：

```bash
python -B tools/verify_fixed_flow.py \
  --input ../补偿速度估计/data/HW_DEV_SINE40_A12 \
  --flow-cache ../补偿速度估计/verification/smoke/cache/flow \
  --reference-csv ../补偿速度估计/verification/smoke/speeds.csv \
  --output verification/repeated_fixed_flow
```

## 黑屏缺陷的定位与修复

检查中额外运行了真实 RAFT，而非人为提供零置信度。在均匀黑图上，RAFT 产生的伪流通过了前后向一致性和原平面支持流程，初始版本错误输出有效速度，约 vx=-0.0286、vy=0.0538 m/s。原记录保留在 `verification/blank_actual_raft/`，它是修复前证据。

最终版本检查全幅空间亮度变化与路面内部纹理；该输入现在被拒绝，融合和 RGB 输出也不能将完全无信息的图像伪装成有效估计。正常工况重跑、固定光流回归和全部测试在此修复之后通过。

## 边界

四工况仅验证了指定中段，共 128 张图像、每个方法 112 个可评价窗口；没有重新执行全部 960 张开发图像。nuScenes 只做了 8 帧真图像推理和两场景标签导出，没有完成全量训练、全量独立测试或跨场景泛化验收。

新融合代码已通过真实训练/重载功能测试，但未交付经过正式数据训练与独立评价的 V4 融合权重。默认运行的 F_RC/F_R 不需要新训练。主代码会拒绝将旧 V3 checkpoint 当成 V4 融合 checkpoint。

环境：Linux、PyTorch 2.10.0+cu128、torchvision 0.25.0+cu128、NumPy 1.26.4、SciPy 1.15.3、OpenCV 4.11.0。GPU 测试使用 NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition。耗时统计已修正，本文不据此声明已达成实时部署要求。
