# 方法分类与输入契约

分类以实际预测路径为准；分支名、目录名或类名含有 visual/fusion/chassis 都不是充分依据。

| 方法 ID | 代码/配置 | 推理输入 | 输出 | 底盘相关边界 |
|---|---|---|---|---|
| cnn-image-v1 | cnn-image-only/CNN_speed_est | 两帧 RGB | 平面车速大小 | 无底盘融合；pose 仅作标签 |
| flexinet-image-original | flexinet-image-only/FlexiNet-main | 灰度图像序列 | 平面车速大小 | CAN 文件用于场景筛选，不输入模型 |
| flexinet-image-xy | flexinet-image-only/FlexiNet_ImageXY，vision_only 配置 | 灰度图像序列 | vx/vy | 关闭底盘预测分支；旧数据入口仍需要 CAN |
| v3-rgb-direct | purevision-flow/PureVision_V3，direct_pretrain | RGB 序列 | vx/vy | 无光流、无 CAN；直接预训练权重与融合模型内部 RGB 专家分别记录 |
| v3-flow-geometry | V3 run_v3_ablation.py，flow_geometry_only | 光流/置信度、dt、固定标定/地面 | vx/vy | 无 CAN；该独立对照不读取 RGB 网络特征，不等同于完整模型的 learned-support 几何输出 |
| v3-visual-fusion | V3 fusion | RGB + 光流 + dt + 固定标定/地面 | vx/vy | 视觉内部融合；无底盘融合 |
| finite-F_R | 补偿版/V4 F_R | RGB + 光流 + 固定法向/标定 | vx/vy | 有限运动求解同时估计旋转和平移 |
| finite-F_RC | 补偿版/V4 F_RC | 同上 + 视觉侧倾法向 | vx/vy | 固定俯仰/高度，非 IMU 补偿 |
| finite-F_B | 补偿版/V4 F_B | 同上 + 完整视觉法向 | vx/vy | 实验变体，不宣称普遍优于 F_RC |
| v4-visual-fusion | V4 fusion/rgb_only | 新几何、RGB 特征及新训练头；rgb_only 为直接分支 | vx/vy | 新头尚需正式训练，附带 V3 权重不是训练完成的 V4 权重 |
| v5-R00--R05 | PureVision_V5/configs | 因果图像派生的冻结描述子、数值光流、基础/FB 掩码、历史 dt，按实验选取 | 后轴中点 yaw 系 vx/vy PLANAR_PROXY | 不读取 CAN/真实姿态；独立特征融合后仅一个速度头 |

目录均相对于 `methods/`。方法分支只包含所属目录；跨路线实现通过 main 或对应分支访问。

## 不归入严格纯视觉的历史方法

旧 fair_flow/fair_rgb_flow_geom 仍融合底盘；KD/VKD 使用运动学和动力学代理特征；V2 visual/geometry 仍可能通过 CAN 角速度去旋转、通过轮速修正尺度或回退；V2 visual_kinematic 另有弱运动学修正。全部保存在 archive/legacy。

## 版本、权重和评价口径

同一方法不同 seed、best/last、预训练/微调权重属于实验记录，不能只按文件名视为新算法。每项记录要关联来源提交、配置、权重哈希和数据划分。

CNN/原版 FlexiNet 的车速大小不能直接作为侧向速度评价对象。V3 旧 yaw 平面标签与 V4 完整四元数转换的标签处理不等价；nuScenes 的二维定位不能因该转换升级为真实三维体轴标签。V5 明确使用非均匀三点导数的 PLANAR_PROXY。必须记录坐标轴、参考点、标签定义、测量时刻及有效覆盖率后再比较。

本次整理验证代码完整性、加载兼容性及已有测试；没有重新训练或做完整数据集精度验收。历史结果留在归档，V4 已有小样本证据留在其 verification/ 和 docs/ 中，均不当作新增精度结论。
