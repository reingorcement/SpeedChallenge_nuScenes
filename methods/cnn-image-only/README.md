# 普通 CNN：纯图像车速基线

方法 ID：`cnn-image-v1`。代码从原 `main@d415109` 提取，算法文件和原权重保持字节一致。

输入是两帧 RGB 拼成的 6 通道张量（224×224），模型为 `CommaSpeedModel`，输出平面车速大小，单位 m/s；不输出独立 vx/vy。ego pose 只生成监督标签。`optical_flow.py` 是独立可视化脚本，不在速度模型的训练预测链上。

进入 `CNN_speed_est/` 后，原训练入口是 `python step7_train.py`，验证入口是 `python step8_validate.py`。按原脚本配置 nuScenes/nuImages 数据路径；训练会覆盖该目录的 `speed_model_v1.pth`，重训前应保存原权重副本。原代码仍含历史路径、依赖和场景切分假设，本次没有改动数据集逻辑，也没有重新宣称精度。

本分支保留已存在的权重。历史图表及完整原目录见 `archive/legacy` 的 `snapshots/main/CNN_speed_est/`。
