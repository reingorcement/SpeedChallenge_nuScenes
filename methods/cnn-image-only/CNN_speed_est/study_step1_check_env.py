# step1_check_env.py
import torch

# 打印当前安装的 PyTorch 版本
print(f"PyTorch 版本: {torch.__version__}")

# 检查是否能成功调用 GPU (CUDA)
if torch.cuda.is_available():
    print("恭喜！GPU 加速已就绪，小学生大军已准备完毕！")
    print(f"检测到的显卡: {torch.cuda.get_device_name(0)}")
else:
    print("未检测到可用的 GPU。没关系，我们可以先用 CPU (数学教授) 进行代码调试和初步学习。")
    
# 创建一个随机的 3x3 矩阵 (Tensor)，这和你在线代里学的矩阵是一回事
dummy_matrix = torch.rand(3, 3)
print("\n测试生成一个 3x3 的张量 (Tensor):")
print(dummy_matrix)