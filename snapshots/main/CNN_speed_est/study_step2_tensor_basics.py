import torch

# 1. 模拟一张彩色图像的张量 (通道数 C=3, 高度 H=224, 宽度 W=224)
# 在 PyTorch 中，处理图像的标准维度顺序是 (C, H, W)
dummy_image = torch.rand(3, 224, 224)
print(f"单张图像的维度 (Shape): {dummy_image.shape}")

# 2. 模拟神经网络的批处理输入 (Batch Size N=4)
# 也就是一次性打包 4 张图片，维度变成 4 维张量 (N, C, H, W)
batch_images = torch.rand(4, 3, 224, 224)
print(f"批量图像的维度: {batch_images.shape}")

# 3. 矩阵乘法模拟
# 这在神经网络的全连接层中非常常用。回顾一下线代规则：(2x5) 的矩阵乘以 (5x3) 的矩阵。
matrix_A = torch.rand(2, 5)
matrix_B = torch.rand(5, 3)
result_matrix = torch.matmul(matrix_A, matrix_B)  # 也可以简写为 matrix_A @ matrix_B
print(f"\n矩阵 A 乘 矩阵 B 的结果维度: {result_matrix.shape}")

# 4. 改变张量形状 (Reshape / View) —— 极其重要！
# 比如在估计车速的最后一步，我们需要把立体的特征图“展平”成一个一维长条，才能输出一个具体的车速数值。
# -1 是一个偷懒的写法，代表“让 PyTorch 自动帮我计算这个维度应该是多少”
flattened_image = dummy_image.view(-1) 
print(f"\n展平后的单张图像维度: {flattened_image.shape}")