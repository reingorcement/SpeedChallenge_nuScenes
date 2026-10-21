import torch
import torch.nn as nn
import torch.optim as optim

# 1. 定义一个极其简单的卷积神经网络 (CNN)
class SimpleSpeedCNN(nn.Module):
    def __init__(self):
        super(SimpleSpeedCNN, self).__init__()
        # 卷积层：假设输入是一张灰度图 (1个通道)，输出4个特征图，卷积核大小3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)
        # 激活函数：把线性的矩阵运算变成非线性的，让模型变聪明
        self.relu = nn.ReLU()
        # 全连接层：把卷积提取的特征“展平”后，映射到一个单一的数值（预测的车速）
        # 这里 222 是因为 224x224 的图经过 3x3 卷积没加 padding，尺寸变成了 222x222
        self.fc = nn.Linear(in_features=4 * 222 * 222, out_features=1)

    def forward(self, x):
        x = self.conv1(x)       # 提取特征
        x = self.relu(x)        # 激活一下
        x = x.view(x.size(0), -1) # 展平操作 (你在第2阶段学过的！)
        speed = self.fc(x)      # 输出预测车速
        return speed

# 2. 实例化模型、损失函数和优化器
model = SimpleSpeedCNN()
criterion = nn.MSELoss() # 使用均方误差 (最小二乘的思想)
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam 是一种非常主流的优化器，lr 是学习率(步长)

# 3. 伪造一些“玩具”数据
# 模拟 10 张单通道图像，大小 224x224
dummy_images = torch.rand(10, 1, 224, 224) 
# 模拟这 10 张图对应的真实车速 (随便生成的，比如在 0~30 m/s 之间)
true_speeds = torch.rand(10, 1) * 30 

print("开始训练啦！睁大眼睛看 Loss 的变化：\n")

# 4. 模拟训练循环 (迭代 5 次)
epochs = 5
for epoch in range(epochs):
    # 【第一步】前向传播：让模型猜一下车速
    predicted_speeds = model(dummy_images)
    
    # 【第二步】计算误差：用 MSE 对比预测车速和真实车速
    loss = criterion(predicted_speeds, true_speeds)
    
    # 【第三步】反向传播与参数更新 (最神奇的一步)
    optimizer.zero_grad() # 坑点预警：每次更新前，必须把上一次的梯度清零！
    loss.backward()       # 自动计算所有参数的梯度
    optimizer.step()      # 优化器根据梯度更新参数
    
    # 打印一下当前的进度
    print(f"第 {epoch + 1} 轮迭代 | 当前误差 (Loss): {loss.item():.4f}")

print("\n恭喜！你刚刚完成了一个神经网络的完整训练闭环！")