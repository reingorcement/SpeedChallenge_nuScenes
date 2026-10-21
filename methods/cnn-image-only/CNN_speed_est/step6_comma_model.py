import torch
import torch.nn as nn

class CommaSpeedModel(nn.Module):
    def __init__(self):
        super(CommaSpeedModel, self).__init__()
        
        # --- 特征提取区 (卷积层) ---
        # 模仿经典的自动驾驶网络架构：5 个卷积层不断提取更深层的运动和边缘特征
        # 重点：in_channels=6，完美对接我们第 5 阶段输出的两帧拼接图！
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=24, kernel_size=5, stride=2),
            nn.ELU(), # 开源项目很喜欢用 ELU 而不是 ReLU，通常在这个任务上表现更好
            
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2),
            nn.ELU(),
            
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2),
            nn.ELU(),
            
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3),
            nn.ELU(),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ELU()
        )
        
        # --- 决策输出区 (全连接层) ---
        # 把立体特征图展平后，逐步降维，最后得出一个具体的车速数值
        # 这里的 64 * 21 * 21 是怎么来的？是 224x224 的图片经过上面 5 次卷积后缩小的尺寸
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 21 * 21, 100), 
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1) # 最终输出：1个代表速度的数值
        )

    def forward(self, x):
        # 1. 图像进入卷积层提取特征
        x = self.conv_layers(x)
        
        # 2. 展平操作！(把多维张量拍扁成一维向量)
        x = x.view(x.size(0), -1) 
        
        # 3. 进入全连接层得出速度预测
        speed = self.fc_layers(x)
        return speed

# --- 测试我们的实战级模型 ---
if __name__ == "__main__":
    # 实例化模型
    model = CommaSpeedModel()
    
    # 模拟第 5 阶段 DataLoader 给我们的 Batch 数据：4 张图，6通道，尺寸 224x224
    dummy_input = torch.rand(4, 6, 224, 224)
    
    print("模型已搭建完毕，准备输入测试数据...")
    print(f"输入数据的形状: {dummy_input.shape}")
    
    # 前向传播
    predicted_speeds = model(dummy_input)
    
    print(f"输出数据的形状: {predicted_speeds.shape}")
    print(f"模型预测的瞎猜车速结果 (因为还没训练):\n{predicted_speeds}")