import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nuscenes.nuscenes import NuScenes

# 【关键点】导入你前两阶段亲手写好的代码！
# 确保 step5_data_pipeline.py 和 step6_comma_model.py 都在当前目录下
from step5_data_pipeline import NuScenesSpeedDataset
from step6_comma_model import CommaSpeedModel

def main():
    # 1. 设备配置 (优先使用 GPU 小学生大军)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")

    # 2. 准备数据流水线
    print("正在加载 nuScenes 数据集，请稍候...")
    nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes_mini', verbose=False)
    # 我们先只拿场景 0 来做快速验证，熟练了之后你可以把所有场景的数据都拼起来
    # train_dataset = NuScenesSpeedDataset(nusc, scene_idx=0) 
    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True) # 一次看 8 组图
    train_dataset = NuScenesSpeedDataset(nusc, is_train=True) 
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    print(f"数据加载完毕！共有 {len(train_dataset)} 个样本，分为 {len(train_loader)} 个 Batch。")

    # 3. 实例化模型，并把它搬到对应的设备上
    model = CommaSpeedModel().to(device)

    # 4. 定义损失函数和优化器
    criterion = nn.MSELoss() # 均方误差
    optimizer = optim.Adam(model.parameters(), lr=0.001) # 学习率设为 0.001

    # 5. 开始训练循环
    num_epochs = 15 # 让模型把这些数据看 15 遍
    print("\n🚀 开始训练！")
    
    for epoch in range(num_epochs):
        model.train() # 告诉模型：现在是训练模式
        running_loss = 0.0
        
        for batch_idx, (images, true_speeds) in enumerate(train_loader):
            # 将数据也搬到 GPU (或 CPU) 上，确保和模型在同一个地方！
            images = images.to(device)
            true_speeds = true_speeds.to(device)
            
            # [步骤 A] 清空上一次的梯度
            optimizer.zero_grad()
            
            # [步骤 B] 前向传播：模型预测
            predicted_speeds = model(images)
            
            # [步骤 C] 计算误差 (Loss)
            loss = criterion(predicted_speeds, true_speeds)
            
            # [步骤 D] 反向传播：计算梯度
            loss.backward()
            
            # [步骤 E] 更新参数：优化器走一步
            optimizer.step()
            
            running_loss += loss.item()
            
        # 算一下这个 Epoch 的平均 Loss
        avg_loss = running_loss / len(train_loader)
        print(f"第 {epoch+1:02d}/{num_epochs} 轮 (Epoch) | 平均训练误差 (Loss): {avg_loss:.4f}")

    # 6. 保存模型参数
    torch.save(model.state_dict(), 'speed_model_v1.pth')
    print("\n🎉 训练完成！模型权重已保存为 'speed_model_v1.pth'")

if __name__ == "__main__":
    main()