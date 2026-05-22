import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import matplotlib.pyplot as plt

# 引入我们自定义的 Dataset 和 FlexiNet 模型
from dataset.nuscenes_mini import NuScenesFlexiDataset
from model.FlexiNet import FlexiNet

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate(model, dataloader_val, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in tqdm(dataloader_val, desc="验证集", leave=False):
            img_batch, speed = data
            img_batch = img_batch.to(device)
            speed = speed.to(device)
            
            predicted_speed = model(img_batch)
            val_loss = criterion(predicted_speed, speed.unsqueeze(1))
            running_loss += val_loss.item()
            
    return running_loss / len(dataloader_val)

def train_model(args):
    # 1. 设备与环境初始化
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 使用计算设备: {device}")
    
    os.makedirs(args.save_model_path, exist_ok=True)
    writer = SummaryWriter(log_dir=args.save_model_path)

    # 2. 挂载数据流水线 (主脚本只负责下发指令给 Dataset)
    print("⏳ 正在加载训练和验证数据...")
    # dataset_train = NuScenesFlexiDataset(
    #     dataroot=args.data, 
    #     version=args.version, 
    #     split='train', 
    #     img_size=(args.crop_height, args.crop_width),
    #     seq_len=13  # 匹配 FlexiNet 对 13 帧深度的需求
    # )
    dataset_train = NuScenesFlexiDataset(
        dataroot=args.data, 
        version=args.version, 
        split='train', 
        img_size=(args.crop_height, args.crop_width),
        seq_len=5  # 匹配 FlexiNet 对 13 帧深度的需求
    )
    # dataset_val = NuScenesFlexiDataset(
    #     dataroot=args.data, 
    #     version=args.version, 
    #     split='val', 
    #     img_size=(args.crop_height, args.crop_width),
    #     seq_len=13
    # )
    dataset_val = NuScenesFlexiDataset(
        dataroot=args.data, 
        version=args.version, 
        split='val', 
        img_size=(args.crop_height, args.crop_width),
        seq_len=5
    )
    
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 3. 实例化网络模型、损失函数和优化器
    model = FlexiNet(input_channels=1, num_classes=1).to(device)
    model = model.float()
    print(f"🧠 模型参数量: {sum(p.numel() for p in model.parameters())}")

    criterion = nn.L1Loss() if args.loss_type == 'L1' else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)  # 🚀 优化1：增大 weight_decay，更强地限制模型过拟合
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)  # 🚀 优化2：余弦退火学习率，帮助模型更好地逃离局部最优
    
    # 混合精度训练，提升训练速度并节约显存
    scaler = torch.amp.GradScaler('cuda') if args.use_amp else None

    best_val_loss = float('inf')
    
    # 记录每个 epoch 的 loss，用于绘制并实时覆盖保存曲线图
    history_train_loss = []
    history_val_loss = []
    global_step = 0  # 记录总的 batch 步数，用于 TensorBoard 实时追踪

    # 4. 训练大循环
    print("🚀 开始训练...")
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        
        # 使用 tqdm 包装训练轮次，看起来更专业
        train_bar = tqdm(dataloader_train, desc=f"Epoch [{epoch+1}/{args.num_epochs}]", unit="batch")
        
        for img_batch, speed in train_bar:
            img_batch = img_batch.to(device)
            speed = speed.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=args.use_amp):
                predicted_speed = model(img_batch)
                loss = criterion(predicted_speed, speed.unsqueeze(1))
                
            if args.use_amp:
                scaler.scale(loss).backward()
                # 🚀 优化3：梯度裁剪 (配合AMP需先解缩放) 防止处理时序数据时梯度爆炸，稳定 val_loss 曲线
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # 🚀 优化3：普通模式下的梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                
            running_loss += loss.item()
            train_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
            # 将每个 batch 的 loss 实时写入 TensorBoard，实现秒级实时监控
            writer.add_scalar('Loss/Batch_Train', loss.item(), global_step)
            global_step += 1
            
        train_loss = running_loss / len(dataloader_train)
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 5. 执行验证并保存模型
        val_loss = validate(model, dataloader_val, criterion, device)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        scheduler.step()  # CosineAnnealingLR 不依赖 val_loss，每个 epoch 无条件 step()
        
        # 保存到历史记录中
        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        
        # 实时绘制并覆盖保存 Loss 曲线图 (保存在权重同一目录下)
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch + 2), history_train_loss, label='Train Loss', color='blue', marker='o', markersize=4)
        plt.plot(range(1, epoch + 2), history_val_loss, label='Validation Loss', color='red', marker='s', markersize=4)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss (Epoch {epoch + 1})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.save_model_path, 'loss_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图像防止内存泄漏

        print(f"👉 轮次汇总: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.save_model_path, 'nuscenes_mini_flexinet_0423.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"🎉 发现更优模型！已保存至 {best_model_path}")

    writer.close()
    print("✅ 训练全部完成！")

if __name__ == '__main__':
    # 预设常见数据集路径，如果有其他路径可以直接在这里添加
    DATA_PATHS = {
        'mini': r'D:/data/Master/01_Master_1/01_Research/09_SpeedChallenge_nuScenes/data/nuscenes_mini',
        'trainval': r'D:/data/Master/01_Master_1/01_Research/09_SpeedChallenge_nuScenes/data/nuscenes'
    }

    parser = argparse.ArgumentParser(description="nuScenes FlexiNet Training Pipeline")
    parser.add_argument('--dataset', type=str, choices=['mini', 'trainval', 'custom'], default='mini', help='选择预设的数据集: mini, trainval, custom')
    parser.add_argument('--data', type=str, default='', help='当 --dataset 为 custom 时，手动输入自定义的数据集路径(dataroot)')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='nuScenes的版本')
    parser.add_argument('--save_model_path', type=str, default='./log/nuscenes', help='保存日志和权重的路径')
    parser.add_argument('--num_epochs', type=int, default=20, help='训练总轮次')
    parser.add_argument('--batch_size', type=int, default=4, help='由于序列较长(13帧)，建议 Batch Size 小一点')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--crop_height', type=int, default=64)
    parser.add_argument('--crop_width', type=int, default=64)
    parser.add_argument('--loss_type', type=str, choices=['L1', 'L2'], default='L1')
    parser.add_argument('--use_amp', action='store_true', default=True, help='默认启用混合精度加速')
    
    args = parser.parse_args()

    # 根据用户选择的预设自动填充路径和版本号
    if args.dataset != 'custom':
        args.data = DATA_PATHS[args.dataset]
        if args.dataset == 'mini':
            args.version = 'v1.0-mini'
        elif args.dataset == 'trainval':
            args.version = 'v1.0-trainval'
            
    if not args.data:
        raise ValueError("❌ 未找到有效的数据集路径！请指定预设(--dataset mini)或使用自定义路径(--dataset custom --data [你的路径])")
        
    print(f"📂 自动挂载数据集: {args.data} (使用版本: {args.version})")

    train_model(args)
