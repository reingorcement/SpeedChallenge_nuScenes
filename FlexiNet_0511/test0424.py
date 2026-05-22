import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt  # 🌟 新增：画图库

from dataset.nuimages import NuImagesDataset
from dataset.kitti import KITTIDataset
from model.FlexiNet import FlexiNet
from torch.utils.tensorboard import SummaryWriter
import sys

def set_seed(seed_value):
    """
    Set seed for reproducibility across various libraries and frameworks.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)          
    torch.cuda.manual_seed_all(seed_value)      
    torch.backends.cudnn.deterministic = True    
    torch.backends.cudnn.benchmark = False       


def test(args, model, dataloader_test):
    """
    执行测试，收集预测数据，并在整个测试集上计算全局的 MAE 和 RMSE
    """
    model.eval()  
    
    true_speeds_list = []
    pred_speeds_list = []

    with torch.no_grad():  
        for data in tqdm(dataloader_test, desc="Testing", unit="batch"):
            img_batch, speed = data
            # 确保数据格式统一为 float
            img_batch = img_batch.to(args.device).float()
            speed = speed.to(args.device).float()
            
            predicted_speed = model(img_batch)
            
            # 收集数据：.flatten() 确保维度拍平，.cpu().numpy() 转到内存
            true_speeds_list.extend(speed.cpu().numpy().flatten().tolist())
            pred_speeds_list.extend(predicted_speed.cpu().numpy().flatten().tolist())

    # 将列表转换为 numpy 数组，方便进行向量化数学计算
    y_true = np.array(true_speeds_list)
    y_pred = np.array(pred_speeds_list)

    # 🌟 新增：全局计算 MAE 和 RMSE
    # MAE = Mean Absolute Error (平均绝对误差)
    mae = np.mean(np.abs(y_true - y_pred))
    # RMSE = Root Mean Square Error (均方根误差)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    return mae, rmse, true_speeds_list, pred_speeds_list


def load_model(model, checkpoint_path, device):
    """
    Load the model checkpoint.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"📦 正在加载模型权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise KeyError("'state_dict' key not found in checkpoint file.")
    
    print("✅ 权重加载成功！")
    return model


def main(params):
    set_seed(0)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Testing Pipeline for FlexiNet")
    parser.add_argument('--data', type=str, default='D:/data/Master/01_Master_1/01_Research/09_SpeedChallenge_nuScenes/data/nuimages', help='Path of test data')
    parser.add_argument('--dataset', type=str, default="nuImages", help='Dataset to use: Kitti, nuImages')
    parser.add_argument('--crop_height', type=int, default=64, help='Height of cropped input image to network')
    parser.add_argument('--crop_width', type=int, default=64, help='Width of cropped input image to network')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--cuda', type=str, default='0', help='GPU id(s) to use for testing')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Whether to use GPU for testing')
    parser.add_argument('--checkpoint_path', type=str, default='./pretrained_models/checkpoint_epoch_100_nuimage_L1_best.pth', help='Path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default='./log/nuimages_flexinet', help='Path to save the model')
    parser.add_argument('--plot_frames', type=int, default=80, help='Number of frames to draw in final comparison plot')
    # 测试代码里其实不需要 loss_type 参数了，因为我们会同时算出 MAE 和 RMSE，但我保留了它防止你在命令行传参时报错
    parser.add_argument('--loss_type', type=str, choices=['L1', 'L2'], default='L1', help='Loss function type: L1 or L2')

    args = parser.parse_args(params)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"🔥 Using device: {args.device}")

    # Load the appropriate dataset
    dataset_name = args.dataset.lower()
    if dataset_name == 'kitti':
        dataset_test = KITTIDataset(
            dataroot=args.data,
            split='valid',
            img_size=(args.crop_height, args.crop_width),
        )
    elif dataset_name == 'nuimages':
        dataset_test = NuImagesDataset(
            dataroot=args.data,
            version='v1.0-test',
            img_size=(args.crop_height, args.crop_width),
            subset=False
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Choose from 'Kitti', 'nuImages'.")

    if os.name == 'nt' and dataset_name == 'nuimages' and args.num_workers > 0:
        print("⚠️ Detected Windows + nuImages. Overriding num_workers to 0 to avoid multiprocessing recursion errors.")
        args.num_workers = 0

    # 务必保证 shuffle=False，否则画出来的折线图是乱飞的
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    model = FlexiNet(input_channels=1, num_classes=1)

    if torch.cuda.device_count() > 1 and args.use_gpu:
        model = nn.DataParallel(model)
        
    model = model.to(args.device)
    model = model.float() # 保险起见，统一精度
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Load pretrained model
    model = load_model(model, args.checkpoint_path, args.device)

    # ==========================================
    # 执行测试流程
    # ==========================================
    print(f"🚀 开始在 {args.dataset} 集合上进行评估...")
    mae, rmse, true_speeds_list, pred_speeds_list = test(args, model, dataloader_test)
    
    print("🎉 测试完成！")
    print("=" * 40)
    print(f"📊 整体测试指标 -> MAE : {mae:.4f}")
    print(f"📊 整体测试指标 -> RMSE: {rmse:.4f}")
    print("=" * 40)

    # ==========================================
    # 🌟 新增：画图对比功能
    # ==========================================
    print("📈 正在生成对比曲线图...")
    
    # 设置 Windows 字体，防止中文乱码
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 

    # 仅截取前 N 帧画图，默认前 80 帧，避免点太多导致曲线难以阅读
    plot_num = min(max(args.plot_frames, 1), len(true_speeds_list))
    y_true_plot = true_speeds_list[:plot_num]
    y_pred_plot = pred_speeds_list[:plot_num]

    plt.figure(figsize=(10, 5))
    plt.plot(y_true_plot, label='真实车速 (Ground Truth)', color='blue', linewidth=2, marker='o', markersize=4)
    plt.plot(y_pred_plot, label='预测车速 (Prediction)', color='red', linestyle='--', linewidth=2)

    plt.title(f"基于图像的单目车速估计结果对比 (截取前{plot_num}帧)")
    plt.xlabel("时间帧序列 (Frames)")
    plt.ylabel("速度 (m/s)")
    plt.legend()
    plt.grid(True)

    # 保存图片
    save_fig_path = os.path.join(os.path.dirname(args.checkpoint_path), f"speed_comparison_{args.dataset}.png")
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    print(f"💾 图像已自动保存至: {save_fig_path}")

    # 弹出窗口
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])