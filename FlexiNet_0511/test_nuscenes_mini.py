# import os
# import argparse
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import random
# import numpy as np

# # 导入画图库
# import matplotlib.pyplot as plt

# # 引入我们自定义的 Dataset 和 FlexiNet 模型
# from dataset.nuscenes_mini import NuScenesFlexiDataset
# from model.FlexiNet import FlexiNet

# def set_seed(seed_value):
#     """固定随机种子以保证测试结果的可复现性"""
#     random.seed(seed_value)
#     np.random.seed(seed_value)
#     torch.manual_seed(seed_value)
#     torch.cuda.manual_seed_all(seed_value)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def load_model(model, checkpoint_path, device):
#     """加载模型权重"""
#     if not os.path.isfile(checkpoint_path):
#         raise FileNotFoundError(f"❌ 找不到权重文件: {checkpoint_path}")

#     print(f"📦 正在加载模型权重: {checkpoint_path}")
#     checkpoint = torch.load(checkpoint_path, map_location=device)

#     if 'state_dict' in checkpoint:
#         model.load_state_dict(checkpoint['state_dict'])
#     elif isinstance(checkpoint, dict):
#         model.load_state_dict(checkpoint)
#     else:
#         raise KeyError("无法识别的 checkpoint 格式。")
    
#     print("✅ 权重加载成功！")
#     return model

# def test_model(args, model, dataloader_test, device):
#     """
#     执行测试，计算平均损失，并收集预测数据用于画图
#     """
#     model.eval()
#     running_test_loss = 0.0
#     eps = 1e-6  # 用于 L2 损失的数值稳定性
    
#     # 🌟 新增：用于收集真实车速和预测车速的列表
#     true_speeds_list = []
#     pred_speeds_list = []
    
#     if args.loss_type == 'L1':
#         criterion = nn.L1Loss()
#     elif args.loss_type == 'L2':
#         criterion = nn.MSELoss()
#     else:
#         raise ValueError("无效的损失类型，请选择 'L1' 或 'L2'")

#     with torch.no_grad():
#         test_bar = tqdm(dataloader_test, desc="正在测试", unit="batch")
#         for data in test_bar:
#             img_batch, speed = data
#             img_batch = img_batch.to(device).float()
#             speed = speed.to(device).float()
            
#             predicted_speed = model(img_batch)
            
#             # 🌟 新增：将 tensor 转换为 numpy 数组并存入 list 中
#             # .squeeze() 去除多余维度，.cpu().numpy() 转移到内存，.tolist() 转为原生列表
#             true_speeds_list.extend(speed.cpu().numpy().tolist())
#             pred_speeds_list.extend(predicted_speed.squeeze().cpu().numpy().tolist())
            
#             if args.loss_type == 'L1':
#                 loss = criterion(predicted_speed, speed.unsqueeze(1))
#             elif args.loss_type == 'L2':
#                 mse_loss = criterion(predicted_speed, speed.unsqueeze(1))
#                 loss = torch.sqrt(mse_loss + eps)
                
#             running_test_loss += loss.item()
#             test_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

#     average_test_loss = running_test_loss / len(dataloader_test)
    
#     # 返回损失值的同时，一并返回车速列表
#     return average_test_loss, true_speeds_list, pred_speeds_list

# def main(args):
#     # 1. 设备与环境初始化
#     set_seed(42)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"🔥 使用计算设备: {device}")

#     # 2. 挂载数据流水线
#     print(f"⏳ 正在加载测试数据 (分割集: {args.split})...")
#     dataset_test = NuScenesFlexiDataset(
#         dataroot=args.data, 
#         version=args.version, 
#         split=args.split, 
#         img_size=(args.crop_height, args.crop_width),
#         seq_len=13 
#     )
    
#     dataloader_test = DataLoader(
#         dataset_test, 
#         batch_size=args.batch_size, 
#         shuffle=False,  # 测试集严格按照顺序读取，不要打乱，否则画出来的图是乱飞的！
#         num_workers=args.num_workers
#     )

#     # 3. 实例化网络模型并加载权重
#     model = FlexiNet(input_channels=1, num_classes=1).to(device)
#     model = model.float()
#     print(f"🧠 模型参数量: {sum(p.numel() for p in model.parameters())}")
#     model = load_model(model, args.checkpoint_path, device)

#     # 4. 开始测试
#     print(f"🚀 开始在 {args.split} 集合上进行评估，评估指标: {args.loss_type} ...")
#     test_loss, true_speeds_list, pred_speeds_list = test_model(args, model, dataloader_test, device)
    
#     print(f"🎉 测试完成！")
#     print(f"📊 最终平均测试误差 ({args.loss_type}): {test_loss:.4f}")

#     # ==========================================
#     # 5. 画图对比 (完全参考你提供的图片样式)
#     # ==========================================
#     print("📈 预测完毕，正在生成对比曲线图...")
    
#     # 设置 Windows 下的中文字体（黑体），防止中文显示为方块
#     plt.rcParams['font.sans-serif'] = ['SimHei'] 
#     plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    
#     # 如果验证集数据太多（比如上千帧），曲线会糊成一团。这里为了可视化美观，可以选择只画前 200 帧
#     # 如果想画所有的，把 [:200] 删掉即可。
#     plot_num = min(200, len(true_speeds_list)) 
#     #plot_num = min(len(true_speeds_list))
#     y_true = true_speeds_list[:plot_num]
#     y_pred = pred_speeds_list[:plot_num]

#     plt.figure(figsize=(10, 5))
#     plt.plot(y_true, label='真实车速 (Ground Truth)', color='blue', linewidth=2, marker='o', markersize=4)
#     plt.plot(y_pred, label='预测车速 (Prediction)', color='red', linestyle='--', linewidth=2)

#     plt.title("基于图像的单目车速估计结果对比")
#     plt.xlabel("时间帧序列 (Frames)")
#     plt.ylabel("速度 (m/s)")
#     plt.legend()
#     plt.grid(True)

#     # 顺手帮你加个保存图片的功能，方便贴到论文或报告里
#     save_fig_path = os.path.join(os.path.dirname(args.checkpoint_path), "speed_comparison.png")
#     plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
#     print(f"💾 图像已保存至: {save_fig_path}")

#     # 弹出图像窗口
#     plt.show()


# if __name__ == '__main__':
#     # 预设常见数据集路径
#     DATA_PATHS = {
#         'mini': r'D:/data/Master/01_Master_1/01_Research/09_SpeedChallenge_nuScenes/data/nuscenes_mini',
#         'trainval': r'D:/data/Master/01_Master_1/01_Research/09_SpeedChallenge_nuScenes/data/nuscenes'
#     }

#     parser = argparse.ArgumentParser(description="nuScenes FlexiNet Testing Pipeline")
#     parser.add_argument('--dataset', type=str, choices=['mini', 'trainval', 'custom'], default='mini', help='选择预设的数据集: mini, trainval, custom')
#     parser.add_argument('--data', type=str, default='', help='自定义的数据集路径')
#     parser.add_argument('--version', type=str, default='v1.0-mini', help='nuScenes的版本')
#     parser.add_argument('--split', type=str, default='val', help='测试集名称')
#     parser.add_argument('--checkpoint_path', type=str, default='./log/nuscenes/nuscenes_mini_flexinet_0423.pth', help='待测试的模型权重文件路径 (.pth)')
    
#     parser.add_argument('--batch_size', type=int, default=4, help='测试时的 Batch Size')
#     parser.add_argument('--num_workers', type=int, default=4)
#     parser.add_argument('--crop_height', type=int, default=64)
#     parser.add_argument('--crop_width', type=int, default=64)
#     parser.add_argument('--loss_type', type=str, choices=['L1', 'L2'], default='L1', help='评估指标: L1(MAE) 或 L2(RMSE)')
    
#     args = parser.parse_args()

#     if args.dataset != 'custom':
#         args.data = DATA_PATHS[args.dataset]
#         if args.dataset == 'mini':
#             args.version = 'v1.0-mini'
#         elif args.dataset == 'trainval':
#             args.version = 'v1.0-trainval'
            
#     if not args.data:
#         raise ValueError("❌ 未找到有效的数据集路径！")
        
#     main(args)
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np

# 导入画图库
import matplotlib.pyplot as plt

# 引入我们自定义的 Dataset 和 FlexiNet 模型
from dataset.nuscenes_mini import NuScenesFlexiDataset
from model.FlexiNet import FlexiNet

def set_seed(seed_value):
    """固定随机种子以保证测试结果的可复现性"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(model, checkpoint_path, device):
    """加载模型权重"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"❌ 找不到权重文件: {checkpoint_path}")

    print(f"📦 正在加载模型权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise KeyError("无法识别的 checkpoint 格式。")
    
    print("✅ 权重加载成功！")
    return model

def test_model(args, model, dataloader_test, device):
    """
    执行测试，计算平均损失，并收集预测数据用于画图
    """
    model.eval()
    running_test_loss = 0.0
    eps = 1e-6  # 用于 L2 损失的数值稳定性
    
    # 用于收集真实车速和预测车速的列表
    true_speeds_list = []
    pred_speeds_list = []
    
    if args.loss_type == 'L1':
        criterion = nn.L1Loss()
    elif args.loss_type == 'L2':
        criterion = nn.MSELoss()
    else:
        raise ValueError("无效的损失类型，请选择 'L1' 或 'L2'")

    with torch.no_grad():
        test_bar = tqdm(dataloader_test, desc="正在测试", unit="batch")
        for data in test_bar:
            img_batch, speed = data
            img_batch = img_batch.to(device).float()
            speed = speed.to(device).float()
            
            predicted_speed = model(img_batch)
            
            # 将 tensor 转换为 numpy 数组并存入 list 中
            # .squeeze() 去除多余维度，.cpu().numpy() 转移到内存，.tolist() 转为原生列表
            true_speeds_list.extend(speed.cpu().numpy().tolist())
            pred_speeds_list.extend(predicted_speed.squeeze().cpu().numpy().tolist())
            
            if args.loss_type == 'L1':
                loss = criterion(predicted_speed, speed.unsqueeze(1))
            elif args.loss_type == 'L2':
                mse_loss = criterion(predicted_speed, speed.unsqueeze(1))
                loss = torch.sqrt(mse_loss + eps)
                
            running_test_loss += loss.item()
            test_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    average_test_loss = running_test_loss / len(dataloader_test)
    
    # 返回损失值的同时，一并返回车速列表
    return average_test_loss, true_speeds_list, pred_speeds_list

def main(args):
    # 1. 设备与环境初始化
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 使用计算设备: {device}")

    # 2. 挂载数据流水线
    print(f"⏳ 正在加载测试数据 (分割集: {args.split})...")
    dataset_test = NuScenesFlexiDataset(
        dataroot=args.data, 
        version=args.version, 
        split=args.split, 
        img_size=(args.crop_height, args.crop_width),
        seq_len=13 
    )
    
    dataloader_test = DataLoader(
        dataset_test, 
        batch_size=args.batch_size, 
        shuffle=False,  # 测试集严格按照顺序读取，不要打乱，否则画出来的图是乱飞的！
        num_workers=args.num_workers
    )

    # 3. 实例化网络模型并加载权重
    model = FlexiNet(input_channels=1, num_classes=1).to(device)
    model = model.float()
    print(f"🧠 模型参数量: {sum(p.numel() for p in model.parameters())}")
    model = load_model(model, args.checkpoint_path, device)

    # 4. 开始测试
    print(f"🚀 开始在 {args.split} 集合上进行评估，评估指标: {args.loss_type} ...")
    test_loss, true_speeds_list, pred_speeds_list = test_model(args, model, dataloader_test, device)
    
    # ==========================================
    # 🌟 新增功能：计算整体的 MAE 和 RMSE
    # ==========================================
    # 将列表转换为 numpy 数组以便进行向量化计算
    true_speeds_arr = np.array(true_speeds_list)
    pred_speeds_arr = np.array(pred_speeds_list)
    
    # 计算 MAE
    final_mae = np.mean(np.abs(true_speeds_arr - pred_speeds_arr))
    # 计算 RMSE
    final_rmse = np.sqrt(np.mean((true_speeds_arr - pred_speeds_arr) ** 2))

    print(f"🎉 测试完成！")
    print(f"📊 模型 Loss ({args.loss_type}): {test_loss:.4f}")
    print(f"🎯 最终 MAE (平均绝对误差): {final_mae:.4f} m/s")
    print(f"🎯 最终 RMSE (均方根误差): {final_rmse:.4f} m/s")
    # ==========================================

    # ==========================================
    # 5. 画图对比
    # ==========================================
    print("📈 预测完毕，正在生成对比曲线图...")
    
    # 设置 Windows 下的中文字体（黑体），防止中文显示为方块
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    
    # 如果验证集数据太多（比如上千帧），曲线会糊成一团。这里为了可视化美观，可以选择只画前 200 帧
    plot_num = min(200, len(true_speeds_list)) 
    y_true = true_speeds_list[:plot_num]
    y_pred = pred_speeds_list[:plot_num]

    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='真实车速 (Ground Truth)', color='blue', linewidth=2, marker='o', markersize=4)
    plt.plot(y_pred, label='预测车速 (Prediction)', color='red', linestyle='--', linewidth=2)

    # 🌟 可选：在图表标题中展示 MAE 和 RMSE，方便截图放进论文
    plt.title(f"基于图像的单目车速估计结果对比\n(Overall MAE: {final_mae:.2f} m/s, RMSE: {final_rmse:.2f} m/s)")
    plt.xlabel("时间帧序列 (Frames)")
    plt.ylabel("速度 (m/s)")
    plt.legend()
    plt.grid(True)

    # 顺手帮你加个保存图片的功能，方便贴到论文或报告里
    save_fig_path = os.path.join(os.path.dirname(args.checkpoint_path), "speed_comparison.png")
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    print(f"💾 图像已保存至: {save_fig_path}")

    # 弹出图像窗口
    plt.show()

if __name__ == '__main__':
    # 预设常见数据集路径
    DATA_PATHS = {
        'mini': r'D:/data/Master/01_Master_1/01_Research/09_SpeedChallenge_nuScenes/data/nuscenes_mini',
        'trainval': r'D:/data/Master/01_Master_1/01_Research/09_SpeedChallenge_nuScenes/data/nuscenes'
    }

    parser = argparse.ArgumentParser(description="nuScenes FlexiNet Testing Pipeline")
    parser.add_argument('--dataset', type=str, choices=['mini', 'trainval', 'custom'], default='mini', help='选择预设的数据集: mini, trainval, custom')
    parser.add_argument('--data', type=str, default='', help='自定义的数据集路径')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='nuScenes的版本')
    parser.add_argument('--split', type=str, default='val', help='测试集名称')
    parser.add_argument('--checkpoint_path', type=str, default='./log/nuscenes/nuscenes_mini_flexinet_0423.pth', help='待测试的模型权重文件路径 (.pth)')
    
    parser.add_argument('--batch_size', type=int, default=4, help='测试时的 Batch Size')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--crop_height', type=int, default=64)
    parser.add_argument('--crop_width', type=int, default=64)
    parser.add_argument('--loss_type', type=str, choices=['L1', 'L2'], default='L1', help='评估指标: L1(MAE) 或 L2(RMSE)')
    
    args = parser.parse_args()

    if args.dataset != 'custom':
        args.data = DATA_PATHS[args.dataset]
        if args.dataset == 'mini':
            args.version = 'v1.0-mini'
        elif args.dataset == 'trainval':
            args.version = 'v1.0-trainval'
            
    if not args.data:
        raise ValueError("❌ 未找到有效的数据集路径！")
        
    main(args)