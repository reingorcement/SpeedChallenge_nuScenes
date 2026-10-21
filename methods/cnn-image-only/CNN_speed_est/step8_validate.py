import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from nuscenes.nuscenes import NuScenes
import matplotlib
import os
from PIL import Image
from torchvision import transforms

try:
    from nuimages import NuImages
except ImportError:
    from nuimages.nuimages import NuImages

# 使用新的场景进行测试
# 设置中文字体，防止画图时中文显示为方块
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from step5_data_pipeline import NuScenesSpeedDataset
from step6_comma_model import CommaSpeedModel


class NuImagesSpeedDataset(Dataset):
    def __init__(self, nuim, extra_data_roots=None):
        self.nuim = nuim
        self.frame_pairs = []
        self.extra_data_roots = extra_data_roots or []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self._build_pairs()

    def _is_cam_front(self, sample_data):
        try:
            calib = self.nuim.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
            sensor = self.nuim.get('sensor', calib['sensor_token'])
            return sensor.get('channel', '') == 'CAM_FRONT'
        except Exception:
            return 'CAM_FRONT' in sample_data.get('filename', '').upper()

    def _build_pairs(self):
        for sample_data in self.nuim.sample_data:
            if not self._is_cam_front(sample_data):
                continue

            next_token = sample_data.get('next', '')
            if not next_token:
                continue

            try:
                next_sample_data = self.nuim.get('sample_data', next_token)
            except Exception:
                continue

            if not self._is_cam_front(next_sample_data):
                continue

            self.frame_pairs.append((sample_data['token'], next_token))

    def __len__(self):
        return len(self.frame_pairs)

    def _get_image_path(self, sample_data):
        rel_path = sample_data['filename']
        candidate_paths = [os.path.join(self.nuim.dataroot, rel_path)]
        candidate_paths.extend(os.path.join(root, rel_path) for root in self.extra_data_roots)

        for path in candidate_paths:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f"找不到图像文件: {rel_path}。已检查路径: {candidate_paths}"
        )

    def _compute_speed(self, sample_data_1, sample_data_2):
        try:
            pose1 = self.nuim.get('ego_pose', sample_data_1['ego_pose_token'])
            pose2 = self.nuim.get('ego_pose', sample_data_2['ego_pose_token'])
            x1, y1 = pose1['translation'][0], pose1['translation'][1]
            x2, y2 = pose2['translation'][0], pose2['translation'][1]
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            dt = (pose2['timestamp'] - pose1['timestamp']) / 1e6
            return distance / dt if dt > 0 else 0.0
        except Exception:
            return 0.0

    def __getitem__(self, idx):
        token1, token2 = self.frame_pairs[idx]
        sample_data_1 = self.nuim.get('sample_data', token1)
        sample_data_2 = self.nuim.get('sample_data', token2)

        img1 = Image.open(self._get_image_path(sample_data_1)).convert('RGB')
        img2 = Image.open(self._get_image_path(sample_data_2)).convert('RGB')

        tensor1 = self.transform(img1)
        tensor2 = self.transform(img2)
        stacked_imgs = torch.cat((tensor1, tensor2), dim=0)

        speed = self._compute_speed(sample_data_1, sample_data_2)
        speed_tensor = torch.tensor([speed], dtype=torch.float32)

        return stacked_imgs, speed_tensor

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"验证设备: {device}")

    # 1. 准备验证数据（切换为 nuImages）
    print("正在加载 nuImages 验证集数据...")

    # 原 nuScenes 验证集代码（按要求注释保留）
    # nusc = NuScenes(
    #     version='v1.0-mini',
    #     dataroot='D:/data/Master/01_Master_1/01_Research/09_SpeedChallenge_nuScenes/data/nuscenes_mini',
    #     verbose=False
    # )
    # val_dataset = NuScenesSpeedDataset(nusc, is_train=False)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    nuimages_root = 'D:/data/Master/01_Master_1/01_Research/09_SpeedChallenge_nuScenes/data/nuimages'

    # 兼容 nuImages 分包目录：metadata 和 samples/sweeps 不在同一层
    metadata_root = nuimages_root
    packaged_metadata_root = os.path.join(nuimages_root, 'nuimages-v1.0-all-metadata')
    if os.path.isdir(packaged_metadata_root):
        metadata_root = packaged_metadata_root

    preferred_versions = ['v1.0-mini', 'v1.0-val', 'v1.0-train', 'v1.0-test']
    available_versions = [v for v in preferred_versions if os.path.isdir(os.path.join(metadata_root, v))]
    if not available_versions:
        print(f"❌ 在 {metadata_root} 下未找到可用 nuImages 版本目录。")
        print("请确认至少存在 v1.0-mini / v1.0-val / v1.0-train / v1.0-test 之一。")
        return

    selected_version = available_versions[0]
    print(f"使用 nuImages 版本: {selected_version}")

    extra_data_roots = []
    packaged_samples_root = os.path.join(nuimages_root, 'nuimages-v1.0-all-samples')
    packaged_sweeps_root = os.path.join(nuimages_root, 'nuimages-v1.0-all-sweeps-cam-front')
    if os.path.isdir(packaged_samples_root):
        extra_data_roots.append(packaged_samples_root)
    if os.path.isdir(packaged_sweeps_root):
        extra_data_roots.append(packaged_sweeps_root)

    try:
        nuim = NuImages(version=selected_version, dataroot=metadata_root, verbose=False, lazy=True)
    except TypeError:
        nuim = NuImages(version=selected_version, dataroot=metadata_root, verbose=False)

    val_dataset = NuImagesSpeedDataset(nuim, extra_data_roots=extra_data_roots)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(f"nuImages 验证样本数: {len(val_dataset)}")

    # 2. 加载“注入了灵魂”的模型
    model = CommaSpeedModel().to(device)
    try:
        model.load_state_dict(torch.load('speed_model_v1.pth', map_location=device))
        print("✅ 成功加载预训练权重 'speed_model_v1.pth'")
    except FileNotFoundError:
        print("❌ 找不到权重文件，请确保你完整运行了第 7 阶段！")
        return

    # 3. 切换到评估(推理)模式！非常关键！
    model.eval()

    true_speeds_list = []
    pred_speeds_list = []

    print("开始进行车速预测...")
    # 4. 关闭梯度计算，节省算力
    with torch.no_grad():
        for images, true_speeds in val_loader:
            images = images.to(device)
            
            # 模型给出预测
            preds = model(images)
            
            # 把 GPU 里的张量数据转移到 CPU 上，并转成普通的 Python 数值 (item)
            true_speeds_list.append(true_speeds.item())
            pred_speeds_list.append(preds.item())

    # 5. 画图对比 (这就和你在 MATLAB 里 plot 曲线一样)
    print("预测完毕，正在生成对比曲线图...")
    max_plot_frames = 80
    plot_true_speeds = true_speeds_list[:max_plot_frames]
    plot_pred_speeds = pred_speeds_list[:max_plot_frames]

    plt.figure(figsize=(10, 5))
    plt.plot(plot_true_speeds, label='真实车速 (Ground Truth)', color='blue', linewidth=2, marker='o', markersize=4)
    plt.plot(plot_pred_speeds, label='预测车速 (Prediction)', color='red', linestyle='--', linewidth=2)
    
    plt.title("基于图像的单目车速估计结果对比")
    plt.xlabel("时间帧序列 (Frames)")
    plt.ylabel("速度 (m/s)")
    plt.legend()
    plt.grid(True)
    output_plot_path = 'validate_result_top80.png'
    plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
    print(f"已保存前 {max_plot_frames} 帧结果图到: {output_plot_path}")

    # 6. 计算并输出最终 MAE 和 RMSE
    if len(true_speeds_list) > 0:
        errors = [pred - true for pred, true in zip(pred_speeds_list, true_speeds_list)]
        mae = sum(abs(err) for err in errors) / len(errors)
        rmse = (sum(err ** 2 for err in errors) / len(errors)) ** 0.5
        print(f"最终 MAE: {mae:.4f} m/s")
        print(f"最终 RMSE: {rmse:.4f} m/s")
    else:
        print("验证集为空，无法计算 MAE 和 RMSE。")
    
    # 弹出图像窗口
    plt.show()

if __name__ == "__main__":
    main()