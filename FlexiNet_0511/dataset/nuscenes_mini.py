import os
import math
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from nuscenes.nuscenes import NuScenes

# 这个文件定义了一个专门为 FlexiNet 定制的 nuScenes_mini 数据集类 NuScenesFlexiDataset。
# 它自动提取连续的 seq_len(默认13) 帧序列，输出尺寸适配 3D 卷积，并计算对应的速度标签。

class NuScenesFlexiDataset(Dataset):
    def __init__(self, dataroot, version='v1.0-mini', split='train', img_size=(64, 64), seq_len=13):
        """
        专门为 FlexiNet 定制的 nuScenes 数据集类。
        自动提取连续的 seq_len(默认13) 帧序列，输出尺寸适配 3D 卷积。
        """
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.seq_len = seq_len
        self.cam_tokens = []
        
        # 划分训练集和验证集 (以 mini 数据集的 10 个场景为例：0-7 训练，8-9 验证)
        scene_indices = range(0, 8) if split == 'train' else range(8, 10)
        
        for idx in scene_indices:
            if idx >= len(self.nusc.scene):
                break
            scene = self.nusc.scene[idx]
            first_sample = self.nusc.get('sample', scene['first_sample_token'])
            
            # 我们依然提取前置摄像头数据
            current_cam_token = first_sample['data']['CAM_FRONT']
            
            # 收集该场景下所有相机的连续 token
            scene_tokens = []
            while current_cam_token != '':
                scene_tokens.append(current_cam_token)
                cam_data = self.nusc.get('sample_data', current_cam_token)
                current_cam_token = cam_data.get('next', '')
            
            # 使用滑动窗口，每次截取连续的 seq_len 帧作为一个样本序列
            # 比如 13 帧，这样就能适配 FlexiNet 的 3D 卷积时间深度
            for i in range(len(scene_tokens) - self.seq_len + 1):
                self.cam_tokens.append(scene_tokens[i : i + self.seq_len])

        # 定义图像变换：缩放、转灰度图 (FlexiNet 默认 1 通道)、归一化
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        print(f"[{split.upper()}] 成功构建 nuScenes 数据集，提取了 {len(self.cam_tokens)} 组序列(每组 {seq_len} 帧)")

    def __len__(self):
        return len(self.cam_tokens)

    def __getitem__(self, idx):
        # 获取当前序列的 token 列表 (例如 13 个 token)
        tokens = self.cam_tokens[idx]
        
        frames = []
        for token in tokens:
            img_path = self.nusc.get_sample_data_path(token)
            # 转为灰度图，适配 FlexiNet
            img = Image.open(img_path).convert('L')
            tensor = self.transform(img) # Shape: (1, H, W)
            frames.append(tensor)
            
        # 沿着时间维度堆叠所有帧 -> Shape: (13, 1, H, W)
        # 这个形状完美契合 FlexiNet 的 3D 网络输入
        stacked_frames = torch.stack(frames)
        
        # 计算速度：我们用序列中心时刻 (例如第 7 帧) 的速度作为标签
        center_idx = self.seq_len // 2
        token1 = tokens[center_idx]
        token2 = tokens[center_idx + 1] # 用中心帧和下一帧计算速度
        
        cam1_data = self.nusc.get('sample_data', token1)
        cam2_data = self.nusc.get('sample_data', token2)
        
        pose1 = self.nusc.get('ego_pose', cam1_data['ego_pose_token'])
        pose2 = self.nusc.get('ego_pose', cam2_data['ego_pose_token'])
        
        x1, y1 = pose1['translation'][0], pose1['translation'][1]
        x2, y2 = pose2['translation'][0], pose2['translation'][1]
        
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        dt = (pose2['timestamp'] - pose1['timestamp']) / 1e6
        speed = distance / dt if dt > 0 else 0.0
        
        # 返回 (特征, 标签)
        return stacked_frames.float(), torch.tensor(speed, dtype=torch.float32)