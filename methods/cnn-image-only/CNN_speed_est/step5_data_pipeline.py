import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import math
from nuscenes.nuscenes import NuScenes

class NuScenesSpeedDataset(Dataset):
    def __init__(self, nusc, is_train=True):
        self.nusc = nusc
        # 【改动点 1】不再存储 sample，而是存储相机的连续帧 token 列表
        self.cam_tokens = [] 
        
        scene_indices = range(0, 8) if is_train else range(8, 10)
        
        for idx in scene_indices:
            scene = self.nusc.scene[idx]
            # 找到这个场景的第一个样本 (2Hz 关键帧)
            first_sample = self.nusc.get('sample', scene['first_sample_token'])
            
            # 【改动点 2】顺藤摸瓜：直接拿到第一个样本里的前置摄像头数据 token
            current_cam_token = first_sample['data']['CAM_FRONT']
            
            # 【改动点 3】沿着摄像头的专属时间线往后遍历（包含所有 12Hz 的 sweeps 和 samples）
            while current_cam_token != '':
                self.cam_tokens.append(current_cam_token)
                # 获取当前这帧相机数据
                cam_data = self.nusc.get('sample_data', current_cam_token)
                # 指向摄像头的下一帧！
                current_cam_token = cam_data['next'] 
                
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor() 
        ])

    def __len__(self):
        # 长度依然是总帧数减 1
        return len(self.cam_tokens) - 1

    def __getitem__(self, idx):
        # 【改动点 4】直接拿连续的两帧相机 token
        token1 = self.cam_tokens[idx]
        token2 = self.cam_tokens[idx + 1]

        # 提取真实传感器记录
        cam1_data = self.nusc.get('sample_data', token1)
        cam2_data = self.nusc.get('sample_data', token2)

        # 读取图像 (nusc.get_sample_data_path 会自动去 samples 或 sweeps 里找正确的文件)
        img1 = Image.open(self.nusc.get_sample_data_path(token1))
        img2 = Image.open(self.nusc.get_sample_data_path(token2))
        
        tensor1 = self.transform(img1)
        tensor2 = self.transform(img2)
        stacked_imgs = torch.cat((tensor1, tensor2), dim=0)

        # 提取 Ego Pose 计算车速 (逻辑完全不用动，因为自带高精度时间戳)
        pose1 = self.nusc.get('ego_pose', cam1_data['ego_pose_token'])
        pose2 = self.nusc.get('ego_pose', cam2_data['ego_pose_token'])

        x1, y1 = pose1['translation'][0], pose1['translation'][1]
        x2, y2 = pose2['translation'][0], pose2['translation'][1]
        
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        dt = (pose2['timestamp'] - pose1['timestamp']) / 1e6
        speed = distance / dt if dt > 0 else 0.0
        
        speed_tensor = torch.tensor([speed], dtype=torch.float32)

        return stacked_imgs, speed_tensor

# --- 测试流水线是否升级成功 ---
if __name__ == "__main__":
    print("正在加载 nuScenes 数据库...")
    nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes_mini', verbose=False)
    my_dataset = NuScenesSpeedDataset(nusc, is_train=True)
    
    # 打印看看数据量是不是从 300 多暴增到了 2000 多！
    print(f"成功构建 12Hz 数据集，共提取了 {len(my_dataset)} 组连续高频帧样本！")