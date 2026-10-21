from nuscenes.nuscenes import NuScenes

# 1. 初始化 nuScenes (加载 JSON 数据库)
# dataroot 是你解压的数据集所在路径，version 必须匹配你下载的版本
print("正在加载 nuScenes 数据库，请稍候...")
nusc = NuScenes(version='v1.0-mini', dataroot=r'D:\data\Master\01_Master_1\01_Research\09_SpeedChallenge_nuScenes\data\nuscenes_mini', verbose=True)

# 2. 拿到数据集里的第一个“场景” (Scene)
# 一个 Scene 通常是持续 20 秒左右的一段连续驾驶记录
my_scene = nusc.scene[0]
print(f"\n✅ 成功读取场景！场景描述: {my_scene['description']}")

# 3. 拿到这个场景下的第一个“样本” (Sample，相当于某一时刻所有传感器的快照)
first_sample_token = my_scene['first_sample_token']
my_sample = nusc.get('sample', first_sample_token)


# 4. 看看这个样本包含了哪些传感器的数据
print(f"\n📷 该时刻包含的传感器数据列表:")
for sensor_name in my_sample['data'].keys():
    print(f" - {sensor_name}")

# 5. 顺藤摸瓜：获取前置摄像头 (CAM_FRONT) 拍照那一刻的自车状态
cam_front_token = my_sample['data']['CAM_FRONT']
cam_front_data = nusc.get('sample_data', cam_front_token)

# 获取对应的 Ego Pose (自车姿态)
ego_pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])

print(f"\n🚗 拍摄前置摄像头照片时的自车世界坐标 (x, y, z):")
print(ego_pose['translation'])