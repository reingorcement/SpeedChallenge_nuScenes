import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def draw_ppt_flow(image, flow, step=48, scale=3.0, min_motion=2.0):
    """
    绘制适合 PPT 展示的高质量光流图
    
    参数:
        image: 原图 (BGR格式)
        flow: 光流场 (H x W x 2)
        step: 采样步长。建议设大一点(如 48 或 64)，让画面更清爽
        scale: 光流放大系数。将微小的运动放大，使趋势更明显
        min_motion: 最小运动阈值，过滤静止背景噪声
    """
    # 1. 压暗背景：降低图像亮度，作为“半透明底板”，凸显箭头
    # 这里将原图亮度降低为原来的 70%
    output_img = cv2.convertScaleAbs(image, alpha=0.7, beta=0)
    
    h, w = output_img.shape[:2]
    
    # 配色方案 (BGR格式)
    arrow_color = (0, 255, 255)  # 主体颜色：高对比度明黄色
    outline_color = (0, 255, 255)    # 描边颜色：纯黑
    dot_color = (0, 255, 255)      # 起点颜色：正红
    
    # 2. 遍历图像像素进行采样绘制
    # 从 step//2 开始采样，避免箭头贴合图像边缘
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            dx, dy = flow[y, x]
            
            # 计算运动幅值
            magnitude = np.sqrt(dx**2 + dy**2)
            
            # 过滤微小运动
            if magnitude > min_motion:
                # 3. 乘上放大系数，计算箭头终点
                end_x = int(x + dx * scale)
                end_y = int(y + dy * scale)
                
                # --- 绘制层叠效果 (已调小尺寸) ---
                
                # A. 画箭头描边 (较粗的黑线：粗细由 4 改为 2)
                cv2.arrowedLine(output_img, (x, y), (end_x, end_y), 
                                outline_color, thickness=2, tipLength=0.3)
                
                # B. 画箭头主体 (较细的黄线：粗细由 2 改为 1)
                cv2.arrowedLine(output_img, (x, y), (end_x, end_y), 
                                arrow_color, thickness=1, tipLength=0.3)
                
                # C. 画起点锚点描边 (黑色大圆：半径由 4 改为 2)
                cv2.circle(output_img, (x, y), 2, outline_color, -1)
                
                # D. 画起点锚点红心 (红色小圆：半径由 2 改为 1)
                cv2.circle(output_img, (x, y), 1, dot_color, -1)

    return output_img

def main():
    # 1. 设置路径 (请替换为你的实际路径)
    # frame1_path = 'frame1.png' 
    frame1_path = 'cexiang_qianyizhen.jpg' 
    frame2_path = 'cexiang_houyizhen.jpg'  
    output_path = 'cexiang_ppt_flow_result.png'
    
    # 读取图像
    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)
    
    if frame1 is None or frame2 is None:
        print("错误：无法读取图像，请检查路径。")
        return

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # # 2. 计算 Farneback 稠密光流 (保持不变)
    # print("正在计算 Farneback 光流...")
    # flow = cv2.calcOpticalFlowFarneback(
    #     prev=gray1, next=gray2, flow=None, 
    #     pyr_scale=0.5, levels=3, winsize=15, 
    #     iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    # )
    # 2. 计算 DIS 稠密光流 (精度与速度的优秀折中)
    print("正在计算 DIS 光流...")
    # DISOPTICAL_FLOW_PRESET_MEDIUM 侧重于较高的精度。如果需要极速，可改用 PRESET_FAST
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(gray1, gray2, None)

    # 3. 生成学术汇报级可视化图
    print("正在生成高质量可视化图...")
    # 将 scale 从 4.0 调小到 1.5，箭头长度会显著变短
    result_img = draw_ppt_flow(frame1, flow, step=48, scale=0.5, min_motion=1.5)

    # 4. 直接保存结果 (规避无头环境的 imshow 报错)
    cv2.imwrite(output_path, result_img)
    print(f"高质量结果已保存至: {os.path.abspath(output_path)}")
    
    # 5. 使用 Matplotlib 预览 (可选，如果不希望弹窗可将此段注释)
    print("正在启动预览...")
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 7))  # 设定一个宽屏比例
    plt.imshow(result_img_rgb)
    plt.title('Optical Flow Estimation', fontsize=16)
    plt.axis('off') # 隐藏坐标轴
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()