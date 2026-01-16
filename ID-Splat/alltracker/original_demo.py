import json
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

def read_tracking_data(data_path):

    # 尝试NumPy格式
    npz_path = data_path.replace('.json', '.npz')
    if os.path.exists(npz_path):
        print(f"Loading NumPy format: {npz_path}")
        data = np.load(npz_path)
        return {
            'trajs': data['trajs'],
            'visconfs': data['visconfs']
        }
    


def read_json(json_path):
    """保持向后兼容的函数"""
    return read_tracking_data(json_path)


def transform_mask_by_trajectory(mask, trajs, visconfs, query_frame=0):

    B, T, N, _ = trajs.shape
    H, W = mask.shape
    
    trajs = trajs[0]  # [T, N, 2]
    visconfs = visconfs[0]  # [T, N, 2]
    
    transformed_masks = np.zeros((T, H, W), dtype=np.uint8)
    
    for t in range(T):
        frame_mask = np.zeros((H, W), dtype=np.uint8)
        
        for i in range(N):
            if visconfs[t, i, 1] > 0.2:  # 置信度阈值
                x, y = trajs[t, i]
                x, y = int(x), int(y)
                
                if 0 <= x < W and 0 <= y < H:
                    if query_frame < T:
                        # 获取query_frame时该像素点的原始位置
                        original_x, original_y = trajs[query_frame, i]
                        original_x, original_y = int(original_x), int(original_y)
                        
                        if 0 <= original_x < W and 0 <= original_y < H:
                            # 获取原始mask值
                            mask_val = mask[original_y, original_x]
                            
                            # 在当前轨迹位置设置相同的mask值
                            frame_mask[y, x] = mask_val
        
        transformed_masks[t] = frame_mask
    
    return transformed_masks


def transform_mask_advanced(mask, trajs, visconfs, query_frame=0, method='pixel_tracking'):
    """
    高级mask变换函数，支持像素级跟踪
    
    Args:
        mask: 原始mask [H, W]
        trajs: 轨迹数据 [B, T, N, 2] - N个轨迹点，每个点有T帧的轨迹
        visconfs: 可见性置信度 [B, T, N, 2]
        query_frame: 查询帧索引
        method: 变换方法 ('pixel_tracking', 'sparse', 'dense', 'interpolation')
    
    Returns:
        transformed_masks: 变换后的mask序列 [T, H, W]
    """
    B, T, N, _ = trajs.shape
    H, W = mask.shape
    
    # 获取轨迹和可见性
    trajs = trajs[0]  # [T, N, 2]
    visconfs = visconfs[0]  # [T, N, 2]
    
    # 创建变换后的mask序列
    transformed_masks = np.zeros((T, H, W), dtype=np.uint8)
    
    if method == 'pixel_tracking':
        # 像素级跟踪：让mask中的每个像素按照其轨迹移动
        print("Using pixel-level tracking method...")
        print(f"Mask shape: {mask.shape}, Trajectories: {trajs.shape}")
        
        # 首先，我们需要理解轨迹数据是如何对应到mask的
        # trajs[0] 是query_frame时的位置，trajs[t] 是第t帧的位置
        
        for t in range(T):
            frame_mask = np.zeros((H, W), dtype=np.uint8)
            
            # 对于每个轨迹点
            for i in range(N):
                if visconfs[t, i, 1] > 0.1:  # 置信度阈值
                    # 获取当前帧中该轨迹点的位置
                    x, y = trajs[t, i]
                    x, y = int(x), int(y)
                    
                    # 检查边界
                    if 0 <= x < W and 0 <= y < H:
                        # 获取query_frame时该轨迹点的位置（这是原始位置）
                        if query_frame < len(trajs):
                            original_x, original_y = trajs[0, i]  # 使用第0帧作为原始位置
                            original_x, original_y = int(original_x), int(original_y)
                            
                            if 0 <= original_x < W and 0 <= original_y < H:
                                # 获取原始mask值
                                mask_val = mask[original_y, original_x]
                                
                                # 在当前轨迹位置设置相同的mask值
                                frame_mask[y, x] = mask_val
            
            transformed_masks[t] = frame_mask
    
    elif method == 'sparse':
        # 稀疏变换：只在轨迹点位置绘制
        for t in range(T):
            frame_mask = np.zeros((H, W), dtype=np.uint8)
            
            for i in range(N):
                if visconfs[t, i, 1] > 0.1:
                    x, y = trajs[t, i]
                    x, y = int(x), int(y)
                    
                    if 0 <= x < W and 0 <= y < H:
                        # 获取原始mask值
                        if query_frame < T:
                            query_mask_val = int(mask[y, x])  # 确保是整数
                        else:
                            query_mask_val = 255
                        
                        # 绘制圆形
                        radius = max(1, int(query_mask_val / 50))
                        cv2.circle(frame_mask, (x, y), radius, query_mask_val, -1)
            
            transformed_masks[t] = frame_mask
    
    elif method == 'dense':
        # 密集变换：为每个像素计算变换
        for t in range(T):
            frame_mask = np.zeros((H, W), dtype=np.uint8)
            
            # 创建网格坐标
            y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            
            # 对于每个轨迹点，计算其对周围像素的影响
            for i in range(N):
                if visconfs[t, i, 1] > 0.1:  # 置信度阈值
                    traj_x, traj_y = trajs[t, i]
                    
                    # 计算距离
                    dist = np.sqrt((x_coords - traj_x)**2 + (y_coords - traj_y)**2)
                    
                    # 使用高斯核进行平滑
                    sigma = 5.0
                    influence = np.exp(-dist**2 / (2 * sigma**2))
                    
                    # 根据原始mask值加权
                    if query_frame < T:
                        # 获取query frame的mask值
                        query_mask_val = mask[int(traj_y), int(traj_x)] if 0 <= int(traj_y) < H and 0 <= int(traj_x) < W else 0
                        influence *= (query_mask_val / 255.0)
                    
                    frame_mask += (influence * 255).astype(np.uint8)
            
            # 归一化
            frame_mask = np.clip(frame_mask, 0, 255)
            transformed_masks[t] = frame_mask
    
    elif method == 'interpolation':
        # 插值变换：使用最近邻插值
        for t in range(T):
            frame_mask = np.zeros((H, W), dtype=np.uint8)
            
            # 创建变换矩阵
            valid_points = visconfs[t, :, 1] > 0.1
            if np.sum(valid_points) > 0:
                valid_trajs = trajs[t, valid_points]
                
                # 使用最近邻插值
                for y in range(H):
                    for x in range(W):
                        # 找到最近的轨迹点
                        distances = np.sqrt(np.sum((valid_trajs - [x, y])**2, axis=1))
                        nearest_idx = np.argmin(distances)
                        nearest_traj = valid_trajs[nearest_idx]
                        
                        # 获取原始mask值
                        if 0 <= int(nearest_traj[1]) < H and 0 <= int(nearest_traj[0]) < W:
                            mask_val = mask[int(nearest_traj[1]), int(nearest_traj[0])]
                            
                            # 根据距离衰减
                            dist = distances[nearest_idx]
                            if dist < 10:  # 阈值
                                frame_mask[y, x] = mask_val
            
            transformed_masks[t] = frame_mask
    
    return transformed_masks


def create_mask_visualization(original_mask, transformed_masks, trajs, visconfs, save_path=None):
    """
    创建mask跟踪的可视化
    
    Args:
        original_mask: 原始mask [H, W]
        transformed_masks: 变换后的mask序列 [T, H, W]
        trajs: 轨迹数据 [B, T, N, 2]
        visconfs: 可见性置信度 [B, T, N, 2]
        save_path: 保存路径
    """
    T, H, W = transformed_masks.shape
    trajs = trajs[0]  # [T, N, 2]
    visconfs = visconfs[0]  # [T, N, 2]
    
    # 创建可视化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # 显示原始mask
    axes[0].imshow(original_mask, cmap='gray')
    axes[0].set_title('Original Mask')
    axes[0].axis('off')
    
    # 显示几个关键帧的变换后mask
    frame_indices = [0, T//4, T//2, 3*T//4, T-1]
    for i, frame_idx in enumerate(frame_indices[:3]):
        if frame_idx < T:
            axes[i+1].imshow(transformed_masks[frame_idx], cmap='gray')
            axes[i+1].set_title(f'Frame {frame_idx}')
            axes[i+1].axis('off')
    
    # 显示轨迹
    axes[4].imshow(original_mask, cmap='gray', alpha=0.5)
    
    # 绘制一些轨迹线
    colors = plt.cm.rainbow(np.linspace(0, 1, min(10, trajs.shape[1])))
    for i in range(min(10, trajs.shape[1])):
        traj = trajs[:, i]  # [T, 2]
        vis = visconfs[:, i, 1]  # [T]
        
        # 只绘制可见的轨迹
        visible = vis > 0.1
        if np.any(visible):
            traj_visible = traj[visible]
            axes[4].plot(traj_visible[:, 0], traj_visible[:, 1], 
                        color=colors[i], linewidth=2, alpha=0.7)
    
    axes[4].set_title('Trajectories')
    axes[4].axis('off')
    
    # 显示mask覆盖
    axes[5].imshow(original_mask, cmap='gray', alpha=0.3)
    axes[5].imshow(transformed_masks[T-1], cmap='hot', alpha=0.7)
    axes[5].set_title('Final Mask Overlay')
    axes[5].axis('off')
    
    # 显示置信度分布
    conf_hist = visconfs[:, :, 1].flatten()
    axes[6].hist(conf_hist, bins=50, alpha=0.7)
    axes[6].set_title('Confidence Distribution')
    axes[6].set_xlabel('Confidence')
    axes[6].set_ylabel('Count')
    
    # 显示轨迹统计
    valid_trajs = np.sum(visconfs[:, :, 1] > 0.1, axis=0)
    axes[7].hist(valid_trajs, bins=20, alpha=0.7)
    axes[7].set_title('Valid Trajectories per Frame')
    axes[7].set_xlabel('Valid Trajectories')
    axes[7].set_ylabel('Frame Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def create_simple_frame_save(original_mask, transformed_masks, original_rgbs=None, save_dir=None):
    """
    简单的逐帧保存，每帧保存一张图片，与视频可视化保持一致
    
    Args:
        original_mask: 原始mask [H, W]
        transformed_masks: 变换后的mask序列 [T, H, W]
        original_rgbs: 原始RGB图像序列 [T, H, W, 3]，如果提供则与mask叠加
        save_dir: 保存目录
    """
    T, H, W = transformed_masks.shape
    
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    print(f"Creating frame-by-frame saves for {T} frames...")
    
    for t in range(T):
        # 创建彩色mask可视化帧，与视频可视化保持一致
        mask = transformed_masks[t]
        
        # 创建彩色映射：使用热力图颜色
        # 处理-1值（没有mask的区域）
        mask_vis = mask.copy()
        mask_vis[mask_vis == -1.0] = 0  # 将-1转换为0用于可视化
        
        # 将mask值映射到彩色
        mask_normalized = mask_vis.astype(np.float32) / 255.0
        
        # 使用matplotlib的hot colormap
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # 创建热力图
        hot_colormap = cm.jet
        mask_colored = hot_colormap(mask_normalized)[:, :, :3]  # 取RGB通道
        mask_colored = (mask_colored * 255).astype(np.uint8)
        
        # 如果提供了原始RGB图像，则与mask叠加
        if original_rgbs is not None and t < len(original_rgbs):
            # 确保RGB图像尺寸匹配
            rgb_frame = original_rgbs[t]
            if rgb_frame.shape[:2] != (H, W):
                rgb_frame = cv2.resize(rgb_frame, (W, H))
            
            # 将mask可视化叠加到RGB图像上
            alpha = 0.7  # mask的透明度
            combined_frame = (mask_colored.astype(np.float32) * alpha + 
                            rgb_frame.astype(np.float32) * (1 - alpha)).astype(np.uint8)
            
            # 保存叠加后的图像
            if save_dir:
                frame_path = os.path.join(save_dir, f"frame_{t:04d}.png")
                plt.imsave(frame_path, combined_frame)
                print(f"Saved frame {t:04d}.png")
        else:
            # 只保存彩色mask
            if save_dir:
                frame_path = os.path.join(save_dir, f"frame_{t:04d}.png")
                plt.imsave(frame_path, mask_colored)
                print(f"Saved frame {t:04d}.png")
    
    print(f"All {T} frames saved to {save_dir}")


def create_video_visualization(original_mask, transformed_masks, original_rgbs=None, save_path=None):
    """
    创建视频可视化
    
    Args:
        original_mask: 原始mask [H, W]
        transformed_masks: 变换后的mask序列 [T, H, W]
        original_rgbs: 原始RGB图像序列 [T, H, W, 3]如果提供则与mask叠加
        save_path: 保存路径
    """
    T, H, W = transformed_masks.shape
    
    # 创建彩色可视化
    video_frames = []
    
    for t in range(T):
        # 创建彩色mask可视化帧
        mask_frame = np.zeros((H, W, 3), dtype=np.uint8)
        
        # 使用彩色映射显示变换后的mask
        mask = transformed_masks[t]
        
        # 创建彩色映射：使用热力图颜色
        # 处理-1值（没有mask的区域）
        mask_vis = mask.copy()
        mask_vis[mask_vis == -1.0] = 0  # 将-1转换为0用于可视化
        
        # 将mask值映射到彩色
        mask_normalized = mask_vis.astype(np.float32) / 255.0
        
        # 使用matplotlib的hot colormap
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # 创建热力图
        hot_colormap = cm.jet
        mask_colored = hot_colormap(mask_normalized)[:, :, :3]  # 取RGB通道
        mask_colored = (mask_colored * 255).astype(np.uint8)
        
        mask_frame = mask_colored
        
        # 如果提供了原始RGB图像，则与mask叠加
        if original_rgbs is not None and t < len(original_rgbs):
            # 确保RGB图像尺寸匹配
            rgb_frame = original_rgbs[t]
            if rgb_frame.shape[:2] != (H, W):
                rgb_frame = cv2.resize(rgb_frame, (W, H))
            
            # 将mask可视化叠加到RGB图像上
            # 使用alpha混合：mask_frame * alpha + rgb_frame * (1-alpha)
            alpha = 0.7  # mask的透明度
            combined_frame = (mask_frame.astype(np.float32) * alpha + 
                            rgb_frame.astype(np.float32) * (1 - alpha)).astype(np.uint8)
            video_frames.append(combined_frame)
        else:
            video_frames.append(mask_frame)
    
    # 保存为视频
    if save_path:
        # 确定最终帧的尺寸
        final_height = video_frames[0].shape[0]
        final_width = video_frames[0].shape[1]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, 10.0, (final_width, final_height))
        
        for frame in video_frames:
            # 转换BGR格式
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Video saved to {save_path}")
    
    return video_frames


def read_video_frames(video_path):
    """
    读取视频文件并返回RGB帧序列
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        frames: RGB帧序列 [T, H, W, 3]
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 转换BGR到RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    return np.array(frames)

def get_mask_information(mask_information_path, query_frame, transformered_mask, hull_masks):
    transformered_mask = torch.from_numpy(transformered_mask)
    all_file = os.listdir(mask_information_path)
    all_file = [file_name for file_name in all_file if file_name.endswith('_s.npy')]
    all_file.sort()
    all_file_name = all_file[query_frame+1:] + all_file[:query_frame]

    # === 第一步：收集所有帧的数据范围以计算全局归一化参数 ===
    print("=== Step 1: Collecting global data range for consistent normalization ===")
    all_data_min = float('inf')
    all_data_max = float('-inf')
    
    # 检查query frame的数据范围
    query_mask_information_path = f"{mask_information_path}/{all_file[query_frame]}"
    query_mask_information = np.load(query_mask_information_path)
    query_mask_information = query_mask_information[1]
    
    print(f"Query frame data range: {query_mask_information.min()} to {query_mask_information.max()}")
    all_data_min = min(all_data_min, query_mask_information.min())
    all_data_max = max(all_data_max, query_mask_information.max())
    
    # 检查其他帧的数据范围
    for file_name_item in all_file_name:
        mask_information = f"/save/RS_3Dopen/Feature/GoogleMap/language_features/City/scene0/{file_name_item}"
        mask_information = np.load(mask_information)
        mask_information_s = mask_information[0]
        
        all_data_min = min(all_data_min, mask_information_s.min())
        all_data_max = max(all_data_max, mask_information_s.max())
    
    print(f"Global data range across all frames: {all_data_min} to {all_data_max}")
    
    # 确定是否需要归一化以及归一化参数
    need_normalization = all_data_max > 255
    if need_normalization:
        print(f"Data range exceeds 255, will apply global normalization...")
        global_range = all_data_max - all_data_min
        if global_range == 0:
            global_range = 1  # 避免除零错误
    else:
        print(f"Data range within 255, no normalization needed")
    
    # === 第二步：使用全局归一化参数处理所有帧 ===
    print("=== Step 2: Processing all frames with consistent normalization ===")
    
    # 处理query frame
    if need_normalization:
        query_mask_information = query_mask_information.astype(np.float32)
        query_mask_information = (query_mask_information - all_data_min) / global_range * 255.0
        query_mask_information = query_mask_information.astype(np.uint8)
    else:
        query_mask_information = query_mask_information.astype(np.uint8)
    
    query_mask_information = torch.from_numpy(query_mask_information)
    query_mask_information = F.interpolate(query_mask_information.unsqueeze(0).unsqueeze(0), size=(568, 1024), mode='nearest').squeeze(0).squeeze(0)
    # 转换为int32以支持-1值
    query_mask_information = query_mask_information.to(torch.int32)

    query_mask_information_unique = torch.unique(query_mask_information)
    print(f"Query frame unique values after normalization: {query_mask_information_unique}")

    new_mask_tracking = []
    for frame_idx, file_name_item in enumerate(all_file_name):
        mask_information = f"/save/RS_3Dopen/Feature/GoogleMap/language_features/City/scene0/{file_name_item}"
        mask_information = np.load(mask_information)
        mask_information_s = mask_information[0]
        
        # 使用全局归一化参数
        if need_normalization:
            mask_information_s = mask_information_s.astype(np.float32)
            mask_information_s = (mask_information_s - all_data_min) / global_range * 255.0
            mask_information_s = mask_information_s.astype(np.uint8)
        else:
            mask_information_s = mask_information_s.astype(np.uint8)
        
        mask_information_s = torch.from_numpy(mask_information_s)
        mask_information_s = F.interpolate(mask_information_s.unsqueeze(0).unsqueeze(0), size=(568, 1024), mode='nearest').squeeze(0).squeeze(0)

        infer_idx = frame_idx  + 1
        hull_mask = hull_masks[infer_idx]

        infer_mask_idx = transformered_mask[infer_idx]
        mask_information_s[mask_information_s == -1] = -1
        # mask_information_s[infer_mask_idx == 0] = -1
        
        mask_information_s_unique = torch.unique(mask_information_s)[1:]
        # 使用int32类型以支持-1值
        new_mask_information_s = torch.full_like(mask_information_s, -1, dtype=torch.int32)
        for item in mask_information_s_unique:
            coor_y, coor_x = torch.where(mask_information_s == item)

            mask_pixel_number = len(coor_x)
            infer_mask = infer_mask_idx[coor_y, coor_x]
            infer_mask_int = infer_mask.long()
            counts = torch.bincount(infer_mask_int)
            # 将0值设为0，避免选择背景
            if len(counts) > 0:
                counts[0] = 0
            most_frequent_idx = torch.argmax(counts)
            change_mask_idx = most_frequent_idx.float()  # 转换为浮点数类型
            infer_mask_idx_number = len(torch.where(change_mask_idx == infer_mask_int)[0])
            ratio = infer_mask_idx_number/mask_pixel_number
            if ratio > 0.05:
                # 确保数据类型匹配
                change_mask_idx_int32 = change_mask_idx.to(new_mask_information_s.dtype)
                new_mask_information_s[coor_y, coor_x] = change_mask_idx_int32

        
        # 计算当前帧中值为0的比例
        zero_ratio = (new_mask_information_s == -1).float().mean().item()
        # 确保hull_mask转换为正确的数据类型
        hull_mask = torch.from_numpy(hull_mask) if isinstance(hull_mask, np.ndarray) else hull_mask
        new_mask_information_s[hull_mask == 0] = -1        
        zero_ratio = (new_mask_information_s == -1).float().mean().item()
        new_mask_tracking.append(new_mask_information_s)
    
    new_mask_tracking = torch.stack(new_mask_tracking)
    new_mask_tracking_unique = torch.unique(new_mask_tracking)
    print("new_mask_tracking_unique", new_mask_tracking_unique)
    
    # 确保维度正确
    new_mask_tracking = torch.cat([query_mask_information.unsqueeze(0), new_mask_tracking], dim=0)
    total_zero_ratio = (new_mask_tracking == -1).float().mean().item()
    print(f"整个序列中值为-1的比例 = {total_zero_ratio:.4f}")
    
    # 验证归一化的一致性
    final_unique = torch.unique(new_mask_tracking)
    final_unique = final_unique[final_unique > 0]  # 排除-1和0
    print(f"Final unique IDs after global normalization: {len(final_unique)} IDs")
    print(f"ID range: {final_unique.min().item()} to {final_unique.max().item()}")
    
    return new_mask_tracking
        # print(mask_information_s_unique)
        # print(len(mask_information_s_unique))


def visualize_new_mask_with_rgb(new_mask_tracking, original_rgbs=None, save_path=None):
    """
    可视化new_mask_tracking与RGB图像的叠加效果
    
    Args:
        new_mask_tracking: 新的mask跟踪结果 [T, H, W]
        original_rgbs: 原始RGB图像序列 [T, H, W, 3]
        save_path: 保存路径
    """
    T, H, W = new_mask_tracking.shape
    
    # 计算全局的ID值范围以保持颜色一致性
    global_max = new_mask_tracking.max().item()
    global_min = new_mask_tracking.min().item()
    print(f"Global ID range for color mapping: {global_min} to {global_max}")
    
    # 创建彩色可视化
    video_frames = []
    
    for t in range(T):
        # 创建彩色mask可视化帧
        mask = new_mask_tracking[t]
        
        # 创建彩色映射：使用热力图颜色
        # 处理-1值（没有mask的区域）
        mask_vis = mask.clone()
        mask_vis[mask_vis == -1] = 0  # 将-1转换为0用于可视化
        
        # 将mask值映射到彩色 - 使用全局范围归一化
        mask_normalized = mask_vis.float() / global_max if global_max > 0 else mask_vis.float()
        
        # 使用matplotlib的hot colormap
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # 创建热力图
        hot_colormap = cm.jet
        mask_colored = hot_colormap(mask_normalized.numpy())[:, :, :3]  # 取RGB通道
        mask_colored = (mask_colored * 255).astype(np.uint8)
        
        # 创建最终的可视化帧
        combined_frame = np.zeros((H, W, 3), dtype=np.uint8)
        
        # 如果提供了原始RGB图像，则在有ID的地方与mask叠加
        if original_rgbs is not None and t < len(original_rgbs):
            # 确保RGB图像尺寸匹配
            rgb_frame = original_rgbs[t]
            if rgb_frame.shape[:2] != (H, W):
                rgb_frame = cv2.resize(rgb_frame, (W, H))
            
            # 只在有ID的地方进行混合
            mask_areas = (mask_vis > 0)
            if mask_areas.any():
                alpha = 0.7  # mask的透明度
                combined_frame[mask_areas] = (mask_colored[mask_areas].astype(np.float32) * alpha + 
                                            rgb_frame[mask_areas].astype(np.float32) * (1 - alpha)).astype(np.uint8)
        else:
            # 如果没有原始RGB图像，只在有ID的地方显示颜色
            mask_areas = (mask_vis > 0)
            combined_frame[mask_areas] = mask_colored[mask_areas]
        
        video_frames.append(combined_frame)
    
    # 保存为视频
    if save_path:
        # 确定最终帧的尺寸
        final_height = video_frames[0].shape[0]
        final_width = video_frames[0].shape[1]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, 10.0, (final_width, final_height))
        
        for frame in video_frames:
            # 转换BGR格式
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"New mask visualization video saved to {save_path}")
    
    return video_frames


def visualize_mask_with_id_numbers(mask_tracking, original_rgbs=None, save_path=None, show_numbers=True):
    """
    可视化mask并在每个区域显示对应的id数字
    
    Args:
        mask_tracking: mask跟踪结果 [T, H, W]
        original_rgbs: 原始RGB图像序列 [T, H, W, 3]
        save_path: 保存路径
        show_numbers: 是否显示id数字
    """
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    T, H, W = mask_tracking.shape
    
    # 计算全局的ID值范围以保持颜色一致性
    global_max = mask_tracking.max().item()
    global_min = mask_tracking.min().item()
    print(f"Global ID range for color mapping: {global_min} to {global_max}")
    
    # 创建彩色可视化
    video_frames = []
    
    for t in range(T):
        # 创建彩色mask可视化帧
        mask = mask_tracking[t]
        
        # 创建彩色映射：使用热力图颜色
        # 处理-1值（没有mask的区域）
        mask_vis = mask.clone()
        mask_vis[mask_vis == -1] = 0  # 将-1转换为0用于可视化
        
        # 将mask值映射到彩色 - 使用全局范围归一化
        if global_max > 0:
            mask_normalized = mask_vis.float() / global_max
        else:
            mask_normalized = mask_vis.float()
        
        # 使用matplotlib的jet colormap
        hot_colormap = cm.jet
        mask_colored = hot_colormap(mask_normalized.numpy())[:, :, :3]  # 取RGB通道
        mask_colored = (mask_colored * 255).astype(np.uint8)
        
        # 创建最终的可视化帧
        combined_frame = np.zeros((H, W, 3), dtype=np.uint8)
        
        # 如果提供了原始RGB图像，则在有ID的地方与mask叠加
        if original_rgbs is not None and t < len(original_rgbs):
            # 确保RGB图像尺寸匹配
            rgb_frame = original_rgbs[t]
            if rgb_frame.shape[:2] != (H, W):
                rgb_frame = cv2.resize(rgb_frame, (W, H))
            
            # 只在有ID的地方进行混合
            mask_areas = (mask_vis > 0)
            if mask_areas.any():
                alpha = 0.7  # mask的透明度
                combined_frame[mask_areas] = (mask_colored[mask_areas].astype(np.float32) * alpha + 
                                            rgb_frame[mask_areas].astype(np.float32) * (1 - alpha)).astype(np.uint8)
        else:
            # 如果没有原始RGB图像，只在有ID的地方显示颜色
            mask_areas = (mask_vis > 0)
            combined_frame[mask_areas] = mask_colored[mask_areas]
        
        # 在mask区域上添加id数字
        if show_numbers:
            # 获取当前帧中的所有unique id（排除-1和0）
            unique_ids = torch.unique(mask)
            unique_ids = unique_ids[unique_ids > 0]  # 只保留正数id
            
            for id_val in unique_ids:
                # 找到该id的所有像素位置
                id_pixels = torch.where(mask == id_val)
                
                if len(id_pixels[0]) > 0:
                    # 计算该id区域的中心位置
                    center_y = int(torch.mean(id_pixels[0].float()).item())
                    center_x = int(torch.mean(id_pixels[1].float()).item())
                    
                    # 在图像上绘制id数字
                    id_text = str(int(id_val.item()))
                    
                    # 选择合适的字体大小
                    font_scale = 0.6
                    font_thickness = 2
                    
                    # 获取文本大小
                    text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    
                    # 绘制黑色背景
                    cv2.rectangle(combined_frame, 
                                (center_x - text_size[0]//2 - 2, center_y - text_size[1]//2 - 2),
                                (center_x + text_size[0]//2 + 2, center_y + text_size[1]//2 + 2),
                                (0, 0, 0), -1)
                    
                    # 绘制白色文字
                    cv2.putText(combined_frame, id_text, 
                              (center_x - text_size[0]//2, center_y + text_size[1]//2),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        video_frames.append(combined_frame)
    
    # 保存为视频
    if save_path:
        # 确定最终帧的尺寸
        final_height = video_frames[0].shape[0]
        final_width = video_frames[0].shape[1]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, 10.0, (final_width, final_height))
        
        for frame in video_frames:
            # 转换BGR格式
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Mask with ID numbers visualization saved to {save_path}")
    
    return video_frames


def create_id_legend(mask_tracking, save_path=None):
    """
    创建id数字的图例，显示每个id对应的颜色
    
    Args:
        mask_tracking: mask跟踪结果 [T, H, W]
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    
    # 获取所有unique id
    all_ids = torch.unique(mask_tracking)
    all_ids = all_ids[all_ids > 0]  # 只保留正数id
    
    if len(all_ids) == 0:
        print("No valid IDs found for legend")
        return
    
    # 创建颜色映射
    max_id = all_ids.max().item()
    colormap = cm.jet
    
    # 创建图例
    fig, ax = plt.subplots(figsize=(10, max(6, len(all_ids) * 0.3)))
    
    # 为每个id创建颜色条
    for i, id_val in enumerate(all_ids):
        normalized_val = id_val.item() / max_id
        color = colormap(normalized_val)
        
        # 绘制颜色条
        ax.barh(i, 1, color=color, alpha=0.8)
        ax.text(1.1, i, f'ID: {int(id_val.item())}', 
                va='center', ha='left', fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.5, len(all_ids) - 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('ID Color Legend', fontsize=14, fontweight='bold')
    
    # 移除边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ID legend saved to {save_path}")
    
    plt.show()


def create_trajectory_hull_masks(transformed_masks):
    """
    从transformed_masks中找到轨迹点，创建凸包，圈内为1，圈外为0
    
    Args:
        transformed_masks: 变换后的mask序列 [T, H, W]
        
    Returns:
        hull_masks: 凸包mask序列 [T, H, W]
    """
    from scipy.spatial import ConvexHull
    import cv2
    
    T, H, W = transformed_masks.shape
    hull_masks = np.zeros_like(transformed_masks)
    
    for t in range(T):
        mask = transformed_masks[t]
        
        # 找到所有有轨迹的点（非0且非-1的点）
        trajectory_points = []
        
        # 从每行每列找第一个有轨迹的点
        # 从上到下找每行的第一个轨迹点
        for y in range(H):
            row = mask[y, :]
            valid_indices = np.where((row != 0) & (row != -1))[0]
            if len(valid_indices) > 0:
                x = valid_indices[0]  # 第一个有轨迹的点
                trajectory_points.append([x, y])
        
        # 从下到上找每行的第一个轨迹点
        for y in range(H-1, -1, -1):
            row = mask[y, :]
            valid_indices = np.where((row != 0) & (row != -1))[0]
            if len(valid_indices) > 0:
                x = valid_indices[-1]  # 最后一个有轨迹的点
                trajectory_points.append([x, y])
        
        # 从左到右找每列的第一个轨迹点
        for x in range(W):
            col = mask[:, x]
            valid_indices = np.where((col != 0) & (col != -1))[0]
            if len(valid_indices) > 0:
                y = valid_indices[0]  # 第一个有轨迹的点
                trajectory_points.append([x, y])
        
        # 从右到左找每列的第一个轨迹点
        for x in range(W-1, -1, -1):
            col = mask[:, x]
            valid_indices = np.where((col != 0) & (col != -1))[0]
            if len(valid_indices) > 0:
                y = valid_indices[-1]  # 最后一个有轨迹的点
                trajectory_points.append([x, y])
        
        # 去重
        trajectory_points = np.unique(np.array(trajectory_points), axis=0)
        
        if len(trajectory_points) >= 3:  # 至少需要3个点才能形成凸包
            try:
                # 计算凸包
                hull = ConvexHull(trajectory_points)
                hull_points = trajectory_points[hull.vertices]
                
                # 创建凸包mask
                hull_mask = np.zeros((H, W), dtype=np.uint8)
                
                # 将凸包点转换为OpenCV格式
                hull_points_cv = hull_points.reshape((-1, 1, 2)).astype(np.int32)
                
                # 填充凸包内部
                cv2.fillPoly(hull_mask, [hull_points_cv], 1)
                
                hull_masks[t] = hull_mask
                
                print(f"Frame {t}: 找到 {len(trajectory_points)} 个轨迹点，创建凸包")
                
            except Exception as e:
                print(f"Frame {t}: 创建凸包失败: {e}")
                hull_masks[t] = np.zeros((H, W))
        else:
            print(f"Frame {t}: 轨迹点不足 ({len(trajectory_points)} < 3)，跳过")
            hull_masks[t] = np.zeros((H, W))
    
    return hull_masks


if __name__ == '__main__':
    json_path = '/home/amax/wyj/RS_3DOPEN/alltracker/pt_vis_city_scene0_video_rate2_q0.json'
    
    data = read_tracking_data(json_path)
    trajs = data['trajs']
    visconfs = data['visconfs']
    query_frame = json_path.split('_')[-1].split('.')[0][1:]
    mask_information_path = f"/save/RS_3Dopen/Feature/GoogleMap/language_features/City/scene0/"

    mask_information = f"/save/RS_3Dopen/Feature/GoogleMap/language_features/City/scene0/frame_{int(query_frame):03d}_s.npy"
    mask_information = np.load(mask_information)
    mask_information_s = mask_information[0]
    mask_information_s_idx = torch.unique(torch.from_numpy(mask_information_s))
    print(f"Original mask_information_s range: {mask_information_s.min()} to {mask_information_s.max()}")
    print(f"Unique IDs in mask_information_s: {len(mask_information_s_idx)}")
    print(f"ID values: {mask_information_s_idx}")
    

    
    
    # 处理超过255的数值
    if mask_information_s.max() > 255:
        print(f"Data range exceeds 255, normalizing...")
        mask_information_s = mask_information_s.astype(np.float32)
        # 归一化到0-255范围
        mask_information_s = (mask_information_s - mask_information_s.min()) / (mask_information_s.max() - mask_information_s.min()) * 255.0
        mask_information_s = mask_information_s.astype(np.uint8)
    else:
        mask_information_s = mask_information_s.astype(np.uint8)
    mask_information_s = torch.from_numpy(mask_information_s)
    mask_information_s = F.interpolate(mask_information_s.unsqueeze(0).unsqueeze(0), size=(568, 1024), mode='nearest').squeeze(0).squeeze(0)
    query_frame = int(query_frame)


    # 将mask_information_s转换为torch tensor用于可视化
    # mask_tensor = torch.from_numpy(mask_information_s)
    
    # 可视化原始mask信息（单帧）
    print("\n=== Visualizing Original Mask Information ===")
    
    # 创建单帧的可视化（添加batch维度）
    mask_single_frame = mask_information_s.unsqueeze(0)  # 添加时间维度 [1, H, W]
    
    # 先进行跟踪处理以获得全局颜色范围
    print("Processing tracking for global color range...")
    
    transformed_masks = transform_mask_by_trajectory(mask_information_s, trajs, visconfs, query_frame)
    transformed_masks = transformed_masks.astype(np.float32)
    
    hull_masks = create_trajectory_hull_masks(transformed_masks)
    new_mask_tracking = get_mask_information(mask_information_path, query_frame, transformed_masks, hull_masks)
    
    # 获取全局颜色范围（包括原始帧和跟踪帧）
    global_max_all = max(mask_information_s.max().item(), new_mask_tracking.max().item())
    global_min_all = min(mask_information_s.min().item(), new_mask_tracking.min().item())
    print(f"Global color range for all frames: {global_min_all} to {global_max_all}")
    
    # 修改原始mask可视化以使用全局范围
    def visualize_original_with_global_range(mask_single, global_max, save_path):
        """使用全局颜色范围可视化原始mask"""
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        T, H, W = mask_single.shape
        video_frames = []
        
        for t in range(T):
            mask = mask_single[t]
            mask_vis = mask.clone()
            mask_vis[mask_vis == -1] = 0
            
            # 使用全局范围归一化
            if global_max > 0:
                mask_normalized = mask_vis.float() / global_max
            else:
                mask_normalized = mask_vis.float()
            
            hot_colormap = cm.jet
            mask_colored = hot_colormap(mask_normalized.numpy())[:, :, :3]
            mask_colored = (mask_colored * 255).astype(np.uint8)
            
            combined_frame = np.zeros((H, W, 3), dtype=np.uint8)
            mask_areas = (mask_vis > 0)
            combined_frame[mask_areas] = mask_colored[mask_areas]
            
            # 添加ID数字
            unique_ids = torch.unique(mask)
            unique_ids = unique_ids[unique_ids > 0]
            
            for id_val in unique_ids:
                id_pixels = torch.where(mask == id_val)
                if len(id_pixels[0]) > 0:
                    center_y = int(torch.mean(id_pixels[0].float()).item())
                    center_x = int(torch.mean(id_pixels[1].float()).item())
                    id_text = str(int(id_val.item()))
                    font_scale = 0.6
                    font_thickness = 2
                    text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    
                    cv2.rectangle(combined_frame, 
                                (center_x - text_size[0]//2 - 2, center_y - text_size[1]//2 - 2),
                                (center_x + text_size[0]//2 + 2, center_y + text_size[1]//2 + 2),
                                (0, 0, 0), -1)
                    cv2.putText(combined_frame, id_text, 
                              (center_x - text_size[0]//2, center_y + text_size[1]//2),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
            
            video_frames.append(combined_frame)
        
        # 保存视频
        if save_path:
            final_height = video_frames[0].shape[0]
            final_width = video_frames[0].shape[1]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, 10.0, (final_width, final_height))
            for frame in video_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
            print(f"Original mask with consistent colors saved to {save_path}")
    
    # 使用全局颜色范围可视化原始mask
    visualize_original_with_global_range(mask_single_frame, global_max_all, 'original_mask_with_ids.mp4')
    
    # 创建原始mask的ID颜色图例（使用全局范围）
    create_id_legend(mask_single_frame, save_path='original_mask_id_legend.png')
    
    # 显示原始mask信息统计
    print("\n=== Original Mask Statistics ===")
    unique_ids = torch.unique(mask_information_s)
    unique_ids = unique_ids[unique_ids > 0]  # 只保留正数id
    print(f"Total unique IDs: {len(unique_ids)}")
    if len(unique_ids) > 0:
        print(f"ID range: {unique_ids.min().item()} to {unique_ids.max().item()}")
        
        # 统计每个ID的像素数量
        print("\nID pixel counts:")
        for id_val in unique_ids[:30]:  # 显示前30个ID
            count = (mask_information_s == id_val).sum().item()
            print(f"ID {int(id_val.item())}: {count} pixels")
        
        if len(unique_ids) > 30:
            print(f"... and {len(unique_ids) - 30} more IDs")
    
    print("\n=== Original Mask Visualization Completed! ===")
    print("Generated files:")
    print("- original_mask_with_ids.mp4: Original mask with ID numbers")
    print("- original_mask_id_legend.png: Color legend for original mask IDs")


    # 保存跟踪结果
    save_path = f"/save/RS_3Dopen/Feature/GoogleMap/language_features/City/scene0/mask_tracking_frames.npy"
    np.save(save_path, transformed_masks)
    
    hull_save_path = f"/save/RS_3Dopen/Feature/GoogleMap/language_features/City/scene0/hull_masks.npy"
    np.save(hull_save_path, hull_masks)
    print(f"Hull masks saved to {hull_save_path}")
    
    print(f"Tracking results saved. Using global max: {global_max_all}")
    
    # 读取原始RGB图像用于可视化
    original_rgbs = None
    try:
        video_path = '/home/amax/wyj/RS_3DOPEN/gaussian-grouping/city_scene0_video.mp4'  
        original_rgbs = read_video_frames(video_path)
        print(f"Successfully loaded {len(original_rgbs)} RGB frames for visualization")
    except Exception as e:
        print(f"Could not load original video: {e}")
        print("Creating visualization without original RGB frames")
    
    # 修改可视化函数以使用全局颜色范围
    def visualize_with_global_range(mask_tracking, global_max, original_rgbs=None, save_path=None):
        """使用全局颜色范围的可视化函数"""
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        T, H, W = mask_tracking.shape
        video_frames = []
        
        for t in range(T):
            mask = mask_tracking[t]
            mask_vis = mask.clone()
            mask_vis[mask_vis == -1] = 0
            
            # 使用全局范围归一化
            mask_normalized = mask_vis.float() / global_max if global_max > 0 else mask_vis.float()
            
            hot_colormap = cm.jet
            mask_colored = hot_colormap(mask_normalized.numpy())[:, :, :3]
            mask_colored = (mask_colored * 255).astype(np.uint8)
            
            combined_frame = np.zeros((H, W, 3), dtype=np.uint8)
            
            if original_rgbs is not None and t < len(original_rgbs):
                rgb_frame = original_rgbs[t]
                if rgb_frame.shape[:2] != (H, W):
                    rgb_frame = cv2.resize(rgb_frame, (W, H))
                
                mask_areas = (mask_vis > 0)
                if mask_areas.any():
                    alpha = 0.7
                    combined_frame[mask_areas] = (mask_colored[mask_areas].astype(np.float32) * alpha + 
                                                rgb_frame[mask_areas].astype(np.float32) * (1 - alpha)).astype(np.uint8)
            else:
                mask_areas = (mask_vis > 0)
                combined_frame[mask_areas] = mask_colored[mask_areas]
            
            video_frames.append(combined_frame)
        
        if save_path:
            final_height = video_frames[0].shape[0]
            final_width = video_frames[0].shape[1]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, 10.0, (final_width, final_height))
            for frame in video_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
            print(f"Visualization with consistent colors saved to {save_path}")
    
    # 使用全局颜色范围可视化
    print("Creating visualizations with consistent global color range...")
    visualize_with_global_range(new_mask_tracking, global_max_all, original_rgbs=original_rgbs, save_path='new_mask_tracking_rgb.mp4')
    
    hull_masks_tensor = torch.from_numpy(hull_masks)
    visualize_with_global_range(hull_masks_tensor, global_max_all, original_rgbs=original_rgbs, save_path='hull_masks_rgb.mp4')
    
    # 只可视化特定ID的mask
    print("\n=== Creating Specific ID Visualization ===")
    # 111, 180,
    target_ids = [111,144, 180]
    print(f"Target IDs: {target_ids}")
    
    # 创建筛选后的mask
    filtered_mask = torch.zeros_like(new_mask_tracking)
    
    for target_id in target_ids:
        # 找到该ID在原始mask中的位置
        mask_positions = (new_mask_tracking == target_id)
        if mask_positions.any():
            # 在筛选后的mask中保留该ID
            filtered_mask[mask_positions] = target_id
            count = mask_positions.sum().item()
            print(f"ID {target_id}: {count} pixels found")
        else:
            print(f"ID {target_id}: NOT FOUND in mask")
    
    # 统计筛选后的mask信息
    unique_filtered_ids = torch.unique(filtered_mask)
    unique_filtered_ids = unique_filtered_ids[unique_filtered_ids > 0]
    print(f"Successfully filtered IDs: {unique_filtered_ids.tolist()}")
    
    # 创建带ID数字的可视化函数
    def visualize_with_ids_global_range(mask_tracking, global_max, original_rgbs=None, save_path=None):
        """带ID数字的全局颜色范围可视化"""
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        T, H, W = mask_tracking.shape
        video_frames = []
        
        for t in range(T):
            mask = mask_tracking[t]
            mask_vis = mask.clone()
            mask_vis[mask_vis == -1] = 0
            
            # 使用全局范围归一化
            mask_normalized = mask_vis.float() / global_max if global_max > 0 else mask_vis.float()
            
            hot_colormap = cm.jet
            mask_colored = hot_colormap(mask_normalized.numpy())[:, :, :3]
            mask_colored = (mask_colored * 255).astype(np.uint8)
            
            combined_frame = np.zeros((H, W, 3), dtype=np.uint8)
            
            if original_rgbs is not None and t < len(original_rgbs):
                rgb_frame = original_rgbs[t]
                if rgb_frame.shape[:2] != (H, W):
                    rgb_frame = cv2.resize(rgb_frame, (W, H))
                
                mask_areas = (mask_vis > 0)
                if mask_areas.any():
                    alpha = 0.7
                    combined_frame[mask_areas] = (mask_colored[mask_areas].astype(np.float32) * alpha + 
                                                rgb_frame[mask_areas].astype(np.float32) * (1 - alpha)).astype(np.uint8)
            else:
                mask_areas = (mask_vis > 0)
                combined_frame[mask_areas] = mask_colored[mask_areas]
            
            # 添加ID数字
            unique_ids = torch.unique(mask)
            unique_ids = unique_ids[unique_ids > 0]
            
            for id_val in unique_ids:
                id_pixels = torch.where(mask == id_val)
                if len(id_pixels[0]) > 0:
                    center_y = int(torch.mean(id_pixels[0].float()).item())
                    center_x = int(torch.mean(id_pixels[1].float()).item())
                    id_text = str(int(id_val.item()))
                    font_scale = 0.6
                    font_thickness = 2
                    text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    
                    cv2.rectangle(combined_frame, 
                                (center_x - text_size[0]//2 - 2, center_y - text_size[1]//2 - 2),
                                (center_x + text_size[0]//2 + 2, center_y + text_size[1]//2 + 2),
                                (0, 0, 0), -1)
                    cv2.putText(combined_frame, id_text, 
                              (center_x - text_size[0]//2, center_y + text_size[1]//2),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
            
            video_frames.append(combined_frame)
        
        if save_path:
            final_height = video_frames[0].shape[0]
            final_width = video_frames[0].shape[1]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, 10.0, (final_width, final_height))
            for frame in video_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
            print(f"Filtered mask with consistent colors and IDs saved to {save_path}")
    
    # 使用全局颜色范围可视化筛选后的mask
    print("Creating filtered mask visualizations with consistent colors...")
    visualize_with_ids_global_range(filtered_mask, global_max_all, original_rgbs=original_rgbs, 
                                   save_path='filtered_mask_with_ids.mp4')
    
    visualize_with_global_range(filtered_mask, global_max_all, original_rgbs=original_rgbs, 
                               save_path='filtered_mask_no_ids.mp4')
    
    # 创建筛选后mask的ID颜色图例
    if len(unique_filtered_ids) > 0:
        create_id_legend(filtered_mask, save_path='filtered_mask_legend.png')
    
    # 显示筛选后mask的统计信息
    print("\n=== Filtered Mask Statistics ===")
    total_pixels = (filtered_mask > 0).sum().item()
    total_mask_pixels = (new_mask_tracking > 0).sum().item()
    print(f"Total filtered pixels: {total_pixels}")
    print(f"Total original pixels: {total_mask_pixels}")
    print(f"Filtered ratio: {total_pixels/total_mask_pixels:.4f}")
    
    for id_val in unique_filtered_ids:
        count = (filtered_mask == id_val).sum().item()
        print(f"ID {int(id_val.item())}: {count} pixels")
    
    print("\n=== Filtered Mask Visualization Completed! ===")
    print("Generated files:")
    print("- filtered_mask_with_ids.mp4: Filtered mask with ID numbers")
    print("- filtered_mask_no_ids.mp4: Filtered mask without ID numbers")
    print("- filtered_mask_legend.png: Color legend for filtered IDs")







    # original_rgbs = None
    # try:
    #     video_path = '/home/amax/wyj/RS_3DOPEN/gaussian-grouping/city_scene0_video.mp4'  
    #     original_rgbs = read_video_frames(video_path)
    #     print(f"Successfully loaded {len(original_rgbs)} RGB frames")
    # except Exception as e:
    #     print(f"Could not load original video: {e}")
    #     print("Creating visualization without original RGB frames")
    
    # create_simple_frame_save(mask_information_s, transformed_masks, original_rgbs=original_rgbs,  save_dir='mask_tracking_frames')

    