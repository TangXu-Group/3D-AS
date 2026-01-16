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
from datetime import datetime
from tqdm import tqdm
from scipy import ndimage
from natsort import natsorted
import json
import argparse





def read_tracking_data(data_path):

    npz_path = data_path.replace('.json', '.npz')
    if os.path.exists(npz_path):
        print(f"Loading NumPy format: {npz_path}")
        data = np.load(npz_path)
        return {
            'trajs': data['trajs'],
            'visconfs': data['visconfs']
        }

def read_json(json_path):
    return read_tracking_data(json_path)

def create_trajectory_hull_masks(transformed_masks):
    """
    Args:
        transformed_masks:  [T, H, W]
        
    Returns:
        hull_masks:  [T, H, W]
    """
    from scipy.spatial import ConvexHull
    import cv2
    
    T, H, W = transformed_masks.shape
    hull_masks = np.zeros_like(transformed_masks)
    
    for t in range(T):
        mask = transformed_masks[t]
        
        trajectory_points = []
        
        for y in range(H):
            row = mask[y, :]
            valid_indices = np.where((row != 0) & (row != -1))[0]
            if len(valid_indices) > 0:
                x = valid_indices[0]
                trajectory_points.append([x, y])
        
        for y in range(H-1, -1, -1):
            row = mask[y, :]
            valid_indices = np.where((row != 0) & (row != -1))[0]
            if len(valid_indices) > 0:
                x = valid_indices[-1]
                trajectory_points.append([x, y])
        
        for x in range(W):
            col = mask[:, x]
            valid_indices = np.where((col != 0) & (col != -1))[0]
            if len(valid_indices) > 0:
                y = valid_indices[0]
                trajectory_points.append([x, y])
        
        for x in range(W-1, -1, -1):
            col = mask[:, x]
            valid_indices = np.where((col != 0) & (col != -1))[0]
            if len(valid_indices) > 0:
                y = valid_indices[-1]
                trajectory_points.append([x, y])
        
        trajectory_points = np.unique(np.array(trajectory_points), axis=0)
        
        if len(trajectory_points) >= 3:
            try:
                hull = ConvexHull(trajectory_points)
                hull_points = trajectory_points[hull.vertices]
                
                hull_mask = np.zeros((H, W), dtype=np.uint8)
                
                hull_points_cv = hull_points.reshape((-1, 1, 2)).astype(np.int32)
                
                cv2.fillPoly(hull_mask, [hull_points_cv], 1)
                
                hull_masks[t] = hull_mask
                
                
            except Exception as e:
                hull_masks[t] = np.zeros((H, W))
        else:
            hull_masks[t] = np.zeros((H, W))
    
    return hull_masks


def transform_mask_by_trajectory(mask, trajs, visconfs, query_frame=0):

    B, T, N, _ = trajs.shape
    H, W = mask.shape
    trajs = trajs[0]  # [T, N, 2]
    visconfs = visconfs[0]  # [T, N, 2]
    
    transformed_masks = np.zeros((T, H, W), dtype=np.int32)
    
    for t in tqdm(range(T)):
        frame_mask = np.ones((H, W), dtype=np.int32) * -1
        
        for i in range(N):
            if np.mean(visconfs[t, i]) > 0.5:  
                x, y = trajs[t, i]
                x, y = int(x), int(y)
                
                if 0 <= x < W and 0 <= y < H:
                    if query_frame < T:

                        original_x, original_y = trajs[query_frame, i]
                        original_x, original_y = int(original_x), int(original_y)
                        
                        if 0 <= original_x < W and 0 <= original_y < H:
                            
                            mask_val = mask[original_y, original_x]
                            
                            
                            frame_mask[y, x] = mask_val
        
        transformed_masks[t] = frame_mask
    
    return transformed_masks



def get_mask_information_preserve_ids(mask_information_path, query_frame, transformed_mask, hull_masks, scene, scene_name, feature_scale):

    transformed_mask = torch.from_numpy(transformed_mask)
    transformed_mask = transformed_mask + 1


    all_file = os.listdir(mask_information_path)
    all_file = [file_name for file_name in all_file if file_name.endswith('_s.npy')]
    all_file.sort()
    all_file_name = all_file[query_frame+1:] + all_file[:query_frame]
    print("all_file_name", all_file_name)
    print(f"hull_masks: {hull_masks.shape}")
    print(f"hull_masks[query_frame+1:]: {hull_masks[query_frame:].shape}")
    print(f"hull_masks[:query_frame]: {hull_masks[:query_frame].shape}")
    hull_masks = torch.from_numpy(hull_masks)
    hull_masks = torch.cat([hull_masks[query_frame:], hull_masks[:query_frame]], dim=0)
    transformed_mask = torch.cat([transformed_mask[query_frame:], transformed_mask[:query_frame]], dim=0)
    
    query_mask_information_path = f"{mask_information_path}/{all_file[query_frame]}"
    query_mask_information = np.load(query_mask_information_path)
    query_mask_information = query_mask_information[int(feature_scale)]
    
    print(f"Original query_mask_information range: {query_mask_information.min()} to {query_mask_information.max()}")
    

    query_mask_information = torch.from_numpy(query_mask_information.astype(np.float32))
    query_mask_information = F.interpolate(query_mask_information.unsqueeze(0).unsqueeze(0), 
                                          size=(568, 1024), mode='nearest').squeeze(0).squeeze(0)
    query_mask_information = query_mask_information.to(torch.int32)
    
    query_mask_information_unique = torch.unique(query_mask_information)
    query_mask_information_unique = query_mask_information_unique[query_mask_information_unique > 0]
    print(f"Query frame unique IDs: {len(query_mask_information_unique)} IDs")
    print(f"ID range: {query_mask_information_unique.min().item()} to {query_mask_information_unique.max().item()}")

    new_mask_tracking = []
    
    for frame_idx, file_name_item in enumerate(all_file_name):
        mask_information = f"/save/RS_3Dopen/Feature/GoogleMap/language_features/{scene}/{scene_name}/{file_name_item}"
        mask_information = np.load(mask_information)
        mask_information_s = mask_information[int(feature_scale)]
        
        mask_information_s = torch.from_numpy(mask_information_s.astype(np.float32))
        mask_information_s = F.interpolate(mask_information_s.unsqueeze(0).unsqueeze(0), 
                                          size=(568, 1024), mode='nearest').squeeze(0).squeeze(0)
        mask_information_s = mask_information_s.to(torch.int32)

        infer_idx = frame_idx + 1
        hull_mask = hull_masks[infer_idx]
        infer_mask_idx = transformed_mask[infer_idx]
        
        new_mask_information_s = torch.full_like(mask_information_s, -1, dtype=torch.int32)
        new_mask_information_s_dict = {}

        mask_information_s_unique = torch.unique(mask_information_s)
        mask_information_s_unique = mask_information_s_unique[mask_information_s_unique > 0]
        
        for item in mask_information_s_unique:
            coor_y, coor_x = torch.where(mask_information_s == item)
            mask_pixel_number = len(coor_x)
            
            if mask_pixel_number > 0:
                infer_mask = infer_mask_idx[coor_y, coor_x]
                infer_mask_int = infer_mask.long()
                hull_information = hull_mask[coor_y, coor_x]
                hull_information_all = hull_information.sum()

                counts = torch.bincount(infer_mask_int)
                counts = counts[1:]
                if len(counts) > 0:
                    counts[0] = 0
                if counts.sum() > 0:
                    most_frequent_idx = torch.argmax(counts)
                    change_mask_idx = most_frequent_idx.float()
                    infer_mask_idx_number = len(torch.where(change_mask_idx == infer_mask_int)[0])
                    ratio = infer_mask_idx_number / mask_pixel_number
                    new_mask_information_s[coor_y, coor_x] = (change_mask_idx).int()


        new_mask_tracking.append(new_mask_information_s)
    


    new_mask_tracking = torch.stack(new_mask_tracking)
    new_mask_tracking = torch.cat([query_mask_information.unsqueeze(0), new_mask_tracking], dim=0)
    len_new_mask_tracking = len(new_mask_tracking)
    print(f"len_new_mask_tracking: {new_mask_tracking[len_new_mask_tracking-query_frame:].shape}")
    print(f"len_new_mask_tracking: {new_mask_tracking[:len_new_mask_tracking-query_frame].shape}")
    new_mask_tracking = torch.cat([new_mask_tracking[len_new_mask_tracking-query_frame:], new_mask_tracking[:len_new_mask_tracking-query_frame]], dim=0)
    print(f"len_new_mask_tracking: {len_new_mask_tracking}")
    print(f"new_mask_tracking: {new_mask_tracking.shape}")

    final_unique = torch.unique(new_mask_tracking)
    final_unique = final_unique[final_unique > 0]
    print(f"Final unique IDs: {len(final_unique)} IDs")
    if len(final_unique) > 0:
        print(f"Final ID range: {final_unique.min().item()} to {final_unique.max().item()}")
    
    total_zero_ratio = (new_mask_tracking == -1).float().mean().item()
    print(f"Background ratio: {total_zero_ratio:.4f}")
    
    mapping_info = {
        'method': 'keep_original',
        'original_to_mapped': {int(id_val): int(id_val) for id_val in final_unique},
        'mapped_to_original': {int(id_val): int(id_val) for id_val in final_unique},
        'total_original_ids': len(final_unique),
        'total_mapped_ids': len(final_unique)
    }
    
    return new_mask_tracking, mapping_info




if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default='City')
    parser.add_argument("--scene_name", type=str, default='scene1')
    parser.add_argument("--query_frame", type=int, default=0)
    parser.add_argument("--save_path", type=str, default='/YOU_PATH_TO_MASKOBJECT')
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--rate", type=int, default=1)
    parser.add_argument("--pseudo_label", action="store_true")
    
    args = parser.parse_args()
    scale_information = {
        0:"default",
        1:"small",
        2:"medium",
        3:"large",
    }
    feature_scale = str(args.scale)
    scale_name = scale_information[int(feature_scale)]
    save_path = f'{args.save_path}/{args.scene}/{args.scene_name}/objectmask_tracking/rate{args.rate}/{scale_name}/query_frame{args.query_frame}_contain_mask'
    print(f"save_path: {save_path}")
    print(f"args.query_frame: {args.query_frame}")
    os.makedirs(save_path, exist_ok=True)
    npz_path = f'/save/RS_3Dopen/Rebuttal/GoogleMap/{args.scene}/{args.scene_name}/pt_vis_video_rate{args.rate}_q{args.query_frame}.npz'
    data = np.load(npz_path)
    trajs = data['trajs']
    visconfs = data['visconfs']
    query_frame = int(npz_path.split('_')[-1].split('.')[0][1:])


    mask_information_path = f"/YOU_PATH_TO_MASK{args.scene}/{args.scene_name}/"
    mask_information = f"/YOU_PATH_TO_MASK/{args.scene}/{args.scene_name}/frame_{query_frame:03d}_s.npy"
    mask_information = np.load(mask_information)
    mask_information_s = mask_information[int(feature_scale)]
    print(f"mask_information_s: {mask_information_s.shape}")
    
        
    original_mask = torch.from_numpy(mask_information_s.astype(np.float32))
    original_mask = F.interpolate(original_mask.unsqueeze(0).unsqueeze(0), 
                                 size=(568, 1024), mode='nearest').squeeze(0).squeeze(0)
    original_mask = original_mask.to(torch.int32) 
    
    
    original_unique = torch.unique(original_mask)
    original_unique = original_unique[original_unique > 0]
  
    transformed_masks = transform_mask_by_trajectory(original_mask, trajs, visconfs, query_frame)
    np.save(f"{save_path}/transformed_masks.npy", transformed_masks)
    transformed_masks_path = f"{save_path}/transformed_masks.npy"
    transformed_masks = np.load(transformed_masks_path)
    hull_masks = create_trajectory_hull_masks(transformed_masks)


    new_mask_tracking, mapping_info = get_mask_information_preserve_ids(
        mask_information_path, query_frame, transformed_masks, hull_masks, args.scene, args.scene_name, feature_scale
    )

    new_mask_tracking_save_path = f"{save_path}/new_mask_tracking.npy"
    np.save(new_mask_tracking_save_path, new_mask_tracking)
    new_mask_tracking = np.load(new_mask_tracking_save_path)
    max_idx = new_mask_tracking.max()
    min_idx = new_mask_tracking.min()
    
    