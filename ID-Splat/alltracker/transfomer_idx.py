import os
import numpy as np
from tqdm import tqdm

def get_transformer_idx(scene, scene_name, query_frame, scale):
    scale_information = {
        0:"default",
        1:"small",
        2:"medium",
        3:"large",
    }
    scale_name = scale_information[scale]
    transformer_idx_path = f"/YOU_PATH/{scene}/{scene_name}/objectmask_tracking/rate1/{scale_name}/query_frame{query_frame}/new_mask_tracking.npy"
    print(f"transformer_idx_path: {transformer_idx_path}")
    transformer_idx = np.load(transformer_idx_path)
    print(f"transformer_idx: {transformer_idx.shape}")
    transformer_idx_remapped = remap_indices(transformer_idx)
    save_path = f"/YOU_PATH/{scene}/{scene_name}/objectmask_tracking/rate1/{scale_name}/query_frame{query_frame}/transformer_idx.npy"
    np.save(save_path, transformer_idx_remapped)
    return transformer_idx_remapped


def remap_indices(indices):

    remapped = np.copy(indices)
    
    mask = indices != -1
    unique_vals = np.unique(indices[mask])
    
    for i, val in tqdm(enumerate(unique_vals)):
        remapped[indices == val] = i
    
    return remapped


for scene in ["City", "Country", "Port"]:
	for scene_name in ["scene0", "scene1", "scene2"]:
	    for scale in [0, 1, 2, 3]:
		    for query_frame in [0, 7, 14, 21, 28, 35, 42, 49, 56, 63]:
			    transformer_idx = get_transformer_idx(scene, scene_name, query_frame, scale)

