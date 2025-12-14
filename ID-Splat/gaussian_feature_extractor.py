
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from sklearn.decomposition import PCA
import torch.utils.dlpack
import matplotlib.pyplot as plt
import time
from einops import rearrange
from utils.data_utils import read_classes_names, read_segmentation_maps, fielter, smooth, vis_seg, DistinctColors
import torch.nn.functional as F

def extract_gaussian_features(model_path, iteration, source_path, views, gaussians, pipeline, background, feature_level, seg_maps_all, idxes_all, seg_map_shape, dim_length, experiment_name, scene_name, scene_idx):


    language_feature_save_path = os.path.join(model_path, f'chkpnt{iteration}_langfeat_{experiment_name}_{feature_level}.pth')
    train_views = [view for view in views if view.is_train_view]

    
    dc = DistinctColors()
    train_set = ["000"]
    test_set = [ "014", "021" , "035", "042", "049", "056", "063"]
    for frame_idx, frame_item_name in enumerate(train_set):
        frame_item = int(int(frame_item_name)/7)
        object_to_language_mapping = {} 
        for idx, view in enumerate(tqdm(train_views, desc="Rendering progress")):

            render_pkg= render(view, gaussians, pipeline, background)
            render_img = render_pkg["render"]
            H, W = render_img.shape[1], render_img.shape[2]
            objects_map = view.objects[frame_item]
            image_name = view.image_name.split(".")[0].split("_")[-1]

            gt_language_feature = seg_maps_all[idxes_all.index(int(image_name))]
            gt_language_feature = torch.from_numpy(gt_language_feature).to(view.data_device)
            gt_language_feature = gt_language_feature.float()
            gt_language_feature = F.interpolate(gt_language_feature.unsqueeze(0), size=(H, W), mode='nearest')
            gt_language_feature = gt_language_feature.squeeze(0)

            object_list = torch.unique(objects_map)
            gt_feature = rearrange(gt_language_feature, "c h w -> h w c")
            for item in object_list:
                if item.item() not in object_to_language_mapping:
                    object_to_language_mapping[item.item()] = []
                pixil_coor = torch.where(objects_map == item)
                object_features = gt_feature[pixil_coor[1], pixil_coor[2]]
                object_features = torch.mean(object_features, dim=0)
                object_to_language_mapping[item.item()].append(object_features)

            gt_language_feature = F.interpolate(gt_language_feature.unsqueeze(0), size=(H, W), mode='nearest')
            gt_language_feature = gt_language_feature.squeeze(0)
            gt_mask = torch.sum(gt_language_feature, dim=0)
            gt_mask = gt_mask > 0
            activated = render_pkg["info"]["activated"]
            significance = render_pkg["info"]["significance"]
            means2D = render_pkg["info"]["means2d"]

            mask = activated[0] > 0
            objects_map = objects_map.squeeze(0)
            if idx == 0:
                gaussians.accumulate_gaussian_feature_per_view(gt_language_feature = gt_language_feature.permute(1, 2, 0), gt_mask = gt_mask.squeeze(0), mask = mask, significance = significance[0,mask], means2D = means2D[0, mask], feature_dim =dim_length, objects_map=objects_map)
            else:
                gaussians.accumulate_gaussian_feature_per_view_object(gt_language_feature = gt_language_feature.permute(1, 2, 0), gt_mask = gt_mask.squeeze(0), mask = mask, significance = significance[0,mask], means2D = means2D[0, mask], feature_dim =dim_length, objects_map=objects_map)

        for k, v in object_to_language_mapping.items():
            v = torch.stack(v)
            probs = F.softmax(v, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            min_entropy_idx = torch.argmin(entropy)
            object_to_language_mapping[k] = v[min_entropy_idx]
            
        all_object_idx = object_to_language_mapping.keys()
        if 0 not in all_object_idx:
            object_to_language_mapping[0] = torch.zeros(dim_length, device='cuda')
        max_id = max(object_to_language_mapping.keys())
        mapping_tensor = torch.zeros(int(max_id+1), dim_length, device='cuda')
        for obj_id, feature_tensor in object_to_language_mapping.items():
            mapping_tensor[int(obj_id)] = feature_tensor

        pseudo_views = [view for view in views if not view.is_train_view]

        for idx, view in enumerate(tqdm(pseudo_views, desc="Rendering progress")):
            render_pkg= render(view, gaussians, pipeline, background)
            render_img = render_pkg["render"]
            original_image = view.original_image

            H, W = render_img.shape[1], render_img.shape[2]
            objects_map = view.objects[frame_item]
            objects_map = objects_map.squeeze(0)
            image_name = view.image_name.split(".")[0]

            gt_language_feature = torch.zeros(dim_length, H, W)
            y, x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            seg = objects_map[y, x].squeeze(-1).long()

            seg_object_idx = torch.unique(seg)
            valid_object_idx = []
            for obj_id in seg_object_idx:
                if obj_id.item() in object_to_language_mapping:
                    valid_object_idx.append(obj_id)
            seg_object_idx = torch.tensor(valid_object_idx, device=seg.device, dtype=seg.dtype)
            for obj_id in torch.unique(seg):
                if obj_id.item() not in object_to_language_mapping:
                    seg[seg == obj_id] = 0

            point_features_flat = mapping_tensor[seg] 
            gt_language_feature = rearrange(point_features_flat, "(h w) c -> c h w", h = H, w = W)
            gt_language_feature = F.interpolate(gt_language_feature.unsqueeze(0), size=(H, W), mode='nearest')
            gt_language_feature = gt_language_feature.squeeze(0)

            gt_mask = torch.sum(gt_language_feature, dim=0)
            gt_mask = gt_mask > 0


            activated = render_pkg["info"]["activated"]
            significance = render_pkg["info"]["significance"]
            means2D = render_pkg["info"]["means2d"]

            mask = activated[0] > 0
            gaussians.pseudo_accumulate_gaussian_feature_per_view(gt_language_feature_pseudo = gt_language_feature.permute(1, 2, 0), gt_mask = gt_mask.squeeze(0), mask = mask, significance = significance[0,mask], means2D = means2D[0, mask], feature_dim =dim_length, objects_map=objects_map, image_name=image_name, scene_name=scene_name, scene_idx=scene_idx)
        gaussians.feature_update(feature_dim =dim_length)

    gaussians.pseudo_finalize_gaussian_features(feature_dim=dim_length)

    refer_views = [view for view in views if view.is_test_view or view.is_train_view]

    for item in refer_views:
        id_refer_view = item.image_name.split(".")[0].split("_")[-1]
        print(id_refer_view, int(id_refer_view)/7)



    for refer_idx, refer_view in enumerate(tqdm(refer_views, desc="Rendering progress")):
        for idx, view in enumerate(tqdm(pseudo_views, desc="Rendering progress")):
            render_pkg= render(view, gaussians, pipeline, background)
            render_img = render_pkg["render"]
            original_image = view.original_image
            pseudo_frame_item = int(int(refer_view.image_name.split(".")[0].split("_")[-1])/7)
            H, W = render_img.shape[1], render_img.shape[2]
            objects_map = view.objects[pseudo_frame_item]
            objects_map = objects_map.squeeze(0)
            image_name = view.image_name.split(".")[0]

            gt_language_feature = torch.zeros(dim_length, H, W)
            y, x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            seg = objects_map[y, x].squeeze(-1).long()

            seg_object_idx = torch.unique(seg)
            valid_object_idx = []
            for obj_id in seg_object_idx:
                if obj_id.item() in object_to_language_mapping:
                    valid_object_idx.append(obj_id)
            seg_object_idx = torch.tensor(valid_object_idx, device=seg.device, dtype=seg.dtype)
            for obj_id in torch.unique(seg):
                if obj_id.item() not in object_to_language_mapping:
                    seg[seg == obj_id] = 0

            point_features_flat = mapping_tensor[seg] 
            gt_language_feature = rearrange(point_features_flat, "(h w) c -> c h w", h = H, w = W)
            gt_language_feature = F.interpolate(gt_language_feature.unsqueeze(0), size=(H, W), mode='nearest')
            gt_language_feature = gt_language_feature.squeeze(0)

            gt_mask = torch.sum(gt_language_feature, dim=0)
            gt_mask = gt_mask > 0

            activated = render_pkg["info"]["activated"]
            significance = render_pkg["info"]["significance"]
            means2D = render_pkg["info"]["means2d"]

            mask = activated[0] > 0
            gaussians.pseudo_accumulate_gaussian_feature_per_view_object(gt_language_feature_pseudo = gt_language_feature.permute(1, 2, 0), gt_mask = gt_mask.squeeze(0), mask = mask, significance = significance[0,mask], means2D = means2D[0, mask], feature_dim =dim_length, objects_map=objects_map, image_name=image_name)
        gaussians.pseudo_feature_update(feature_dim =dim_length)
    gaussians.pseudo_finalize_gaussian_features_object(feature_dim=dim_length)

    torch.save((gaussians.capture_language_feature(), 0), language_feature_save_path)
            
def process_scene_language_features(dataset : ModelParams, opt : OptimizationParams, iteration : int, pipeline : PipelineParams, feature_level : int, experiment_name : str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, include_feature=True)
        classes = read_classes_names(dataset.source_path)
        classes = sorted([item for item in classes if item != "frame" and item != "mask"])

        scene_name = dataset.source_path.split("/")[-2]
        scene_idx = dataset.source_path.split("/")[-1]


        seg_maps_all, idxes_all, seg_map_shape = read_segmentation_maps(dataset.source_path, classes, downsample=2)
        for item in [2,3,5,6,7,8,9]:
            seg_maps_all[item] = np.zeros_like(seg_maps_all[0])
        checkpoint = os.path.join(args.model_path, f'chkpnt{iteration}.pth')
        (model_params, _) = torch.load(checkpoint)
        gaussians.restore_rgb(model_params, opt)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        extract_gaussian_features(args.model_path, iteration, dataset.source_path, scene.getTrainCameras(), 
                                  gaussians, pipeline, background, feature_level, seg_maps_all, idxes_all, 
                                  seg_map_shape, dim_length=len(classes), experiment_name=experiment_name, scene_name=scene_name, scene_idx=scene_idx)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--experiment_name", default="all_gt", type=str)
    args = get_combined_args(parser)

    safe_state(args.quiet)

    process_scene_language_features(model.extract(args), opt.extract(args), args.iteration, pipeline.extract(args), args.feature_level, experiment_name=args.experiment_name)