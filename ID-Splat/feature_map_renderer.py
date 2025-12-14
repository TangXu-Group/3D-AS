#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

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
from utils.data_utils import read_classes_names, read_segmentation_maps, fielter, smooth, vis_seg, DistinctColors
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import jaccard_score, accuracy_score
import imageio
from loguru import logger

import colorsys
from utils.data_utils import read_classes_names, read_segmentation_maps, fielter, smooth, vis_seg, DistinctColors

def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)
    s = 0.5 + (id % 2) * 0.5
    l = 0.5

    rgb = np.zeros((3, ), dtype=np.uint8)
    if id==0:   #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb

def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask


def render_set(model_path, name, iteration, source_path, views, gaussians, pipeline, background, feature_level, seg_maps_all, idxes_all, seg_map_shape, dim_length, experiment_name):
    all_idx = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63]
    test_idx = [14, 21, 35, 42, 49, 56, 63]

    save_path = os.path.join(model_path, name, "ours_{}_langfeat_{}_{}".format(iteration, feature_level, experiment_name))
    render_path = os.path.join(save_path, "renders")
    gts_path = os.path.join(save_path, "gt")
    render_npy_path = os.path.join(save_path, "renders_npy")
    gts_npy_path = os.path.join(save_path,"gt_npy")
    gt_segmention_map_path = os.path.join(save_path, "gt_segmention_map_test")
    render_object_path = os.path.join(save_path, "render_objects")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    os.makedirs(render_npy_path, exist_ok=True)
    os.makedirs(gts_npy_path, exist_ok=True)
    os.makedirs(gt_segmention_map_path, exist_ok=True)
    os.makedirs(render_object_path, exist_ok=True)

    dc = DistinctColors()
    train_IoUs = []
    train_accuracies = []
    all_frame_idx = []
    if name == "train":
        views = [view for view in views if view.is_train_view]
    else:
        views = [view for view in views if view.is_test_view]
    print("This is views", len(views))
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background, include_feature=True)
        name = view.image_name.split(".")[0].split("_")[-1]
        name = int(name)
        rendering = render_pkg["render"]
        render_object = render_pkg["render_object"]
        gt_image = view.original_image.cpu().numpy()
        
        H_obj, W_obj = render_object.shape[1], render_object.shape[2]
        render_object_flat = render_object.squeeze(0).reshape(-1, render_object.shape[-1]).cpu().numpy()
        render_object_indices = np.argmax(render_object_flat, axis=-1)
        render_object_indices = rearrange(render_object_indices, "(h w) -> h w", h=H_obj, w=W_obj)
        render_object_vis = vis_seg(dc, render_object_indices, H=H_obj, W=W_obj)
        render_object_vis = visualize_obj(render_object_indices.astype(np.uint8))
        imageio.imwrite(os.path.join(render_object_path, f"{view.image_name.split('.')[0]}_object.png"), render_object_vis)
        
        H, W = rendering.shape[1], rendering.shape[2]
        gt = seg_maps_all[all_idx.index(name)]
        gt = torch.from_numpy(gt).to(view.data_device)
        gt = gt.float()
        gt = F.interpolate(gt.unsqueeze(0), size=(H, W), mode='nearest')
        gt = gt.squeeze(0)
        gt_mask = torch.sum(gt, dim=0)
        gt_mask = gt_mask > 0
            
        _, H, W = rendering.shape
        feature_map = F.interpolate(rendering.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
        feature_map_score = rearrange(feature_map, 'c h w -> (h w) c')
        class_index = torch.argmax(feature_map_score, dim=-1).cpu()  # [N1]
        gt_image = rearrange(gt_image, "c h w -> (h w) c")

        gt_seg_mask = gt
        gt_seg_mask = rearrange(gt_seg_mask, 'c h w -> (h w) c')
        gt_seg_mask = torch.sum(gt_seg_mask, dim=-1)
        gt_seg_mask = gt_seg_mask > 0
        gt_seg_mask_index = gt_seg_mask.nonzero(as_tuple=True)[0].cpu()  # Move to CPU

        gt_seg_mask_false = ~(gt_seg_mask > 0)
        gt_seg_mask_index_false = gt_seg_mask_false.nonzero(as_tuple=True)[0].cpu()  # Move to CPU

        class_index[gt_seg_mask_index_false] = -1
        segmentation_map = vis_seg(dc, class_index, H, W, rgb=gt_image)
        gt_segmention_map = torch.argmax(gt, dim=0)
        gt_segmention_map = rearrange(gt_segmention_map, "h w -> (h w) ")
        gt_segmention_map[gt_seg_mask_index_false] = -1
        gt_segmention_map = vis_seg(dc, gt_segmention_map.cpu(), H, W,rgb=gt_image)


        class_index = class_index[gt_seg_mask_index]
        
        gt_cpu = gt.cpu()
        gt_cpu = rearrange(gt_cpu, 'c h w -> (h w) c')
        gt_seg = gt_cpu[gt_seg_mask_index.numpy()].long()
        print("This is gt_seg_mask_index", gt_seg_mask_index.shape)
        print("This is class_index", class_index.shape)
        print("This is gt_image", gt_image.shape)
        one_hot = F.one_hot(class_index.long(), num_classes=gt_seg.shape[-1])  # [N1, n_classes]
        one_hot = one_hot.detach().cpu().numpy().astype(np.int8)
        train_IoUs.append(jaccard_score(gt_seg, one_hot, average=None))
        train_accuracies.append(accuracy_score(gt_seg, one_hot))

        if gt_segmention_map_path is not None:
            imageio.imwrite(f'{gt_segmention_map_path}/{view.image_name.split(".")[0]}_pred.png', segmentation_map)
            imageio.imwrite(f'{gt_segmention_map_path}/{view.image_name.split(".")[0]}_gt.png', gt_segmention_map)


            print("_pred", f"{gt_segmention_map_path}")
            
            mask_vis = np.zeros((H, W, 3), dtype=np.uint8)
            mask_vis[gt_mask.cpu().numpy()] = [0, 0, 0] 
            imageio.imwrite(f'{gt_segmention_map_path}/{view.image_name.split(".")[0]}_mask.png', mask_vis)
            
            pred_with_mask = segmentation_map.copy()
            pred_with_mask[~gt_mask.cpu().numpy()] = [0, 0, 0]  
            imageio.imwrite(f'{gt_segmention_map_path}/{view.image_name.split(".")[0]}_pred_masked.png', pred_with_mask)
            
            gt_with_mask = gt_segmention_map.copy()
            gt_with_mask[~gt_mask.cpu().numpy()] = [0, 0, 0]  
            imageio.imwrite(f'{gt_segmention_map_path}/{view.image_name.split(".")[0]}_gt_masked.png', gt_with_mask)
            
            # # 叠加mask后的预测图像和gt图像，gt系数为0.4
            # overlay_image = (pred_with_mask.astype(np.float32) * 0.35 + gt_image.astype(np.float32) * 0.65).astype(np.uint8)
            # imageio.imwrite(f'{gt_segmention_map_path}/{view.image_name.split(".")[0]}_pred_overlay.png', overlay_image)

            # overlay_image = (gt_with_mask.astype(np.float32) * 0.35 + gt_image.astype(np.float32) * 0.65).astype(np.uint8)
            # imageio.imwrite(f'{gt_segmention_map_path}/{view.image_name.split(".")[0]}_gt_overlay.png', overlay_image)


            

    logger.info(f'\n\niteration: {iteration}')
    logger.info(f'Train overall: mIoU={np.mean(train_IoUs)}, accuracy={np.mean(train_accuracies)}\n')

    for i, iou in enumerate(train_IoUs):
        logger.info(f'{time.strftime("%Y-%m-%d %H:%M:%S")} Test image {i}: mIoU={np.mean(iou)}, accuracy={train_accuracies[i]}')
        logger.info(f'{time.strftime("%Y-%m-%d %H:%M:%S")} Test classes iou: {iou}')

    # Save results to file
    results_file = os.path.join(gt_segmention_map_path, f'results_iteration_frame_28_{iteration}.txt')
    print("This is results_file", results_file)
    with open(results_file, 'w') as f:
        f.write(f'Iteration: {iteration}\n')
        f.write(f'Train overall: mIoU={np.mean(train_IoUs):.4f}, accuracy={np.mean(train_accuracies):.4f}\n\n')
        
        for i, iou in enumerate(train_IoUs):
            f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} Test image {i}: mIoU={np.mean(iou):.4f}, accuracy={train_accuracies[i]:.4f}\n')
            f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} Test classes iou: [{", ".join([f"{x:.6f}" for x in iou])}]\n\n')
    


def render_sets(dataset : ModelParams, opt : OptimizationParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, feature_level : int, experiment_name : str):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, include_feature=True)

        classes = read_classes_names(dataset.source_path)
        classes = sorted([item for item in classes if item != "frame" and item != "mask"])
        seg_maps_all, idxes_all, seg_map_shape = read_segmentation_maps(dataset.source_path, classes, downsample=2)
        # for item in [2,3,5,6,7,8,9]:
        #     seg_maps_all[item] = np.zeros_like(seg_maps_all[0])

        checkpoint = os.path.join(args.model_path, f'chkpnt{iteration}_langfeat_{experiment_name}_{feature_level}.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore_language_features(model_params, opt)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(args.model_path, "train", scene.loaded_iter, dataset.source_path, scene.getTrainCameras(), gaussians, pipeline, background, feature_level, seg_maps_all, idxes_all, seg_map_shape, dim_length=len(classes), experiment_name=experiment_name)

        if not skip_test:
             render_set(args.model_path, "test", scene.loaded_iter, dataset.source_path, scene.getTestCameras(), gaussians, pipeline, background, feature_level, seg_maps_all, idxes_all, seg_map_shape, dim_length=len(classes), experiment_name=experiment_name)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--experiment_name", default="all_gt", type=str)
    args = get_combined_args(parser)

    safe_state(args.quiet)

    render_sets(model.extract(args), opt.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.feature_level, args.experiment_name)