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
import math
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from gsplat import rasterization

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False, include_feature = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    means3D = pc.get_xyz
    opacity = pc.get_opacity

    scales = pc.get_scaling * scaling_modifier
    rotations = pc.get_rotation

    if include_feature:
        features = pc.get_language_feature
    else:
        if override_color is not None:
            colors = override_color # [N, 3]
            sh_degree = None
        else:
            colors = pc.get_features # [N, K, 3]
            sh_degree = pc.active_sh_degree

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]
    if include_feature:
        render_colors, render_alphas, info = rasterization(
            means=means3D,  # [N, 3]
            quats=rotations,  # [N, 4]
            scales=scales,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=features, # [N, D]
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False
        )
        render_object, render_alphas, info = rasterization(
            means=means3D,  # [N, 3]
            quats=rotations,  # [N, 4]
            scales=scales,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=pc.get_language_object.squeeze(1), # [N, D]
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False
        )
    else:
        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        render_colors, render_alphas, info = rasterization(
            means=means3D,  # [N, 3]
            quats=rotations,  # [N, 4]
            scales=scales,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=colors,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            backgrounds=bg_color[None],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            sh_degree=sh_degree,
        )
        render_object = None

    rendered_image = render_colors[0].permute(2, 0, 1)
    radii = info["radii"].squeeze(0) # [N,]
    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass
    out = {
        "render": rendered_image,
        "viewspace_points": info["means2d"],
        "visibility_filter" : radii > 0,
        "radii": radii,
        "info": info,
        "render_object": render_object,
        }
    
    return out
