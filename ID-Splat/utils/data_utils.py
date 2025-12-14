import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os, random
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
import cv2
import math

def read_segmentation_maps(root_dir, classes, downsample):
    # if "Scene" in root_dir:
    #     segmentation_path = os.path.join(root_dir, 'segmentations')
    #     # get a list of all the folders in the directory
    #     folders = [f for f in os.listdir(segmentation_path) if os.path.isdir(os.path.join(segmentation_path, f))]
    #     # print(folders)
    #     seg_maps = []
    #     idxes = [] # the idx of the test imgs
    #     for folder in folders:
    #         idxes.append(int(folder))  # to get the camera id
    #         seg_for_one_image = []
    #         for class_name in classes:
    #             # check if the seg map exists
    #             seg_path = os.path.join(root_dir, f'segmentations/{folder}/{class_name}.png')
    #             # print(seg_path)
    #             if not os.path.exists(seg_path):
    #                 raise Exception(f'Image {class_name}.png does not exist')
    #             img = Image.open(seg_path).convert('L')
    #             # resize the seg map

    #             if downsample != 1.0:
    #                 img_wh = (img.width // downsample, img.height // downsample)
    #                 img = img.resize(img_wh, Image.NEAREST) # [W, H]
    #             img = (np.array(img) / 255.0).astype(np.int8) # [H, W]
    #             img = img.flatten() # [H*W]
    #             seg_for_one_image.append(img)

    #         seg_for_one_image = np.stack(seg_for_one_image, axis=0)
    #         seg_for_one_image = seg_for_one_image.transpose(1, 0)
    #         seg_maps.append(seg_for_one_image)

    #     seg_maps = np.stack(seg_maps, axis=0) # [n_frame, H*W, n_class]
    #     return seg_maps, idxes

    # if "Replica" in root_dir:
    #     return read_segmentation_maps_replica(root_dir, 2)

    # if "lerf" in root_dir:
    #     downsample = 0.5
    #     segmentation_path = os.path.join(root_dir, 'test_mask')
    #     width, height = Image.open(os.path.join(root_dir, 'images/frame_00001.jpg')).size
    #     folders = sorted(f for f in os.listdir(segmentation_path) if os.path.isdir(os.path.join(segmentation_path, f)))
    #     seg_maps = []
    #     idxes = []  # the idx of the test imgs
    #     for folder in folders:
    #         seg_for_one_image = []
    #         for class_name in classes:
    #             # check if the seg map exists
    #             seg_path = os.path.join(root_dir, f'test_mask/{folder}/{class_name}.png')
    #             if not os.path.exists(seg_path):
    #                 img = np.zeros((int(height * downsample), int(width * downsample))).flatten()
    #                 seg_for_one_image.append(img)
    #                 print(f'Image {seg_path} does not exist')
    #                 continue
    #             pil_image = Image.open(seg_path).convert('L')
    #             # resize the seg map
    #             if downsample != 1.0:
    #                 width, height = pil_image.size
    #                 newsize = (int(width * downsample), int(height * downsample + 0.5))
    #                 pil_image = pil_image.resize(newsize, resample=Image.Resampling.BILINEAR)
    #             img = (np.array(pil_image) / 255.0).astype(np.int8)  # [H, W]
    #             # print(img.shape)
    #             img = img.flatten()  # [H*W]
    #             seg_for_one_image.append(img)

    #         seg_for_one_image = np.stack(seg_for_one_image, axis=0)
    #         seg_for_one_image = seg_for_one_image.transpose(1, 0)
    #         seg_maps.append(seg_for_one_image)

    #     seg_maps = np.stack(seg_maps, axis=0)
    #     if "teatime" in root_dir:
    #         idxes = [178, 180]
    #     elif "ramen" in root_dir:
    #         idxes = [132, 133, 134]
    #     elif "figurines" in root_dir:
    #         idxes = [300, 301, 302, 303]

        idxes = []
        seg_maps = []
        if "RS" in root_dir:
            frame_name = ["frame_000", "frame_007", "frame_014", "frame_021", "frame_028", "frame_035", "frame_042", "frame_049", "frame_056", "frame_063"]
            for item in frame_name:
                idx = int(item.split("_")[-1])
                idxes.append(idx)
                seg_for_one_image = []
                for class_name in classes:
                    seg_path = os.path.join(root_dir, f'segmentation/{item}/{class_name}.png')
                    if not os.path.exists(seg_path):
                        raise Exception(f'Image {class_name}.png does not exist')
                    img = Image.open(seg_path).convert('L')

                    if downsample != 1.0:
                        img_wh = (img.width // downsample, img.height // downsample)
                        img = img.resize(img_wh, Image.NEAREST) # [W, H]
                    img = (np.array(img) / 255.0).astype(np.int8) # [H, W]
                    seg_map_shape = img.shape
                    # img = img.flatten() # [H*W]

                    seg_for_one_image.append(img)

                seg_for_one_image = np.stack(seg_for_one_image, axis=0)
                seg_maps.append(seg_for_one_image)

        seg_maps = np.stack(seg_maps, axis=0) # [n_class, H, W]

        return seg_maps, idxes, seg_map_shape
            
def read_segmentation_maps_train(root_dir, classes, downsample):
    # if "Scene" in root_dir:
    #     segmentation_path = os.path.join(root_dir, 'segmentations')
    #     # get a list of all the folders in the directory
    #     folders = [f for f in os.listdir(segmentation_path) if os.path.isdir(os.path.join(segmentation_path, f))]
    #     # print(folders)
    #     seg_maps = []
    #     idxes = [] # the idx of the test imgs
    #     for folder in folders:
    #         idxes.append(int(folder))  # to get the camera id
    #         seg_for_one_image = []
    #         for class_name in classes:
    #             # check if the seg map exists
    #             seg_path = os.path.join(root_dir, f'segmentations/{folder}/{class_name}.png')
    #             # print(seg_path)
    #             if not os.path.exists(seg_path):
    #                 raise Exception(f'Image {class_name}.png does not exist')
    #             img = Image.open(seg_path).convert('L')
    #             # resize the seg map

    #             if downsample != 1.0:
    #                 img_wh = (img.width // downsample, img.height // downsample)
    #                 img = img.resize(img_wh, Image.NEAREST) # [W, H]
    #             img = (np.array(img) / 255.0).astype(np.int8) # [H, W]
    #             img = img.flatten() # [H*W]
    #             seg_for_one_image.append(img)

    #         seg_for_one_image = np.stack(seg_for_one_image, axis=0)
    #         seg_for_one_image = seg_for_one_image.transpose(1, 0)
    #         seg_maps.append(seg_for_one_image)

    #     seg_maps = np.stack(seg_maps, axis=0) # [n_frame, H*W, n_class]
    #     return seg_maps, idxes

    # if "Replica" in root_dir:
    #     return read_segmentation_maps_replica(root_dir, 2)

    # if "lerf" in root_dir:
    #     downsample = 0.5
    #     segmentation_path = os.path.join(root_dir, 'test_mask')
    #     width, height = Image.open(os.path.join(root_dir, 'images/frame_00001.jpg')).size
    #     folders = sorted(f for f in os.listdir(segmentation_path) if os.path.isdir(os.path.join(segmentation_path, f)))
    #     seg_maps = []
    #     idxes = []  # the idx of the test imgs
    #     for folder in folders:
    #         seg_for_one_image = []
    #         for class_name in classes:
    #             # check if the seg map exists
    #             seg_path = os.path.join(root_dir, f'test_mask/{folder}/{class_name}.png')
    #             if not os.path.exists(seg_path):
    #                 img = np.zeros((int(height * downsample), int(width * downsample))).flatten()
    #                 seg_for_one_image.append(img)
    #                 print(f'Image {seg_path} does not exist')
    #                 continue
    #             pil_image = Image.open(seg_path).convert('L')
    #             # resize the seg map
    #             if downsample != 1.0:
    #                 width, height = pil_image.size
    #                 newsize = (int(width * downsample), int(height * downsample + 0.5))
    #                 pil_image = pil_image.resize(newsize, resample=Image.Resampling.BILINEAR)
    #             img = (np.array(pil_image) / 255.0).astype(np.int8)  # [H, W]
    #             # print(img.shape)
    #             img = img.flatten()  # [H*W]
    #             seg_for_one_image.append(img)

    #         seg_for_one_image = np.stack(seg_for_one_image, axis=0)
    #         seg_for_one_image = seg_for_one_image.transpose(1, 0)
    #         seg_maps.append(seg_for_one_image)

    #     seg_maps = np.stack(seg_maps, axis=0)
    #     if "teatime" in root_dir:
    #         idxes = [178, 180]
    #     elif "ramen" in root_dir:
    #         idxes = [132, 133, 134]
    #     elif "figurines" in root_dir:
    #         idxes = [300, 301, 302, 303]

        idxes = []
        seg_maps = []
        if "RS" in root_dir:
            frame_name = ["frame_000", "frame_007", "frame_014", "frame_021", "frame_028", "frame_035", "frame_042", "frame_049", "frame_056", "frame_063"]
            for item in frame_name:
                idx = int(item.split("_")[-1])
                idxes.append(idx)
                seg_for_one_image = []
                for class_name in classes:
                    # check if the seg map exists
                    seg_path = os.path.join(root_dir, f'segmentation/{item}/{class_name}.png')
                    if not os.path.exists(seg_path):
                        raise Exception(f'Image {class_name}.png does not exist')
                    img = Image.open(seg_path).convert('L')
                    # resize the seg map

                    if downsample != 1.0:
                        img_wh = (img.width // downsample, img.height // downsample)
                        img = img.resize(img_wh, Image.NEAREST) # [W, H]
                    img = (np.array(img) / 255.0).astype(np.int8) # [H, W]
                    seg_map_shape = img.shape
                    img = img.flatten() # [H*W]

                    seg_for_one_image.append(img)

                seg_for_one_image = np.stack(seg_for_one_image, axis=0)
                seg_for_one_image = seg_for_one_image.transpose(1, 0)
                seg_maps.append(seg_for_one_image)

        seg_maps = np.stack(seg_maps, axis=0) # [n_frame, H*W, n_class]
        return seg_maps, idxes, frame_name, seg_map_shape

def read_segmentation_maps_replica(root_dir, downsample):
    print("This is root_dir:{}".format(root_dir))
    segmentation_path = os.path.join(root_dir, 'merged_label')
    print("segmentation_path", segmentation_path)

    image_path = os.path.join(root_dir, 'images',"rgb_000.png" )
    image = Image.open(image_path).convert('RGB')

    # 将图像转换为 NumPy 数组以获取形状
    image_array = np.array(image)  # 输出形状为 (H, W, 3)

    # 获取形状
    h, w, C = image_array.shape  # C 是通道数（3），H 是高度，W 是宽度
    print(h, w)

    H, W = int((h + 0.5) / 2), int((w + 0.5) / 2)
    if h == 477:
        H = 238
    if h == 479:
        H = 240
    if w == 639:
        W = 320
    print(H, W)
    seg_maps = glob.glob(f'{segmentation_path}/*.pt')
    seg_maps = sorted(seg_maps,
           key=lambda file_name: int(file_name.split("/")[-1][:-3]))
    image_paths = glob.glob(f"{os.path.join(root_dir, 'images')}/*.pt")
    image_paths = sorted(image_paths,
           key=lambda file_name: int(file_name.split("_")[-1][:-4]))
    seg_for_one_image = []
    idxes = []
    # img_names = [path.split('_')[-1][:-4] for path in image_paths]
    # print(img_names)
    for idx, seg_path in enumerate(seg_maps):

        idxes.append(int(int(seg_path.split("/")[-1][:-3])/3))

        img = torch.load(seg_path).view(480, 640)

        img = F.interpolate(img[None,None, ...].float(), size=[H, W], mode='nearest').squeeze(0).squeeze(0)  # [W, H]
        # img = (np.array(img)).astype(np.int32)
        img = img.flatten()
        # img = np.searchsorted(semantic_classes, img)
        seg_for_one_image.append(img)
        # name = seg_path.split('.')[0].split('/')[-1]

    # feature_seg = torch.stack(seg_class_index, dim=0)
    seg_maps = torch.stack(seg_for_one_image, dim=0)
    print("This is the shape of seg_maps:{}".format(seg_maps.shape))
    return seg_maps, idxes

def read_classes_names(root_dir):
    # read class names
    if "Scene" in root_dir:
        with open(os.path.join(root_dir, 'segmentations/classes.txt'), 'r') as f:
            lines = f.readlines()
            classes = [line.strip() for line in lines]
            classes.sort()
    if "RS" in root_dir:
        seg_path = os.path.join(root_dir, "segmentation/frame_000/")
        all_files = os.listdir(seg_path)
        all_files = [item for item in all_files if 'png' in item]
        classes = [file.split(".")[0] for file in all_files]
        classes.sort()
        # with open(os.path.join(root_dir, 'segmentations/classes.txt'), 'r') as f:
        #     lines = f.readlines()
        #     classes = [line.strip() for line in lines]
        #     classes.sort()
    if "Replica" in root_dir:
        if 'room0' in root_dir:
            classes = ['ceiling', 'floor', 'plant', 'sofa', 'table', 'wall', 'windowpane', 'light']

        elif 'room1' in root_dir:
            classes = ['ceiling', 'floor', 'bed', 'wall', 'windowpane', 'light']

        elif 'office3' in root_dir:
            classes = ['ceiling', 'floor', 'table', 'wall', 'windowpane', 'light', 'tv-stand', 'cushion',
                            'sofa']

        elif 'office4' in root_dir:
            classes = ['ceiling', 'floor', 'table', 'wall', 'windowpane', 'light', 'tv-screen', 'chair',
                            'clock',
                            'bench']

        else:
            raise NotImplementedError

    if "lerf" in root_dir:
        if 'figurines' in root_dir:
            classes = "green apple;green toy chair;old camera;porcelain hand;red apple;red toy chair;rubber duck with red hat".split(
                ';')
        elif 'ramen' in root_dir:
            classes = "chopsticks;egg;glass of water;pork belly;wavy noodles in bowl;yellow bowl".split(';')
        elif 'teatime' in root_dir:
            classes = "apple;bag of cookies;coffee mug;cookies on a plate;paper napkin;plate;sheep;spoon handle;stuffed bear;tea in a glass".split(
                ';')
        else:
            raise NotImplementedError  # You can provide your text prompt here
        classes = sorted(classes)
    return classes

def fielter(valid_maps):
    for i, valid_map in enumerate(valid_maps):
        # print(valid_map.shape)
        scale = 15
        kernel = np.ones((scale, scale)) / (scale ** 2)
        np_relev = valid_map.cpu().numpy()
        avg_filtered = cv2.filter2D(np_relev, -1, kernel)
        avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
        valid_map = 0.5 * (avg_filtered + valid_map)
        valid_maps[i] = valid_map
    return valid_maps

def smooth(mask):
    h, w = mask.shape[:2]
    im_smooth = mask.copy()
    scale = 3
    for i in range(h):
        for j in range(w):
            square = mask[max(0, i-scale) : min(i+scale+1, h-1),
                          max(0, j-scale) : min(j+scale+1, w-1)]
            im_smooth[i, j] = np.argmax(np.bincount(square.reshape(-1)))
    return im_smooth

def hex_to_rgb(x):
    return [int(x[i:i + 2], 16) / 255 for i in (1, 3, 5)]


class DistinctColors:

    def __init__(self):
        colors = [
            '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f55031', '#911eb4', '#42d4f4', '#bfef45', '#fabed4',
            '#469990',
            '#dcb1ff', '#404E55', '#fffac8', '#809900', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9',
            '#f032e6',
            '#806020', '#ffffff',

            "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0030ED", "#3A2465", "#34362D", "#B4A8BD",
            "#0086AA",
            "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81", "#575329", "#00FECF", "#B05B6F", "#8CD0FF",
            "#3B9700",

            "#04F757", "#C8A1A1", "#1E6E00",
            "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
            "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
            "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
        ]
        self.hex_colors = colors
        # 0 = crimson / red, 1 = green, 2 = yellow, 3 = blue
        # 4 = orange, 5 = purple, 6 = sky blue, 7 = lime green
        self.colors = [hex_to_rgb(c) for c in colors]
        self.color_assignments = {}
        self.color_ctr = 0
        self.fast_color_index = torch.from_numpy(
            np.array([hex_to_rgb(colors[i % len(colors)]) for i in range(8096)] + [hex_to_rgb('#000000')]))

    def get_color(self, index, override_color_0=False):
        colors = [x for x in self.hex_colors]
        if override_color_0:
            colors[0] = "#3f3f3f"
        colors = [hex_to_rgb(c) for c in colors]
        if index not in self.color_assignments:
            self.color_assignments[index] = colors[self.color_ctr % len(self.colors)]
            self.color_ctr += 1
        return self.color_assignments[index]

    def get_color_fast_torch(self, index):
        return self.fast_color_index[index]

    def get_color_fast_numpy(self, index, override_color_0=False):
        index = np.array(index).astype(np.int32)
        if override_color_0:
            colors = [x for x in self.hex_colors]
            colors[0] = "#3f3f3f"
            fast_color_index = torch.from_numpy(
                np.array([hex_to_rgb(colors[i % len(colors)]) for i in range(8096)] + [hex_to_rgb('#000000')]))
            return fast_color_index[index % fast_color_index.shape[0]].numpy()
        else:
            return self.fast_color_index[index % self.fast_color_index.shape[0]].numpy()

    def apply_colors(self, arr):
        out_arr = torch.zeros([arr.shape[0], 3])

        for i in range(arr.shape[0]):
            out_arr[i, :] = torch.tensor(self.get_color(arr[i].item()))
        return out_arr

    def apply_colors_fast_torch(self, arr):
        return self.fast_color_index[arr % self.fast_color_index.shape[0]]

    def apply_colors_fast_numpy(self, arr):
        return self.fast_color_index.numpy()[arr % self.fast_color_index.shape[0]]


def get_boundary_mask(arr, dialation_size=1):
    import cv2
    arr_t, arr_r, arr_b, arr_l = arr[1:, :], arr[:, 1:], arr[:-1, :], arr[:, :-1]
    arr_t_1, arr_r_1, arr_b_1, arr_l_1 = arr[2:, :], arr[:, 2:], arr[:-2, :], arr[:, :-2]
    kernel = np.ones((dialation_size, dialation_size), 'uint8')
    if isinstance(arr, torch.Tensor):
        arr_t = torch.cat([arr_t, arr[-1, :].unsqueeze(0)], dim=0)
        arr_r = torch.cat([arr_r, arr[:, -1].unsqueeze(1)], dim=1)
        arr_b = torch.cat([arr[0, :].unsqueeze(0), arr_b], dim=0)
        arr_l = torch.cat([arr[:, 0].unsqueeze(1), arr_l], dim=1)

        arr_t_1 = torch.cat([arr_t_1, arr[-2, :].unsqueeze(0), arr[-1, :].unsqueeze(0)], dim=0)
        arr_r_1 = torch.cat([arr_r_1, arr[:, -2].unsqueeze(1), arr[:, -1].unsqueeze(1)], dim=1)
        arr_b_1 = torch.cat([arr[0, :].unsqueeze(0), arr[1, :].unsqueeze(0), arr_b_1], dim=0)
        arr_l_1 = torch.cat([arr[:, 0].unsqueeze(1), arr[:, 1].unsqueeze(1), arr_l_1], dim=1)

        boundaries = torch.logical_or(torch.logical_or(torch.logical_or(torch.logical_and(arr_t != arr, arr_t_1 != arr),
                                                                        torch.logical_and(arr_r != arr,
                                                                                          arr_r_1 != arr)),
                                                       torch.logical_and(arr_b != arr, arr_b_1 != arr)),
                                      torch.logical_and(arr_l != arr, arr_l_1 != arr))

        boundaries = boundaries.cpu().numpy().astype(np.uint8)
        boundaries = cv2.dilate(boundaries, kernel, iterations=1)
        boundaries = torch.from_numpy(boundaries).to(arr.device)
    else:
        arr_t = np.concatenate([arr_t, arr[-1, :][np.newaxis, :]], axis=0)
        arr_r = np.concatenate([arr_r, arr[:, -1][:, np.newaxis]], axis=1)
        arr_b = np.concatenate([arr[0, :][np.newaxis, :], arr_b], axis=0)
        arr_l = np.concatenate([arr[:, 0][:, np.newaxis], arr_l], axis=1)

        arr_t_1 = np.concatenate([arr_t_1, arr[-2, :][np.newaxis, :], arr[-1, :][np.newaxis, :]], axis=0)
        arr_r_1 = np.concatenate([arr_r_1, arr[:, -2][:, np.newaxis], arr[:, -1][:, np.newaxis]], axis=1)
        arr_b_1 = np.concatenate([arr[0, :][np.newaxis, :], arr[1, :][np.newaxis, :], arr_b_1], axis=0)
        arr_l_1 = np.concatenate([arr[:, 0][:, np.newaxis], arr[:, 1][:, np.newaxis], arr_l_1], axis=1)

        boundaries = np.logical_or(np.logical_or(
            np.logical_or(np.logical_and(arr_t != arr, arr_t_1 != arr), np.logical_and(arr_r != arr, arr_r_1 != arr)),
            np.logical_and(arr_b != arr, arr_b_1 != arr)), np.logical_and(arr_l != arr, arr_l_1 != arr)).astype(
            np.uint8)
        boundaries = cv2.dilate(boundaries, kernel, iterations=1)

    return boundaries


def vis_seg(dc, class_index, H, W, rgb=None, alpha=0.65):
    segmentation_map = dc.apply_colors_fast_torch(class_index)
    if rgb is not None:
        segmentation_map = segmentation_map * alpha + rgb * (1 - alpha)
    # boundaries = get_boundary_mask(class_index.view(H, W))
    segmentation_map = segmentation_map.reshape(H, W, 3)
    # segmentation_map[boundaries > 0, :] = 0
    segmentation_map = segmentation_map.detach().numpy().astype(np.float32)
    segmentation_map *= 255.
    segmentation_map = segmentation_map.astype(np.uint8)
    return segmentation_map
