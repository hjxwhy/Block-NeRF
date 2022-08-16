import numpy as np
import os
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import json
from kornia import create_meshgrid
from tqdm import tqdm
import collections
from typing import Dict

Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far', 'image_indices', 'exposures'))
Rays_keys = Rays._fields


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*map(fn, tup))


def get_ray_directions(H, W, K):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    directions = \
        torch.stack([(i - cx) / fx, -(j - cy) / fy, -
        torch.ones_like(i)], -1)  # (H, W, 3)
    # 相机归一化平面与像素坐标系之间的转换
    # directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    return directions


def get_rays(directions, c2w):
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T  # (H, W, 3)

    # 在directions便已归一化
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape)  # (H, W, 3)

    return rays_o, rays_d


def find_idx_name(elements, img_name):
    #   用于在element里面根据img_name找到其idx
    for element in elements:
        if img_name in element:
            return element[1]
    return None


class WaymoDataset(Dataset):
    def __init__(self, metadata_dicts: Dict, root_dir, img_nums=1, split='train', block='block_0',
                 img_downscale=4,
                 near=1, far=1000,
                 test_img_name=None):
        self.metadata_dicts = metadata_dicts
        self.root_dir = root_dir
        self.split = split
        self.block = block
        self.img_downscale = img_downscale
        self.near = near
        self.far = far
        self.test_img_name = test_img_name

        self.transform = transforms.ToTensor()

        self.read_json()

    def _flatten(self, x):
        # Always flatten out the height x width dimensions
        x = [y.reshape([-1, y.shape[-1]]) for y in x]
        if self.split == 'train':
            # If global batching, also concatenate all data into one list
            x = np.concatenate(x, axis=0)
        return x

    def read_json(self):
        with open(os.path.join(self.root_dir, 'WaymoDataset/json/train.json'), 'r') as fp:
            meta_train = json.load(fp)
        with open(os.path.join(self.root_dir, 'WaymoDataset/json/validation.json'), 'r') as fp:
            meta_val = json.load(fp)

        images = []
        ray_origins = []
        radiis = []
        directions = []
        view_dirs = []
        image_indices = []
        exposures = []
        masks = []
        nears = []
        fars = []

        for key, values in self.metadata_dicts.items():
            if values['type'] == 'train':
                img_info = meta_train[str(key)]
            else:
                img_info = meta_val[str(key)]
            exposure = torch.tensor(img_info['equivalent_exposure'])
            c2w = torch.FloatTensor(img_info['transform_matrix'])

            width = img_info['width'] // self.img_downscale
            height = img_info['height'] // self.img_downscale

            img = Image.open(os.path.join(
                self.root_dir, 'WaymoDataset', 'images_'+values['type'], img_info['image_name'])).convert('RGB')
            if self.img_downscale != 1:
                img = img.resize((width, height), Image.Resampling.LANCZOS)
            img = self.transform(img)  # (3,h,w)
            img = img.permute([1, 2, 0])

            K = np.zeros((3, 3), dtype=np.float32)
            # fx=focal,fy=focal,cx=img_w/2,cy=img_h/2
            K[0, 0] = img_info['intrinsics'][0] / self.img_downscale
            K[1, 1] = img_info['intrinsics'][1] / self.img_downscale
            K[0, 2] = width * 0.5
            K[1, 2] = height * 0.5
            K[2, 2] = 1

            direction = get_ray_directions(height, width, K)
            rays_o, rays_d = get_rays(direction, c2w)
            view_dir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            rays_o *= 100

            # 求半径
            dx_1 = torch.sqrt(
                torch.sum((rays_d[:-1, :, :] - rays_d[1:, :, :]) ** 2, -1))
            dx = torch.cat([dx_1, dx_1[-2:-1, :]], 0)
            radii = dx[..., None] * 2 / torch.sqrt(torch.tensor(12))
            if values['type'] != 'train' and self.split == 'train':
                images.append(img[:, width // 2:, :])
                ray_origins.append(rays_o[:, width // 2:, :])
                radiis.append(radii[:, width // 2:, :])
                directions.append(direction[:, width // 2:, :])
                view_dirs.append(view_dir[:, width//2:, :])
                image_indices.append(values['index'] * torch.ones_like(direction[:, width // 2:, :1]))
                exposures.append(exposure * torch.ones_like(direction[:, width // 2:, :1]))
                masks.append(torch.ones_like(direction[:, width // 2:, :1]))
                nears.append(self.near * torch.ones_like(direction[:, width // 2:, :1]))
                fars.append(self.far * torch.ones_like(direction[:, width // 2:, :1]))
            else:
                images.append(img)
                ray_origins.append(rays_o)
                radiis.append(radii)
                directions.append(direction)
                view_dirs.append(view_dir)
                image_indices.append(values['index'] * torch.ones_like(direction[..., :1]))
                exposures.append(exposure * torch.ones_like(direction[..., :1]))
                masks.append(torch.ones_like(direction[..., :1]))
                nears.append(self.near * torch.ones_like(direction[..., :1]))
                fars.append(self.far * torch.ones_like(direction[..., :1]))

        rays = Rays(
            origins=ray_origins,
            directions=directions,
            viewdirs=view_dirs,
            radii=radiis,
            lossmult=masks,
            near=nears,
            far=fars,
            image_indices=image_indices,
            exposures=exposures)

        if self.split == 'train':
            images = self._flatten(images)
            rays = namedtuple_map(self._flatten, rays)
        self.images = images
        self.rays = rays

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rays = Rays(*[getattr(self.rays, key)[index] for key in Rays_keys])
        return rays, self.images[index]


if __name__ == "__main__":
    # test_train()
    with open('/media/hjx/DataDisk/waymo/WaymoDataset/json/train.json', 'r') as f:
        meta = json.load(f)
        print(meta['1613397460'])
