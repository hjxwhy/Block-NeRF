import json
import random
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import cycle
from pathlib import Path
import numpy as np
import tensorflow as tf
from torch.utils.data import Dataset
from typing import Tuple, List, Optional, Dict, Union
from .decoder_utils import _decoder_tf, _parse_fn, _flatten, _decoder
from tfrecord.torch.dataset import TFRecordDataset

import collections
Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far', 'image_indices', 'exposures'))
Rays_keys = Rays._fields

def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*map(fn, tup))

class FilesystemDataset(Dataset):
    def __init__(self, metadata_dicts: Dict, data_root:str, img_nums: int, near:float, far:float, split:str = 'train'):
        super(FilesystemDataset, self).__init__()
        self.metadata_dicts = metadata_dicts
        self.image_hash = list(metadata_dicts.keys())
        self.img_nums = img_nums #if split=='train' else 1
        # self.image_hash = list(self.image_tfrecord_pairs.keys())
        self.last_chunk_index = len(self.image_hash) // img_nums * img_nums
        self.chunk_index = cycle(range(0, len(self.image_hash), img_nums))
        self.dataset_dir = Path(data_root)/'v1.0'
        self.split = split
        self.prefix = 'waymo_block_nerf_mission_bay_'
        self.chosen_index = None
        self.resume_from_ckpt = False
        self._chunk_load_executor = ThreadPoolExecutor(max_workers=1)
        self._chunk_future = self._chunk_load_executor.submit(self._load_chunk_inner)
        self.near = near
        self.far = far


    def load_chunk(self) -> None:
        # self._loaded_rays, self._loaded_rgbs, self.chosen_index = self._load_chunk_inner()
        self._loaded_rays, self._loaded_rgbs, self.chosen_index = self._chunk_future.result()
        self._chunk_future = self._chunk_load_executor.submit(self._load_chunk_inner)
    

    def set_state(self, chosen_index:int, image_hash:List) -> None:
        self.image_hash = image_hash
        self.resume_from_ckpt = True
        self.chosen_index = chosen_index
        self._chunk_future = self._chunk_load_executor.submit(self._load_chunk_inner)
        self.load_chunk()

    def _get_radii(self, w, h, focal):
        i, j = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
                np.arange(w, dtype=np.float32) + .5,  # X-Axis (columns)
                np.arange(h, dtype=np.float32) + .5,  # Y-Axis (rows)
                indexing='xy')
        direction = np.stack([(i-w/2)/focal, -(j-h/2)/focal, -np.ones_like(i)], axis=-1)
        dx = np.sqrt(np.sum((direction[:-1, :, :] - direction[1:, :, :]) ** 2, -1))
        dx = np.concatenate([dx, dx[-2:-1, :]], 0)
        radii = dx[..., None] * 2 / np.sqrt(12)
        return radii
    
    def _flatten(self, x):
        # Always flatten out the height x width dimensions
        x = [y.reshape([-1, y.shape[-1]]) for y in x]
        if self.split == 'train':
            # If global batching, also concatenate all data into one list
            x = np.concatenate(x, axis=0)
        return x

    def _load_chunk_inner(self):
        if self.resume_from_ckpt:
            next_index = self.chosen_index
            self.resume_from_ckpt = False
        else:
            next_index = next(self.chunk_index)
        chunk_image_hashs = self.image_hash[next_index:next_index+self.img_nums]
        if next_index == self.last_chunk_index:
            random.shuffle(self.image_hash)
        chunk_dict = {}
        for chunk_image_hash in chunk_image_hashs:
            hash_item = self.metadata_dicts[chunk_image_hash]
            tfrecord_name = hash_item['type'] + '.' + hash_item['tfrecord']
            if chunk_dict.get(tfrecord_name) is None:
                chunk_dict[tfrecord_name] = []
            chunk_dict[tfrecord_name].append(chunk_image_hash)

        images = []
        ray_origins = []
        radiis = []
        directions = []
        image_indices = []
        exposures = []
        masks = []
        nears = []
        fars = []
        for key, value in chunk_dict.items():
            tfrecord_path = self.dataset_dir / (self.prefix+key)
            # dataset = tf.data.TFRecordDataset(
            #     tfrecord_path,
            #     compression_type="GZIP",
            # )
            # dataset_map = dataset.map(_parse_fn)
            dataset_map = TFRecordDataset(tfrecord_path, index_path=None, compression_type="gzip", transform=_decoder)
            
            
            for batch in dataset_map:
                img_sh = str(int(batch['image_hash']))
                if img_sh in value:

                    # batch = _decoder_tf(batch)

                    direction = batch['ray_dirs']
                    radii = self._get_radii(batch['width'], batch['height'], batch['intrinsics'][0])
                    it = self.metadata_dicts[img_sh]
                    if it['type'] == 'val' and self.split=='train':
                        height, width = batch['height'], batch['width']
                        batch["image"] = batch['image'][:, width//2:]
                        batch['ray_origins'] = batch['ray_origins'][:, width//2:]
                        radii = radii[:, width//2:]
                        direction = direction[:, width//2:]
                        batch['mask'] = batch['mask'][:, width//2:]
                    images.append(batch["image"].astype(np.float32) / 255)
                    ray_origins.append(batch['ray_origins']*100)
                    radiis.append(radii)
                    directions.append(direction)
                    image_indices.append(np.ones_like(batch['ray_origins'][...,0:1])*it['index'])
                    exposures.append(np.ones_like(batch['ray_origins'][...,0:1])*batch['equivalent_exposure'])
                    nears.append(np.ones_like(batch['ray_origins'][...,0:1])*self.near)
                    fars.append(np.ones_like(batch['ray_origins'][...,0:1])*self.far)
                    if len(batch['mask']) > 0:
                        masks.append(batch['mask'])
                    else:
                        masks.append(np.ones_like(batch['ray_origins'][...,0:1]))

        
        rays = Rays(
            origins=ray_origins,
            directions=directions,
            viewdirs=directions,
            radii=radiis,
            lossmult=masks,
            near=nears,
            far=fars,
            image_indices=image_indices,
            exposures=exposures)

        if self.split == 'train':
            images = self._flatten(images)
            rays = namedtuple_map(self._flatten, rays)

        return rays, images, next_index

    def __len__(self):
        return len(self._loaded_rgbs)

    def __getitem__(self, index):
        rays = Rays(*[getattr(self._loaded_rays, key)[index] for key in Rays_keys])
        return rays, self._loaded_rgbs[index]

   

if __name__ == '__main__':
    import time

    dataset_path = Path('/home/hjx/Documents/nerf/block_nerf')
    with open(dataset_path/'blocks_meta_train.json', 'r') as f:
        block_train = json.load(f)
        block_train = block_train[str(3)]
    with open(dataset_path/'blocks_meta_validation.json', 'r') as f:
        block_val = json.load(f)
        block_val = block_val[str(3)]
    index = 0
    image_hashs_dict = {}
    for key, value in block_train['block_items'].items():
        for val in value:
            image_hashs_dict[val] = {
                'tfrecord': key,
                'type': 'train',
                'index': index,
            }
            index+=1
    val_hashs_dict = {}
    for key, value in block_val['block_items'].items():
        for val in value:
            image_hashs_dict[val] = {
                'tfrecord': key,
                'type': 'validation',
                'index': index,
            }
            val_hashs_dict[val] = {
                'tfrecord': key,
                'type': 'validation',
                'index': index,                   
            }
            index+=1

    filesystem = FilesystemDataset(image_hashs_dict, '/home/hjx/Documents/nerf/block_nerf', 5, 'val')
    from torch.utils.data import DistributedSampler, DataLoader
    # data_loader = DataLoader(filesystem, batch_size=1, shuffle=True, num_workers=6,
    #                                     pin_memory=True)
    for _ in range(10):
        t1 = time.time()
        filesystem.load_chunk()
        # data_loader = DataLoader(filesystem, batch_size=1, shuffle=False, num_workers=6,
        #                                 pin_memory=True)
        # for batch in data_loader:
        #     print(len(batch['rays']))
        print(time.time()-t1)