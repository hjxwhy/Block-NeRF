import json
import random
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import cycle
from pathlib import Path
import numpy as np
import tensorflow as tf
from torch.utils.data import Dataset
from decoder_utils import _decoder_tf, _parse_fn

class FilesystemDataset(Dataset):

    def __init__(self, metadata_path: str, block_id: int, img_nums: int, split:str = 'train'):
        super(FilesystemDataset, self).__init__()
        with open(Path(metadata_path)/('blocks_meta_'+ split +'.json'), 'r') as f:
            blocks_meta = json.load(f)
            block_meta = blocks_meta[str(block_id)]['block_items']
            assert len(block_meta) != 0, 'do not have items in block {} '.format(block_id)
            del blocks_meta
        self.block_id = block_id
        self.image_tfrecord_pairs = {}
        for key, value in block_meta.items():
            for v in value:
                self.image_tfrecord_pairs[v] = key
        self.img_nums = img_nums if split=='train' else 1
        self.image_hash = list(self.image_tfrecord_pairs.keys())
        self.last_chunk_index = len(self.image_hash) // img_nums * img_nums
        self.chunk_index = cycle(range(0, len(self.image_hash), img_nums))
        self.dataset_dir = Path(metadata_path)/'v1.0'
        self.split = split
        if split == 'train':
            self.prefix = 'waymo_block_nerf_mission_bay_train'
        else:
            self.prefix = 'waymo_block_nerf_mission_bay_validation'

        self._chunk_load_executor = ThreadPoolExecutor(max_workers=1)
        self._chunk_future = self._chunk_load_executor.submit(self._load_chunk_inner)

    def load_chunk(self) -> None:
        self._loaded_rays, self._loaded_rgbs = self._chunk_future.result()
        self._chunk_future = self._chunk_load_executor.submit(self._load_chunk_inner)

    def _load_chunk_inner(self):
        next_index = next(self.chunk_index)
        chunk_image_hashs = self.image_hash[next_index:next_index+self.img_nums]
        if next_index == self.last_chunk_index:
            random.shuffle(self.image_hash)
        chunk_dict = {}
        for chunk_image_hash in chunk_image_hashs:
            tfrecord_name = self.image_tfrecord_pairs[chunk_image_hash]
            if chunk_dict.get(tfrecord_name) is None:
                chunk_dict[tfrecord_name] = []
            chunk_dict[tfrecord_name].append(chunk_image_hash)
        images = []
        ray_origins = []
        radiis = []
        directions = []
        masks = []
        for key, value in chunk_dict.items():
            tfrecord_path = self.dataset_dir / (self.prefix+'.'+key)
            dataset = tf.data.TFRecordDataset(
                tfrecord_path,
                compression_type="GZIP",
            )
            dataset_map = dataset.map(_parse_fn)
            for batch in dataset_map:
                if str(int(batch['image_hash'])) in value:
                    batch = _decoder_tf(batch)
                    direction = batch['ray_dirs'] / np.abs(batch['ray_dirs'][:, :, 2:3])
                    dx = np.sqrt(np.sum((direction[:-1, :, :] - direction[1:, :, :]) ** 2, -1))
                    dx = np.concatenate([dx, dx[-2:-1, :]], 0)
                    radii = dx[..., None] * 2 / np.sqrt(12)
                    images.append(batch["image"])
                    ray_origins.append(batch['ray_origins'])
                    radiis.append(radii)
                    directions.append(direction)
                    if len(batch['mask'] > 0):
                        masks.append(batch['mask'])
        images = np.concatenate([img.reshape([-1, 3]) for img in images], dtype=np.float32) / 255
        ray_origins = np.concatenate([orig.reshape([-1, 3]) for orig in ray_origins], dtype=np.float32)
        radiis = np.concatenate([r.reshape([-1, 1]) for r in radiis], dtype=np.float32)
        directions = np.concatenate([d.reshape([-1, 3]) for d in directions], dtype=np.float32)
        rays = np.hstack([ray_origins, directions, radiis])
        del ray_origins, radiis, directions
        if len(masks) > 0:
            masks = np.concatenate([mask.reshape([-1, 1]) for mask in masks], dtype=np.float32)
            images = np.concatenate([images, masks], axis=1)
        return rays, images

    def __len__(self):
        return len(self._loaded_rays)

    def __getitem__(self, idx):
        return {
            'rgbs': self._loaded_rgbs[idx],
            'rays': self._loaded_rays[idx],
        }

if __name__ == '__main__':
    import time
    filesystem = FilesystemDataset('/home/hjx/Documents/nerf/block_nerf', 3, 5, 'train')
    for _ in range(10):
        t1 = time.time()
        filesystem.load_chunk()
        print(time.time()-t1)