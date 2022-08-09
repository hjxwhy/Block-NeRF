import cv2
import numpy as np
import tensorflow as tf
import torch
from einops import rearrange


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*map(fn, tup))

def _flatten(x):
    rays = []
    for items in x:
        flatten_items = [item.reshape([-1, item.shape[-1]]) for item in items]
        rays.append(np.hstack(flatten_items))
    return np.vstack(rays)



def _decoder(batch):
    height, width = int(batch["height"]), int(batch["width"])
    batch["image_hash"] = str(int(batch["image_hash"][0]))
    batch['cam_idx'] = int(batch['cam_idx'][0])
    batch["image"] = np.array(cv2.cvtColor(cv2.imdecode(np.frombuffer(batch["image"], np.uint8), -1), cv2.COLOR_BGR2RGB))
    batch['ray_dirs'] = batch['ray_dirs'].reshape([height, width, 3])
    batch['ray_origins'] = batch['ray_origins'].reshape([height, width, 3])
    batch["height"], batch["width"] = int(batch["height"]), int(batch["width"])
    batch["equivalent_exposure"] = float(batch["equivalent_exposure"])
    if batch.get('mask') is None:
        batch['mask'] = []
    else:
        batch['mask'] = batch['mask'].reshape([height, width, 1])

    return batch

def _decoder_tf(batch):
    height, width = int(batch["height"]), int(batch["width"])
    batch["height"], batch["width"] = int(batch["height"]), int(batch["width"])
    batch["cam_idx"] = int(batch["cam_idx"])
    batch["equivalent_exposure"] = float(batch["equivalent_exposure"])
    batch["image_hash"] = str(int(batch["image_hash"]))
    batch["intrinsics"] = tf.sparse.to_dense(batch["intrinsics"]).numpy()
    # output an RGB image.
    batch["image"] = np.array(tf.io.decode_png(batch["image"], channels=3, dtype=tf.dtypes.uint8, name=None))
    batch['ray_dirs'] = tf.sparse.to_dense(batch["ray_dirs"]).numpy().reshape([height, width, 3])
    batch['ray_origins'] = tf.sparse.to_dense(batch["ray_origins"]).numpy().reshape([height, width, 3])
    batch['mask'] = tf.sparse.to_dense(batch['mask']).numpy()
    return batch

def _parse_fn(record_bytes):
    return tf.io.parse_single_example(
    # Data
    record_bytes,
    # Schema
    {
        "image_hash": tf.io.FixedLenFeature([], dtype=tf.int64),
        "cam_idx": tf.io.FixedLenFeature([], dtype=tf.int64),
        "equivalent_exposure": tf.io.FixedLenFeature([], dtype=tf.float32),
        "height": tf.io.FixedLenFeature([], dtype=tf.int64),
        "width": tf.io.FixedLenFeature([], dtype=tf.int64),
        "image": tf.io.FixedLenFeature([], dtype=tf.string),
        "ray_origins": tf.io.VarLenFeature(tf.float32),
        "ray_dirs": tf.io.VarLenFeature(tf.float32),
        "intrinsics": tf.io.VarLenFeature(tf.float32),
        "mask": tf.io.VarLenFeature(tf.int64),
    },
)

# @torch.cuda.amp.autocast(dtype=torch.float32)
def axisangle_to_R(v):
    """
    Convert an axis-angle vector to rotation matrix
    from https://github.com/ActiveVisionLab/nerfmm/blob/main/utils/lie_group_helper.py#L47

    Inputs:
        v: (B, 3)
    
    Outputs:
        R: (B, 3, 3)
    """
    zero = torch.zeros_like(v[:, :1]) # (B, 1)
    skew_v0 = torch.cat([zero, -v[:, 2:3], v[:, 1:2]], 1) # (B, 3)
    skew_v1 = torch.cat([v[:, 2:3], zero, -v[:, 0:1]], 1)
    skew_v2 = torch.cat([-v[:, 1:2], v[:, 0:1], zero], 1)
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1) # (B, 3, 3)

    norm_v = rearrange(torch.norm(v, dim=1)+1e-7, 'b -> b 1 1')
    eye = torch.eye(3, device=v.device)
    R = eye + (torch.sin(norm_v)/norm_v)*skew_v + \
        ((1-torch.cos(norm_v))/norm_v**2)*(skew_v@skew_v)
    return R