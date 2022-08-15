#   This code is used for extract the image information of the tfrecords

import numpy as np
from tqdm import tqdm
import cv2
import os
import json
import glob
import tensorflow as tf
import time
import torch
from kornia import create_meshgrid

def get_cam_rays(H, W, K):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    directions = \
        torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1)  # (H, W, 3)
    # 求半径
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    return directions.numpy()


def get_Rotate(cam_ray_dir, world_ray_dir):
    cam_ray_dir = cam_ray_dir.reshape(-1, 3)
    world_ray_dir = world_ray_dir.reshape(-1, 3)

    world_r123 = np.mat(world_ray_dir[:, :1]).reshape(-1, 1)
    world_r456 = np.mat(world_ray_dir[:, 1:2]).reshape(-1, 1)
    world_r789 = np.mat(world_ray_dir[:, 2:3]).reshape(-1, 1)

    cam_dir = np.mat(cam_ray_dir)

    t1 = time.time()
    r123 = np.linalg.lstsq(cam_dir, world_r123, rcond=None)[0]
    r456 = np.linalg.lstsq(cam_dir, world_r456, rcond=None)[0]
    r789 = np.linalg.lstsq(cam_dir, world_r789, rcond=None)[0]
    t2 = time.time()

    R = np.zeros([3, 3])
    R[0:1, :] = r123.T
    R[1:2, :] = r456.T
    R[2:3, :] = r789.T

    loss = world_ray_dir - cam_ray_dir @ R.T
    ###cam*[R1,R2,R3]T=[world.x]
    print(f"loss:\t{np.abs(loss).mean()}, cost time:\t{t2 - t1}s ")

    return R


def decode_fn(record_bytes):
    return tf.io.parse_single_example(
        record_bytes,
        {  # mission bay:12个相机，在1.08km内拍摄了12000张图像
            "image_hash": tf.io.FixedLenFeature([], dtype=tf.int64),
            "cam_idx": tf.io.FixedLenFeature([], dtype=tf.int64),  # 0~12
            "equivalent_exposure": tf.io.FixedLenFeature([], dtype=tf.float32),
            "height": tf.io.FixedLenFeature([], dtype=tf.int64),
            "width": tf.io.FixedLenFeature([], dtype=tf.int64),
            "image": tf.io.FixedLenFeature([], dtype=tf.string),
            "ray_origins": tf.io.VarLenFeature(tf.float32),
            "ray_dirs": tf.io.VarLenFeature(tf.float32),
            "intrinsics": tf.io.VarLenFeature(tf.float32),
        }
    )


def handle_one_record(tfrecord, exist_imgs, index, stage):
    dataset = tf.data.TFRecordDataset(
        tfrecord,
        compression_type="GZIP",
    )
    dataset_map = dataset.map(decode_fn)

    os.makedirs(result_root_folder, exist_ok=True)
    meta_folder = os.path.join(result_root_folder, 'json')
    image_folder = os.path.join(result_root_folder, "images_"+stage)
    json_path = os.path.join(meta_folder, stage+".json")
    os.makedirs(meta_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)

    for batch in tqdm(dataset_map):
        image_name = str(int(batch["image_hash"]))

        if image_name + ".png" in exist_imgs:
            #print(f"\t{image_name}.png has been loaded!")
            continue

        index += 1
        imagestr = batch["image"]
        image = tf.io.decode_png(imagestr, channels=0, dtype=tf.dtypes.uint8, name=None)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(image_folder, f"{image_name}.png"), image)

        cam_idx = int(batch["cam_idx"])
        equivalent_exposure = float(batch["equivalent_exposure"])
        height, width = int(batch["height"]), int(batch["width"])
        intrinsics = tf.sparse.to_dense(batch["intrinsics"]).numpy()

        ray_origins = tf.sparse.to_dense(batch["ray_origins"]).numpy().reshape(height, width, 3)
        ray_dirs = tf.sparse.to_dense(batch["ray_dirs"]).numpy().reshape(height, width, 3)
        '''
        with open(os.path.join(image_folder, f"{image_name}_ray_origins.npy"), "wb") as f:
            np.save(f, ray_origins)
        with open(os.path.join(image_folder, f"{image_name}_ray_dirs.npy"), "wb") as f:
            np.save(f, ray_dirs)
        '''
        # 根据世界坐标系下的归一化rays_dir和相机坐标系下的rays_dir求得相机位姿
        K = np.zeros((3, 3), dtype=np.float32)
        # fx=focal,fy=focal,cx=img_w/2,cy=img_h/2
        K[0, 0] = intrinsics[0]
        K[1, 1] = intrinsics[1]
        K[0, 2] = width * 0.5
        K[1, 2] = height * 0.5
        K[2, 2] = 1

        cam_ray_dir = get_cam_rays(height, width, K)  # 归一化后的相机坐标系光线方向向量
        Rotate = get_Rotate(cam_ray_dir, ray_dirs)

        T = np.zeros([3, 4])
        T[:, :3] = Rotate
        T[:, 3:] = ray_origins[0][0].reshape(3, 1)

        cur_data_dict = {
            "image_name": image_name + ".png",
            "cam_idx": cam_idx,
            "equivalent_exposure": equivalent_exposure,
            "height": height,
            "width": width,
            "intrinsics": intrinsics.tolist(),
            "transform_matrix": T.tolist(),  # 相机的位姿
            "origin_pos": ray_origins[0][0].tolist()
        }

        train_meta[image_name] = cur_data_dict
        with open(json_path, "w") as fp:
            json.dump(train_meta, fp, indent=2)
            fp.close()

    return index


def get_the_current_index(root_dir):
    if os.path.exists(os.path.join(root_dir, 'json/train.json')):
        with open(os.path.join(root_dir, 'json/train.json'), 'r') as fp:
            meta = json.load(fp)
        return len(meta)
    return 0


if __name__ == "__main__":
    waymo_root_p = "../data/v1.0"
    result_root_folder = "../data/WaymoDataset"
    ori_waymo_data = sorted(glob.glob(os.path.join(waymo_root_p, "*")))
    exist_img_list = sorted(glob.glob(os.path.join(result_root_folder + "/images", "*.png")))

    exist_imgs = []
    for img_name in exist_img_list:
        exist_imgs.append(os.path.basename(img_name))

    index = get_the_current_index(result_root_folder)
    print(f"Has loaded {index} images!")
    train_meta = {}
    stages = ['train', 'validation']
    for stage in stages:
        for idx, tfrecord in enumerate(tqdm(ori_waymo_data)):
            if stage in tfrecord:
                print(tfrecord)
                print(f"Handling the {idx + 1}/{len(ori_waymo_data)} tfrecord")
                index = handle_one_record(tfrecord, exist_imgs, index, stage)