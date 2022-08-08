import os
import sys
import numpy as np
from scipy.spatial import distance
import argparse
from argparse import Namespace
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# from tfrecord.torch.dataset import TFRecordDataset
from datasets.decoder_utils import _decoder_tf, _parse_fn, _decoder


def _get_opts() -> Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--datasets_path', type=str,
                        default='/media/hjx/2D97AD940A9AD661/v1.0', help='Path to TFRecoder data')
    parser.add_argument('--output_path', type=str, default='/media/hjx/2D97AD940A9AD661',
                        help='Path to write converted dataset to')
    parser.add_argument('--split_only', default=False, action='store_true',
                        help='Only split the region, only you have generate the dataset_mata.json')
    parser.add_argument('--visualize', default=False,
                        action='store_true', help='visualize the block or not')
    parser.add_argument('--radius', type=float, default=2.,
                        help='The radius of a block')
    parser.add_argument('--overlap', type=float,
                        default=0.5, help='overlap each block')
    parser.add_argument('--circle', default=True, action='store_false',
                        help='Split the region circle, or rectangel')

    return parser.parse_args()


def save_datasets_info(dataset_path, output_path, prefix):
    data_split = prefix.rsplit('_')[-1]
    assert not os.path.exists(Path(
        output_path)/('datasets_meta_'+ data_split +'.json')), 'exist dataset_mata.json in {}, please split only, or remove dataset_mata.json'.format(output_path)
    dataset_path = Path(dataset_path)
    all_records = sorted(dataset_path.iterdir())
    datasets_info = {}
    for record in tqdm(all_records):
        if prefix not in record.name:
            continue
        tfrecord_path = dataset_path / record
        tfrecord_name = tfrecord_path.name.split('.')[1]
        datasets_info[tfrecord_name] = {}
        # so slow to load, give up.......
        # dataset = TFRecordDataset(
            # tfrecord_path, index_path=None, compression_type="gzip", transform=_decoder)
        dataset = tf.data.TFRecordDataset(
                tfrecord_path,
                compression_type="GZIP",
            )
        dataset = dataset.map(_parse_fn)
        for batch in dataset:
            # use tf decode
            batch = _decoder_tf(batch)
            datasets_info[tfrecord_name].update({
                str(batch['image_hash']): {
                    'cam_idx': str(batch['cam_idx']),
                    'equivalent_exposure': float(batch['equivalent_exposure']),
                    'height': int(batch['height']),
                    'width': int(batch['width']),
                    'ray_origins': batch['ray_origins'][0, 0, :].tolist(),
                    'intrinsics': batch['intrinsics'][:2].tolist(),
                    'tfrecord_name': tfrecord_name
                }
            })
    with open(Path(output_path)/('datasets_meta_'+ data_split +'.json'), 'w') as f:
        json.dump(datasets_info, f, indent=2)
        print('save datasets_meta.json to {}....'.format(output_path))


def split_dataset(output_path, radius=2, overlap=0.5, prefix=None):        
    output_path = Path(output_path)
    positions = []
    data_split = prefix.rsplit('_')[-1]
    with open(output_path/('datasets_meta_'+ data_split +'.json'), 'r') as f:
        datasets_meta = json.load(f)
        for vals in datasets_meta.values():
            for val in vals.values():
                positions.append(val['ray_origins'])
    positions = np.array(positions)
    if data_split == 'train':
        centroids = get_centroids(positions, 2*radius*overlap)
    else:
        assert os.path.exists(Path(output_path)/('blocks_meta_'+ 'train' +'.json')), 'please generate train block first'
        with open(Path(output_path)/('blocks_meta_'+ 'train' +'.json'), 'r') as f:
            centroid_dicts = json.load(f)
            centroids = []
            for val in centroid_dicts.values():
                centroids.append(val['centroid'])
            centroids = np.array(centroids)
    # plot_cicle(centroids, positions)
    print('Load {} images'.format(len(positions)))
    print('Split to {} blocks.......'.format(len(centroids)))
    if os.path.exists(Path(output_path)/('blocks_meta_'+ data_split +'.json')):
        print('exists blocks_mata.json in {}, return.....'.format(output_path))
        return positions, centroids
    # scipy.spatial.distance.cdist
    blocks = {}
    blocks_dicts = {}
    for key, value in datasets_meta.items():
        sub_keys = np.array(list(value.keys()))
        sub_position = []
        for val in value.values():
            sub_position.append(val['ray_origins'])
        sub_position = np.array(sub_position)
        dist = distance.cdist(sub_position, centroids, 'euclidean')
        for i in range(len(centroids)):
            if blocks.get(i) is None:
                blocks[i] = []
                blocks_dicts[str(i)] = {'centroid': centroids[i].tolist(), 'block_items':{}}
            dist_i = dist[:, i]
            mask = dist_i < radius
            valid_key = sub_keys[mask]
            if len(valid_key) == 0:
                continue
            blocks[i] += valid_key.tolist()
            blocks_dicts[str(i)]['block_items'].update({
                    key: valid_key.tolist()
            }
            )
    with open(Path(output_path)/('blocks_meta_'+ data_split +'.json'), 'w') as f:
        json.dump(blocks_dicts, f, indent=2)
        print('save blocks_meta.json to {}...'.format(output_path))
    return positions, centroids


def get_centroids(positions, radius=2):
    print(radius)
    indices = np.argsort(positions[:, 0:1], axis=0)
    positions = positions[indices, :][:, 0, :]
    centroids = []
    for position in positions[:-1, :]:
        if len(centroids) == 0:
            centroids.append(position)
        else:
            dis = np.linalg.norm(centroids[-1] - position)
            if dis > radius:
                centroids.append(position)
    centroids.append(positions[-1])
    return np.array(centroids)


def plot_cicle(centroids, pos, radius=2):
    from matplotlib.patches import Circle
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for c in centroids:
        cir1 = Circle(xy=c[:-1], radius=radius, alpha=0.5)
        ax.add_patch(cir1)

        x, y = c[:-1]
        ax.plot(x, y, 'ro')
    plt.scatter(pos[:, 0], pos[:, 1])
    plt.axis('scaled')
    # changes limits of x or y axis so that equal increments of x and y have the same length
    plt.axis('equal')

    plt.show()


def main(args):
    data_split = ['train', 'validation']
    for s in data_split:
        prefix = 'waymo_block_nerf_mission_bay_'+s 
        if not args.split_only:
            save_datasets_info(args.datasets_path, args.output_path, prefix)
            positions, centroids = split_dataset(args.output_path, args.radius, args.overlap, prefix)
        else:
            assert os.path.exists(Path(args.output_path)/'datasets_meta.json'), 'please generate meta json, set split_only false'
            positions, centroids = split_dataset(args.output_path, args.radius, args.overlap, prefix)
    if args.visualize:
        plot_cicle(centroids, positions, args.radius)
if __name__ == '__main__':
    main(_get_opts())