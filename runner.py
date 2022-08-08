import datetime
import faulthandler
import math
import os
import random
import shutil
import signal
import sys
import json
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from datasets.filesystem_dataset import FilesystemDataset
# from mega_nerf.datasets.memory_dataset import MemoryDataset
# from mega_nerf.datasets.dataset_utils import get_rgb_index_mask
# from mega_nerf.image_metadata import ImageMetadata
from metrics import psnr, ssim, lpips
# from mega_nerf.misc_utils import main_print, main_tqdm
from models.model_utils import get_nerf, get_visibility
# from mega_nerf.ray_utils import get_rays, get_ray_directions
from models.mip import *
from models.lr_schedule import MipLRDecay
from rendering import render_rays


def main_print(log) -> None:
    if ('LOCAL_RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0:
        print(log)


def main_tqdm(inner):
    if ('LOCAL_RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0:
        return tqdm(inner)
    else:
        return inner


class Runner:
    def __init__(self, hparams: Namespace, set_experiment_path: bool = True):
        faulthandler.register(signal.SIGUSR1)

        if hparams.ckpt_path is not None:
            checkpoint = torch.load(hparams.ckpt_path, map_location='cpu')
            np.random.set_state(checkpoint['np_random_state'])
            torch.set_rng_state(checkpoint['torch_random_state'])
            random.setstate(checkpoint['random_state'])
        else:
            np.random.seed(hparams.random_seed)
            torch.manual_seed(hparams.random_seed)
            random.seed(hparams.random_seed)

        self.hparams = hparams

        if 'RANK' in os.environ:
            dist.init_process_group(
                backend='nccl', timeout=datetime.timedelta(0, hours=24))
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
            self.is_master = (int(os.environ['RANK']) == 0)
            # TODO 确定一下这样做是否合适，这里直接把迭代的轮数减少了
            world_size = int(os.environ['WORLD_SIZE'])
            self.hparams.train_iterations = self.hparams.train_iterations // world_size
        else:
            self.is_master = True

        self.is_local_master = ('RANK' not in os.environ) or int(
            os.environ['LOCAL_RANK']) == 0
        main_print(hparams)

        if set_experiment_path:
            self.experiment_path = self._get_experiment_path() if self.is_master else None
            self.model_path = self.experiment_path / 'models' if self.is_master else None

        self.writer = None

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.near = hparams.near

        if self.hparams.far is not None:
            self.far = hparams.far
        elif hparams.visibility:
            self.far = 1000
        else:
            self.far = 2

        main_print('Ray bounds: {}, {}'.format(self.near, self.far))

        self.train_items, self.val_items = self._get_image_metadata()

        self.nerf = get_nerf(hparams, len(self.train_items)).to(self.device)

        if 'RANK' in os.environ:
            self.nerf = torch.nn.parallel.DistributedDataParallel(self.nerf, device_ids=[int(os.environ['LOCAL_RANK'])],
                                                                  output_device=int(os.environ['LOCAL_RANK']))
        self.visibility = None
        if hparams.visibility:
            self.visibility = get_visibility(hparams).to(self.device)
            if 'RANK' in os.environ:
                self.visibility = torch.nn.parallel.DistributedDataParallel(self.visibility,
                                                                            device_ids=[
                                                                                int(os.environ['LOCAL_RANK'])],
                                                                            output_device=int(os.environ['LOCAL_RANK']))

    def train(self):
        self._setup_experiment_dir()

        scaler = torch.cuda.amp.GradScaler(enabled=self.hparams.amp)

        optimizers = {}
        optimizers['nerf'] = Adam(self.nerf.parameters(), lr=self.hparams.lr)

        if self.visibility is not None:
            optimizers['visibility'] = Adam(
                self.visibility.parameters(), lr=self.hparams.lr)

        if self.hparams.ckpt_path is not None:
            checkpoint = torch.load(self.hparams.ckpt_path, map_location='cpu')
            train_iterations = checkpoint['iteration']

            scaler_dict = scaler.state_dict()
            scaler_dict.update(checkpoint['scaler'])
            scaler.load_state_dict(scaler_dict)

            for key, optimizer in optimizers.items():
                optimizer_dict = optimizer.state_dict()
                optimizer_dict.update(checkpoint['optimizers'][key])
                optimizer.load_state_dict(optimizer_dict)
            discard_index = checkpoint['dataset_index'] if self.hparams.resume_ckpt_state else -1
        else:
            train_iterations = 0
            discard_index = -1

        schedulers = {}
        for key, optimizer in optimizers.items():
            schedulers[key] = MipLRDecay(optimizer, self.hparams.lr, self.hparams.lr_final,
                                         self.hparams.train_iterations, self.hparams.lr_delay_steps,
                                         self.hparams.lr_decay_factor)

        if self.hparams.dataset_type == 'filesystem':
            # Let the local master write data to disk first
            # We could further parallelize the disk writing process by having all of the ranks write data,
            # but it would make determinism trickier
            if 'RANK' in os.environ and (not self.is_local_master):
                dist.barrier()  # 设置栅栏，等待主线程数据加载完毕

            dataset = FilesystemDataset(
                self.train_items, self.hparams.dataset_path, self.hparams.img_nums)
            
            self.val_dataset = FilesystemDataset(
                self.val_items, self.hparams.dataset_path, 1, split='val')

            if self.hparams.ckpt_path is not None and self.hparams.resume_ckpt_state:
                dataset.set_state(checkpoint['chosen_index'], checkpoint['image_hash'])

            if 'RANK' in os.environ and self.is_local_master:
                dist.barrier()  # 让主线程也来栅栏这里，所有线程在一起走

        if self.is_master:
            pbar = tqdm(total=self.hparams.train_iterations)
            pbar.update(train_iterations)
        else:
            pbar = None

        while train_iterations < self.hparams.train_iterations:
            # If discard_index >= 0, we already set to the right chunk through set_state
            if self.hparams.dataset_type == 'filesystem' and discard_index == -1:
                dataset.load_chunk()

            if 'RANK' in os.environ:
                world_size = int(os.environ['WORLD_SIZE'])
                sampler = DistributedSampler(
                    dataset, world_size, int(os.environ['RANK']))
                assert self.hparams.batch_size % world_size == 0
                data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size // world_size, sampler=sampler,
                                         num_workers=6, pin_memory=True)
            else:
                data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=6,
                                         pin_memory=True)

            for dataset_index, item in enumerate(data_loader):
                if dataset_index <= discard_index:
                    continue

                discard_index = -1

                with torch.cuda.amp.autocast(enabled=self.hparams.amp):

                    metrics = self._training_step(
                        item['rgbs'].to(self.device, non_blocking=True),
                        item['rays'].to(self.device, non_blocking=True),
                    )

                    with torch.no_grad():
                        for key, val in metrics.items():
                            # a perfect reproduction will give PSNR = infinity
                            if key == 'psnr' and math.isinf(val):
                                continue

                            if not math.isfinite(val):
                                raise Exception(
                                    'Train metrics not finite: {}'.format(metrics))
                            if key == 'psnr':
                                pbar.set_postfix(psnr=val)

                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)

                scaler.scale(metrics['loss']).backward()

                for key, optimizer in optimizers.items():
                    scaler.step(optimizer)

                scaler.update()

                for scheduler in schedulers.values():
                    scheduler.step()

                train_iterations += 1
                if self.is_master:
                    pbar.update(1)
                    for key, value in metrics.items():
                        self.writer.add_scalar(
                            'train/{}'.format(key), value, train_iterations)

                    if train_iterations > 0 and train_iterations % self.hparams.ckpt_interval == 0:
                        self._save_checkpoint(optimizers, scaler, train_iterations, dataset_index, dataset.chosen_index, dataset.image_hash)

                if train_iterations > 0 and train_iterations % self.hparams.val_interval == 0:
                    self._run_validation(train_iterations)

                if train_iterations >= self.hparams.train_iterations:
                    break

        if 'RANK' in os.environ:
            dist.barrier()

        if self.is_master:
            pbar.close()
            self._save_checkpoint(optimizers, scaler, train_iterations, dataset_index, dataset.chosen_index, dataset.image_hash)


    # def eval(self):
    #     self._setup_experiment_dir()
    #     val_metrics = self._run_validation(0)
    #     self._write_final_metrics(val_metrics)

    # def _write_final_metrics(self, val_metrics: Dict[str, float]) -> None:
    #     if self.is_master:
    #         with (self.experiment_path / 'metrics.txt').open('w') as f:
    #             for key in val_metrics:
    #                 avg_val = val_metrics[key] / len(self.val_items)
    #                 message = 'Average {}: {}'.format(key, avg_val)
    #                 main_print(message)
    #                 f.write('{}\n'.format(message))

    #         self.writer.flush()
    #         self.writer.close()

    def _setup_experiment_dir(self) -> None:
        if self.is_master:
            self.experiment_path.mkdir()
            with (self.experiment_path / 'hparams.txt').open('w') as f:
                for key in vars(self.hparams):
                    f.write('{}: {}\n'.format(key, vars(self.hparams)[key]))
                if 'WORLD_SIZE' in os.environ:
                    f.write('WORLD_SIZE: {}\n'.format(os.environ['WORLD_SIZE']))

            with (self.experiment_path / 'command.txt').open('w') as f:
                f.write(' '.join(sys.argv))
                f.write('\n')

            self.model_path.mkdir(parents=True)

            with (self.experiment_path / 'image_indices.json').open('w') as f:
                json.dump(self.train_items, f, indent=2)
        self.writer = SummaryWriter(str(self.experiment_path / 'tb')) if self.is_master else None

        if 'RANK' in os.environ:
            dist.barrier()

    def _training_step(self, rgbs: torch.Tensor, rays: torch.Tensor) \
            -> Tuple[Dict[str, Union[torch.Tensor, float]], bool]:
        results = render_rays(nerf=self.nerf,
                              visibility=self.visibility,
                              rays=rays,
                              hparams=self.hparams,
                              randomized=self.nerf.training
                              )
        typ = 'fine' if 'fine' in results else 'coarse'
        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        metrics = {
            'psnr': psnr_,
        }

        photo_loss = F.mse_loss(results[f'rgb_{typ}'], rgbs, reduction='mean')
        metrics['photo_loss'] = photo_loss
        metrics['loss'] = photo_loss

        if f'visibility_trans_{typ}' in results:
            metrics['loss'] += 1e-6 * F.mse_loss(
                results[f'visibility_trans_{typ}'], results[f'trans_{typ}'].detach().unsqueeze(-1), reduction='mean')

        if 'rgb_coarse' in results and typ != 'coarse':
            coarse_loss = F.mse_loss(
                results['rgb_coarse'], rgbs, reduction='mean')

            metrics['coarse_loss'] = coarse_loss
            metrics['loss'] += 0.1 * coarse_loss

            if 'visibility_trans_coarse' in results:
                metrics['loss'] += 1e-6 * F.mse_loss(
                    results[f'visibility_trans_coarse'], results[f'trans_coarse'].detach().unsqueeze(-1), reduction='mean')

        return metrics

    def _run_validation(self, train_index: int) -> Dict[str, float]:
        self.val_dataset.load_chunk()

        if 'RANK' in os.environ:
            world_size = int(os.environ['WORLD_SIZE'])
            sampler = DistributedSampler(
                self.val_dataset, world_size, int(os.environ['RANK']))
            assert self.hparams.batch_size % world_size == 0
            data_loader = DataLoader(self.val_dataset, batch_size=1, sampler=sampler,
                                     num_workers=6, pin_memory=True)
        else:
            data_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=True, num_workers=6,
                                     pin_memory=True)
        with torch.inference_mode():
            self.nerf.eval()
            self.visibility.eval()

            val_metrics = defaultdict(float)
            base_tmp_path = None
            try:
                if 'RANK' in os.environ:
                    base_tmp_path = Path(
                        self.hparams.exp_name) / os.environ['TORCHELASTIC_RUN_ID']
                    metric_path = base_tmp_path / 'tmp_val_metrics'
                    image_path = base_tmp_path / 'tmp_val_images'
                    trans_path = base_tmp_path / 'tmp_val_trans'

                    world_size = int(os.environ['WORLD_SIZE'])
                    indices_to_eval = np.arange(
                        int(os.environ['RANK']), len(self.val_items), world_size)
                    if self.is_master:
                        base_tmp_path.mkdir()
                        metric_path.mkdir()
                        image_path.mkdir()
                    dist.barrier()

                for i, batch in main_tqdm(enumerate(data_loader)):

                    viz_rgbs = batch['rgbs'][0]
                    results = self.render_image(batch)
                    typ = 'fine' if 'rgb_fine' in results else 'coarse'
                    viz_result_rgbs = results[f'rgb_{typ}'].view(*viz_rgbs.shape).cpu()

                    eval_rgbs = viz_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()
                    eval_result_rgbs = viz_result_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()
                    mask = (batch['rays'][-1][:, :, viz_rgbs.shape[1] // 2:] > 0)
                    val_psnr = psnr(eval_result_rgbs.view(-1, 3)[mask.flatten(), :], eval_rgbs.view(-1, 3)[mask.flatten(), :])

                    metric_key = 'val/psnr/{}'.format(i)
                    if self.writer is None:
                        torch.save({'value': val_psnr, 'metric_key': metric_key, 'agg_key': 'val/psnr'},
                                   metric_path / 'psnr-{}.pt'.format(i))

                    val_metrics['val/psnr'] += val_psnr

                    val_ssim = ssim(eval_result_rgbs.view(*eval_rgbs.shape)*mask.float(), eval_rgbs*mask.float(), 1)

                    metric_key = 'val/ssim/{}'.format(i)
                    if self.writer is  None:
                        torch.save({'value': val_ssim, 'metric_key': metric_key, 'agg_key': 'val/ssim'},
                                   metric_path / 'ssim-{}.pt'.format(i))

                    val_metrics['val/ssim'] += val_ssim

                    viz_result_rgbs = viz_result_rgbs.view(
                        viz_rgbs.shape[0], viz_rgbs.shape[1], 3).cpu()
                    viz_depth = results[f'depth_{typ}']

                    img = Runner._create_result_image(
                        viz_rgbs, viz_result_rgbs, viz_depth)

                    if self.writer is not None:
                        self.writer.add_image(
                            'val/{}'.format(i), T.ToTensor()(img), train_index)
                    else:
                        img.save(str(image_path / '{}.jpg'.format(i)))
                    
                    if f'visibility_trans_{typ}' in results:
                        vi_trans = results[f'visibility_trans_{typ}']
                        vi_trans = vi_trans.squeeze(-1).sum(-1).float()

                        vi_trans_vis = Runner.visualize_scalars(
                            torch.log(vi_trans + 1e-8).view(viz_rgbs.shape[0], viz_rgbs.shape[1]).cpu())
                        vi_trans_vis = Image.fromarray(vi_trans_vis.astype(np.uint8))
                        if self.writer is not None:
                            self.writer.add_image(
                                'vis_trans/{}'.format(i), T.ToTensor()(vi_trans_vis), train_index)
                        else:
                            vi_trans_vis.save(str(trans_path / '{}.jpg'.format(i)))


                    del results

                if 'RANK' in os.environ:
                    dist.barrier()
                    if self.writer is not None:
                        for metric_file in metric_path.iterdir():
                            metric = torch.load(
                                metric_file, map_location='cpu')
                            # self.writer.add_scalar(
                            #     metric['metric_key'], metric['value'], train_index)
                            val_metrics[metric['agg_key']] += metric['value']
                        for image_file in image_path.iterdir():
                            img = Image.open(str(image_file))
                            self.writer.add_image(
                                'val/{}'.format(image_file.stem), T.ToTensor()(img), train_index)
                        
                        for image_file in trans_path.iterdir():
                            img = Image.open(str(image_file))
                            self.writer.add_image(
                                'vis_trans/{}'.format(image_file.stem), T.ToTensor()(img), train_index)

                        for key in val_metrics:
                            avg_val = val_metrics[key] / len(self.val_dataset)
                            self.writer.add_scalar('{}/avg'.format(key), avg_val, 0)

                    dist.barrier()
                
                else:
                    for key in val_metrics:
                        avg_val = val_metrics[key] / len(self.val_dataset)
                        self.writer.add_scalar('{}/avg'.format(key), avg_val, 0)

                self.nerf.train()
                self.visibility.train()
            finally:
                if self.is_master and base_tmp_path is not None:
                    shutil.rmtree(base_tmp_path)

            return val_metrics

    def _save_checkpoint(self, optimizers: Dict[str, any], scaler: GradScaler, train_index: int, dataset_index: int, chosen_index: int, image_hash: List) -> None:
        dict = {
            'model_state_dict': self.nerf.state_dict(),
            'scaler': scaler.state_dict(),
            'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
            'iteration': train_index,
            'torch_random_state': torch.get_rng_state(),
            'np_random_state': np.random.get_state(),
            'random_state': random.getstate(),
            'dataset_index': dataset_index,
            'chosen_index': chosen_index,
            'image_hash': image_hash
        }

        if self.visibility is not None:
            dict['visibility_model_state_dict'] = self.visibility.state_dict()

        torch.save(dict, self.model_path / '{}.pt'.format(train_index))

    def render_image(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        flatten_items = torch.hstack(
            [item.reshape([-1, item.shape[-1]]) for item in batch['rays']])

        with torch.cuda.amp.autocast(enabled=self.hparams.amp):
            results = {}

            if 'RANK' in os.environ:
                nerf = self.nerf.module
            else:
                nerf = self.nerf

            if self.visibility is not None and 'RANK' in os.environ:
                visibility_nerf = self.visibility.module
            else:
                visibility_nerf = self.visibility
            for i in main_tqdm(range(0, flatten_items.shape[0], self.hparams.image_pixel_batch_size)):
                result_batch = render_rays(nerf=nerf,
                                           visibility=visibility_nerf if self.visibility is not None else None,
                                           rays=flatten_items[i:i+self.hparams.image_pixel_batch_size].to(self.device),
                                           hparams=self.hparams,
                                           randomized=nerf.training,
                                           white_bkgd=False
                                           )
                for key, value in result_batch.items():
                    if key not in results:
                        results[key] = []

                    results[key].append(value.cpu())

            for key, value in results.items():
                results[key] = torch.cat(value)

            return results

    @staticmethod
    def _create_result_image(rgbs: torch.Tensor, result_rgbs: torch.Tensor, result_depths: torch.Tensor) -> Image:
        depth_vis = Runner.visualize_scalars(
            torch.log(result_depths + 1e-8).view(rgbs.shape[0], rgbs.shape[1]).cpu())
        images = (rgbs * 255, result_rgbs * 255, depth_vis)
        return Image.fromarray(np.concatenate(images, 1).astype(np.uint8))

    @staticmethod
    def visualize_scalars(scalar_tensor: torch.Tensor) -> np.ndarray:
        to_use = scalar_tensor.view(-1)
        while to_use.shape[0] > 2 ** 24:
            to_use = to_use[::2]

        mi = torch.quantile(to_use, 0.05)
        ma = torch.quantile(to_use, 0.95)

        scalar_tensor = (scalar_tensor - mi) / \
            max(ma - mi, 1e-8)  # normalize to 0~1
        scalar_tensor = scalar_tensor.clamp_(0, 1)

        scalar_tensor = ((1 - scalar_tensor) *
                         255).byte().numpy()  # inverse heatmap
        return cv2.cvtColor(cv2.applyColorMap(scalar_tensor, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)

    def _get_image_metadata(self):
        dataset_path = Path(self.hparams.dataset_path)
        with open(dataset_path/'blocks_meta_train.json', 'r') as f:
            block_train = json.load(f)
            block_train = block_train[str(self.hparams.block_id)]
        with open(dataset_path/'blocks_meta_validation.json', 'r') as f:
            block_val = json.load(f)
            block_val = block_val[str(self.hparams.block_id)]
        index = 0
        image_hashs_dict = {}
        for key, value in block_train['block_items'].items():
            for val in value:
                image_hashs_dict[val] = {
                    'tfrecord': key,
                    'type': 'train',
                    'index': index,
                }
                index += 1
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
                index += 1
        del block_train, block_val
        return image_hashs_dict, val_hashs_dict

    def _get_experiment_path(self) -> Path:
        exp_dir = Path(self.hparams.exp_name)
        exp_dir.mkdir(parents=True, exist_ok=True)
        existing_versions = [int(x.name) for x in exp_dir.iterdir()]
        version = 0 if len(existing_versions) == 0 else max(
            existing_versions) + 1
        experiment_path = exp_dir / str(version)
        return experiment_path
