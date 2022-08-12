import torch
import json
from pathlib import Path
from tqdm import tqdm
from pytorch_lightning import LightningModule
from models.mip_nerf import MipNerf, Visibility
from models.mip import rearrange_render_image
from utils.metrics import calc_psnr
# from datasets import dataset_dict
from datasets.filesystem_dataset import FilesystemDataset, Rays
from datasets.decoder_utils import axisangle_to_R
from utils.lr_schedule import MipLRDecay
from torch.utils.data import DataLoader
from utils.vis import stack_rgb, visualize_depth
from typing import NamedTuple

class BlockNeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(BlockNeRFSystem, self).__init__()
        self.save_hyperparameters(hparams)
        self.train_randomized = hparams['train.randomized']
        self.val_randomized = hparams['val.randomized']
        self.val_chunk_size = hparams['val.chunk_size']
        self.batch_size = self.hparams['train.batch_size']
        self.image_hashs_dict, self.val_hashs_dict = self._get_image_metadata()
        assert len(self.image_hashs_dict)>0 and len(self.val_hashs_dict)>0, 'make sure train and val data large 0'
        kwargs = {
            'num_samples':hparams['nerf.num_samples'],
            'mlp_net_width':hparams['nerf.mlp.net_width'],
            'deg_exposure': hparams['nerf.mlp.deg_exposure'],
            'appearance_dim': hparams['nerf.mlp.appearance_dim'],
            'appearance_count':len(self.image_hashs_dict),
            'visib':Visibility()
        }
        self.mip_nerf = MipNerf(**kwargs)

        if self.hparams['optimize_ext']:
            N = len(self.image_hashs_dict)
            self.register_parameter('dR',
                torch.nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                torch.nn.Parameter(torch.zeros(N, 3, device=self.device)))        
        # self.visib = Visibility()

    def forward(self, batch_rays: NamedTuple, randomized: bool, white_bkgd: bool=False):
        if self.hparams['optimize_ext']:
            dR = axisangle_to_R(self.dR[batch_rays.image_indices.long().squeeze(-1)])
            dt = self.dT[batch_rays.image_indices.long().squeeze(-1)]
            directions = (batch_rays.directions.unsqueeze(1) @ dR.permute([0, 2, 1])).squeeze(1)
            origins = batch_rays.origins + dt
            batch_rays = Rays(origins, directions, directions, batch_rays.radii, batch_rays.lossmult, batch_rays.near, batch_rays.far, batch_rays.image_indices, batch_rays.exposures)

        res = self.mip_nerf(batch_rays, randomized, white_bkgd)  # num_layers result
        return res

    def setup(self, stage):
        dataset = FilesystemDataset
        self.train_dataset = dataset(self.image_hashs_dict, self.hparams['data_path'], self.hparams['img_nums'],\
                                     near=self.hparams['near'], far=self.hparams['far'])
        self.val_dataset = dataset(self.val_hashs_dict, self.hparams['data_path'], self.hparams['img_nums'], \
                                     near=self.hparams['near'], far=self.hparams['far'], split='val')
        
        self.refresh_datasets()


    def _get_image_metadata(self):
        dataset_path = Path(self.hparams['data_path'])
        with open(dataset_path/'blocks_meta_train.json', 'r') as f:
            block_train = json.load(f)
            block_train = block_train[str(self.hparams['block_id'])]
        with open(dataset_path/'blocks_meta_validation.json', 'r') as f:
            block_val = json.load(f)
            block_val = block_val[str(self.hparams['block_id'])]
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.mip_nerf.parameters(), lr=self.hparams['optimizer.lr_init'])
        scheduler = MipLRDecay(optimizer, self.hparams['optimizer.lr_init'], self.hparams['optimizer.lr_final'],
                               self.hparams['optimizer.max_steps'], self.hparams['optimizer.lr_delay_steps'],
                               self.hparams['optimizer.lr_delay_mult'])
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def refresh_datasets(self):
        self.train_dataset.load_chunk()
        self.val_dataset.load_chunk()

    def train_dataloader(self):
        # self.train_dataset.load_chunk()
        print('choose train index chosen_index-> {}'.format(self.train_dataset.chosen_index))
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=self.hparams['train.num_work'],
                          batch_size=self.hparams['train.batch_size'],
                          pin_memory=True)

    def val_dataloader(self):
        # must give 1 worker
        # self.val_dataset.load_chunk()
        print('choose val index chosen_index-> {}'.format(self.val_dataset.chosen_index))
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=1,
                          # validate one image (H*W rays) at a time
                          batch_size=1,
                          pin_memory=True,
                          persistent_workers=True)

    def training_step(self, batch, batch_nb):
        rays, rgbs = batch
        ret = self(rays, self.train_randomized)
        # calculate loss for coarse and fine
        mask = rays.lossmult
        if self.hparams['loss.disable_multiscale_loss']:
            mask = torch.ones_like(mask)

        loss = 0
        loss += self.hparams['loss.coarse_loss_mult'] * (mask * (ret['rgb_coarse'] - rgbs[..., :3]) ** 2).sum() / mask.sum()
        loss += (mask * (ret['rgb_fine'] - rgbs[..., :3]) ** 2).sum() / mask.sum()
       
        loss += 1e-6 * (mask *(ret[f'visib_trans_fine'].squeeze(-1)-ret[f'trans_fine'].detach())**2).sum() / mask.sum()
        loss += self.hparams['loss.coarse_loss_mult'] * 1e-6 * (mask *(ret[f'visib_trans_coarse'].squeeze(-1)-ret[f'trans_coarse'].detach())**2).sum() / mask.sum()
        
        if self.hparams['optimize_ext']:
            dR = self.dR[rays.image_indices.long().squeeze(-1)]
            dt = self.dT[rays.image_indices.long().squeeze(-1)]
            if self.global_step < 10000:
                # TODO this is may be a hyper param, the paper said 5000 steps
                loss += 1e5 * ((dR**2).mean() + (dt**2).mean())
            else:
                scale = (0.1 - 1e5) * (self.global_step-10000)/(self.hparams['optimizer.max_steps']-10000) + 1e5
                loss += scale * ((dR**2).mean() + (dt**2).mean())

        with torch.no_grad():
            psnr_fine = calc_psnr(ret['rgb_fine'], rgbs[..., :3])
        self.log('lr', self.optimizers().optimizer.param_groups[0]['lr'])
        self.log('train/loss', loss)
        self.log('train/psnr', psnr_fine, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        _, rgbs = batch
        rgb_gt = rgbs[..., :3]
        coarse_rgb, fine_rgb, val_mask = self.render_image(batch)

        val_mse_coarse = (val_mask * (coarse_rgb - rgb_gt)** 2).sum() / val_mask.sum()
        val_mse_fine = (val_mask * (fine_rgb - rgb_gt)** 2).sum() / val_mask.sum()

        val_loss = self.hparams['loss.coarse_loss_mult'] * val_mse_coarse + val_mse_fine

        val_psnr_fine = calc_psnr(fine_rgb, rgb_gt)

        log = {'val/loss': val_loss, 'val/psnr': val_psnr_fine}
        stack = stack_rgb(rgb_gt, coarse_rgb, fine_rgb)  # (3, 3, H, W)
        self.logger.experiment.add_images('val/GT_coarse_fine',
                                          stack, self.global_step)
        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val/loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val/psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)
        self.refresh_datasets()

    def render_image(self, batch):
        rays, rgbs = batch
        _, height, width, _ = rgbs.shape  # N H W C
        single_image_rays, val_mask = rearrange_render_image(
            rays, self.val_chunk_size)
        coarse_rgb, fine_rgb = [], []
        distances = []
        with torch.no_grad():
            for batch_rays in tqdm(single_image_rays):
                ret = self(batch_rays, self.val_randomized)
                coarse_rgb.append(ret['rgb_coarse'])
                fine_rgb.append(ret['rgb_fine'])
                distances.append(ret['depth_fine'])

        coarse_rgb = torch.cat(coarse_rgb, dim=0)
        fine_rgb = torch.cat(fine_rgb, dim=0)
        distances = torch.cat(distances, dim=0)
        distances = distances.reshape(1, height, width)  # H W
        distances = visualize_depth(distances)
        self.logger.experiment.add_image('distance', distances, self.global_step)

        coarse_rgb = coarse_rgb.reshape(
            1, height, width, coarse_rgb.shape[-1])  # N H W C
        fine_rgb = fine_rgb.reshape(
            1, height, width, fine_rgb.shape[-1])  # N H W C
        return coarse_rgb, fine_rgb, val_mask
