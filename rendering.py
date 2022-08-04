import os
from argparse import Namespace
from typing import Optional, Dict, Callable, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from models.mip import sample_along_rays, resample_along_rays, integrated_pos_enc, pos_enc, volumetric_rendering
# from mega_nerf.spherical_harmonics import eval_sh

TO_COMPOSITE = {'rgb', 'depth'}
INTERMEDIATE_KEYS = {'zvals_coarse', 'raw_rgb_coarse', 'raw_sigma_coarse', 'depth_real_coarse'}


def distloss(weight, samples):
    '''
    mip-nerf 360 sec.4
    weight: [B, N]
    samples:[N, N+1]
    '''
    interval = samples[:, 1:] - samples[:, :-1]
    mid_points = (samples[:, 1:] + samples[:, :-1]) * 0.5
    loss_uni = (1/3) * (interval * weight.pow(2)).sum(-1).mean()
    ww = weight.unsqueeze(-1) * weight.unsqueeze(-2)          # [B,N,N]
    mm = (mid_points.unsqueeze(-1) - mid_points.unsqueeze(-2)).abs()  # [B,N,N]
    loss_bi = (ww * mm).sum((-1,-2)).mean()
    return loss_uni + loss_bi

def render_rays(nerf: nn.Module,
                visibility: Optional[nn.Module],
                rays: torch.Tensor,
                hparams: Namespace,
                randomized:bool,
                white_bkgd:bool=True,
                # sphere_center: Optional[torch.Tensor],
                # sphere_radius: Optional[torch.Tensor],
                # get_depth: bool,
                # get_depth_variance: bool,
                # get_bg_fg_rgb: bool
                ):# -> Tuple[Dict[str, torch.Tensor], bool]:
    #rays: ray_origins[i], directions[i], radiis[i], image_indices[i], exposures[i], mask[i](option)
    ret = {}
    t_samples, weights = None, None
    stage = None
    for i_level in range(2):
            if i_level == 0:
                # Stratified sampling along rays
                stage = 'coarse'
                t_samples, means_covs = sample_along_rays(
                    rays[:, :3], #.origins
                    rays[:,3:6], #.directions,
                    rays[:, 6:7], #.radii,
                    hparams.num_samples,
                    hparams.near,
                    hparams.far,
                    randomized,
                    False,
                )
            else:
                ...
    #             stage = 'fine'
    #             t_samples, means_covs = resample_along_rays(
    #                 rays[:, :3], #.origins
    #                 rays[:,3:6], #.directions,
    #                 rays[:, 6:7], #.radii,
    #                 t_samples,
    #                 weights,
    #                 randomized,
    #                 # resample_padding=self.resample_padding,
    #             )
            samples_enc = integrated_pos_enc(
                means_covs,
                0,
                hparams.max_deg_point,
            )  # samples_enc: [B, N, 2*3*L]  L:(max_deg_point - min_deg_point)

            # Point attribute predictions
            viewdirs = rays[:, 3:6] / torch.norm(rays[:, 3:6], dim=-1, keepdim=True)
            viewdirs_enc = pos_enc(
                viewdirs,
                min_deg=0,
                max_deg=hparams.deg_view,
                append_identity=True,
            )
            if hparams.use_exposure:
                exposure_enc = pos_enc(rays[:,8:9], 
                                        min_deg=0, 
                                        max_deg=hparams.deg_exposure, 
                                        append_identity=False)

            Batch, N_samples, _ = samples_enc.shape
            images_id = rays[:, 7:8]
            # samples_enc = samples_enc.view(-1, samples_enc.shape[-1])
            viewdirs_enc = viewdirs_enc.unsqueeze(1).repeat(1, N_samples, 1)
            exposure_enc = exposure_enc.unsqueeze(1).repeat(1, N_samples, 1) if hparams.use_exposure else None
            images_id = images_id.repeat(1, N_samples) if hparams.use_appearance else None
            rgb, density = nerf(samples_enc, viewdirs_enc, images_id, exposure_enc, randomized)
            
            comp_rgb, distance, weights, trans = volumetric_rendering(
                rgb,
                density,
                t_samples,
                rays[:,3:6],
                white_bkgd=white_bkgd,
            )
            ret[f'rgb_{stage}'] = comp_rgb
            # ret[f'dist_{stage}'] = distance
            # ret[f'weights_{stage}'] = weights
            ret[f'trans_{stage}'] = trans

            visibility_trans = visibility(samples_enc, viewdirs_enc)
            ret[f'visibility_trans_{stage}'] = visibility_trans

    return ret