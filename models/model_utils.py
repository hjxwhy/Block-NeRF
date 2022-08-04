from argparse import Namespace

import torch
from torch import nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

# from mega_nerf.models.cascade import Cascade
# from mega_nerf.models.mega_nerf import MegaNeRF
from models.mip_nerf import MipNerf, Visibility

def get_visibility(hparams: Namespace):
    return _get_nerf_inner(hparams, 0, 0, 0, 'visibility_model_state_dict')


def get_nerf(hparams: Namespace, appearance_count: int) -> nn.Module:
    return _get_nerf_inner(hparams, appearance_count, hparams.layer_dim, 3, 'model_state_dict')


def get_bg_nerf(hparams: Namespace, appearance_count: int) -> nn.Module:
    return _get_nerf_inner(hparams, appearance_count, hparams.bg_layer_dim, 4, 'bg_model_state_dict')


def _get_nerf_inner(hparams: Namespace, appearance_count: int, layer_dim: int, xyz_dim: int,
                    weight_key: str) -> nn.Module:
    if hparams.container_path is not None:
        ...
    #     container = torch.jit.load(hparams.container_path, map_location='cpu')
    #     if xyz_dim == 3:
    #         return MegaNeRF([getattr(container, 'sub_module_{}'.format(i)) for i in range(len(container.centroids))],
    #                         container.centroids, hparams.boundary_margin, False, container.cluster_2d)
    #     else:
    #         return MegaNeRF([getattr(container, 'bg_sub_module_{}'.format(i)) for i in range(len(container.centroids))],
    #                         container.centroids, hparams.boundary_margin, True, container.cluster_2d)
    # elif hparams.use_cascade:
    #     nerf = Cascade(
    #         _get_single_nerf_inner(hparams, appearance_count,
    #                                layer_dim if xyz_dim == 4 else layer_dim,
    #                                xyz_dim),
    #         _get_single_nerf_inner(hparams, appearance_count, layer_dim, xyz_dim))
    # elif hparams.train_mega_nerf is not None:
    #     centroid_metadata = torch.load(hparams.train_mega_nerf, map_location='cpu')
    #     centroids = centroid_metadata['centroids']
    #     nerf = MegaNeRF(
    #         [_get_single_nerf_inner(hparams, appearance_count, layer_dim, xyz_dim) for _ in
    #          range(len(centroids))], centroids, 1, xyz_dim == 4, centroid_metadata['cluster_2d'], True)
    else:
        if appearance_count !=0:
            nerf = _get_single_nerf_inner(hparams, appearance_count, layer_dim, xyz_dim)
        else:
            nerf = Visibility()

    if hparams.ckpt_path is not None:
        state_dict = torch.load(hparams.ckpt_path, map_location='cpu')[weight_key]
        consume_prefix_in_state_dict_if_present(state_dict, prefix='module.')

        model_dict = nerf.state_dict()
        model_dict.update(state_dict)
        nerf.load_state_dict(model_dict)

    return nerf


def _get_single_nerf_inner(hparams: Namespace, appearance_count: int, layer_dim: int, xyz_dim: int) -> nn.Module:

    return MipNerf(appearance_count=0, appearance_dim=0)
