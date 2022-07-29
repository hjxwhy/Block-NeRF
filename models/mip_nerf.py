import torch
# from einops import repeat, rearrange
from mip import sample_along_rays, integrated_pos_enc, pos_enc, volumetric_rendering, resample_along_rays
# from collections import namedtuple


def _xavier_init(linear):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)

class Visibility(torch.nn.Module):
    def __init__(self, net_depth:int=4, 
                       net_width:int=128,
                       max_deg_point: int = 16,
                       deg_view: int = 4,
                       append_identity: bool = True,) -> None:
        super().__init__()
        mlp_xyz_dim = max_deg_point * 3 * 2
        mlp_view_dim = deg_view * 3 * 2
        mlp_view_dim = mlp_view_dim + 3 if append_identity else mlp_view_dim

        layers = []
        for i in range(net_depth):
            if i == 0:
                dim_in = mlp_xyz_dim + mlp_view_dim
                dim_out = net_width
            else:
                dim_in = net_width
                dim_out = net_width
            linear = torch.nn.Linear(dim_in, dim_out)
            _xavier_init(linear)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
        linear = torch.nn.Linear(net_width, 1)
        _xavier_init(linear)

        layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
        self.visibility = torch.nn.Sequential(*layers)

    def forward(self, xyz, view_direction):
        return self.visibility(torch.cat([xyz, view_direction], -1))



class MLP(torch.nn.Module):
    """
    A simple MLP.
    """

    def __init__(self, net_depth: int, net_width: int, net_depth_condition: int, net_width_condition: int,
                 skip_index: int, num_rgb_channels: int, xyz_dim: int, view_dim: int, appearance_dim:int ,appearance_count:int):
        """
          net_depth: The depth of the first part of MLP.
          net_width: The width of the first part of MLP.
          net_depth_condition: The depth of the second part of MLP.
          net_width_condition: The width of the second part of MLP.
          skip_index: Add a skip connection to the output of every N layers.
          num_rgb_channels: The number of RGB channels.
          appearance_dim: int, dimention of appearance
          appearance_count: int, numbers of image in a block
        """
        super(MLP, self).__init__()
        self.skip_index: int = skip_index  # Add a skip connection to the output of every N layers.
        layers = []
        for i in range(net_depth):
            if i == 0:
                dim_in = xyz_dim
                dim_out = net_width
            elif (i - 1) % skip_index == 0 and i > 1:
                dim_in = net_width + xyz_dim
                dim_out = net_width
            else:
                dim_in = net_width
                dim_out = net_width
            linear = torch.nn.Linear(dim_in, dim_out)
            _xavier_init(linear)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
        self.layers = torch.nn.ModuleList(layers)
        del layers
        self.density_layer = torch.nn.Linear(net_width, 1)
        _xavier_init(self.density_layer)
        self.extra_layer = torch.nn.Linear(net_width, net_width)  # extra_layer is not the same as NeRF
        _xavier_init(self.extra_layer)
        layers = []
        for i in range(net_depth_condition):
            if i == 0:
                dim_in = net_width + view_dim
                dim_out = net_width_condition
            else:
                dim_in = net_width_condition
                dim_out = net_width_condition
            linear = torch.nn.Linear(dim_in, dim_out)
            _xavier_init(linear)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
        self.view_layers = torch.nn.Sequential(*layers)
        del layers
        self.color_layer = torch.nn.Linear(net_width_condition, num_rgb_channels)

        if appearance_dim > 0:
            self.embedding_a = torch.nn.Embedding(appearance_count, appearance_dim)
        else:
            self.embedding_a = None

    def forward(self, x, view_direction, appearance_id=None, exposure=None):
        """Evaluate the MLP.

        Args:
            x: torch.Tensor(float32), [batch, feature], points.
            view_direction: torch.Tensor(float32), [batch, feature], if not None, this
            variable will be part of the input to the second part of the MLP
            concatenated with the output vector of the first part of the MLP. If
            None, only the first part of the MLP will be used with input x. In the
            original paper, this variable is the view direction.

        Returns:
            raw_rgb: torch.Tensor(float32), with a shape of
                [batch, num_samples, num_rgb_channels].
            raw_density: torch.Tensor(float32), with a shape of
                [batch, num_samples, num_density_channels].
        """
        # num_samples = x.shape[1]
        inputs = x  # [B, 2*3*L]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i % self.skip_index == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)
        raw_density = self.density_layer(x)
        if view_direction is not None:
            # Output of the first part of MLP.
            bottleneck = self.extra_layer(x)
            fc_input = [bottleneck, view_direction]
            if exposure is not None:
                fc_input.append(exposure)
            if self.embedding_a is not None:
                fc_input.append(self.embedding_a(appearance_id.long()))
            x = torch.cat(fc_input, dim=-1)
            # Here use 1 extra layer to align with the original nerf model.
            x = self.view_layers(x)
        raw_rgb = self.color_layer(x)
        return raw_rgb, raw_density


class MipNerf(torch.nn.Module):
    """Nerf NN Model with both coarse and fine MLPs."""

    def __init__(self,near, far,
                 num_samples: int = 256,
                 num_levels: int = 2,
                 resample_padding: float = 0.01,
                 disparity: bool = False,
                 max_deg_point: int = 16,
                 deg_view: int = 4,
                 density_noise: float = 0.,
                 density_bias: float = -1.,
                 rgb_padding: float = 0.001,
                 disable_integration: bool = False,
                 append_identity: bool = True,
                 mlp_net_depth: int = 8,
                 mlp_net_width: int = 512,
                 mlp_net_depth_condition: int = 3,
                 mlp_net_width_condition: int = 128,
                 mlp_skip_index: int = 4,
                 mlp_num_rgb_channels: int = 3,
                 deg_exposure:int=4,
                 appearance_dim:int=32,
                 appearance_count:int=0):
        super(MipNerf, self).__init__()
        self.near = near
        self.far = far
        self.num_levels = num_levels  # The number of sampling levels.
        self.num_samples = num_samples  # The number of samples per level.
        self.disparity = disparity  # If True, sample linearly in disparity, not in depth.
        self.disable_integration = disable_integration  # If True, use PE instead of IPE.
        self.max_deg_point = max_deg_point  # Max degree of positional encoding for 3D points.
        self.deg_view = deg_view  # Degree of positional encoding for viewdirs.
        self.density_noise = density_noise  # Standard deviation of noise added to raw density.
        self.density_bias = density_bias  # The shift added to raw densities pre-activation.
        self.resample_padding = resample_padding  # Dirichlet/alpha "padding" on the histogram.
        mlp_xyz_dim = max_deg_point * 3 * 2
        mlp_view_dim = deg_view * 3 * 2
        mlp_view_dim = mlp_view_dim + 3 if append_identity else mlp_view_dim

        self.use_exposure = deg_exposure>0
        self.use_appearance = appearance_dim > 0
        self.deg_exposure = deg_exposure
        mlp_view_dim = mlp_view_dim + 2*deg_exposure if deg_exposure>0 else mlp_view_dim
        mlp_view_dim = mlp_view_dim + appearance_dim if appearance_dim>0 else mlp_view_dim
        assert (appearance_count>0 and appearance_dim>0) or (appearance_count==0 and appearance_dim==0),\
         "please set correct appearance_count and dim"

        self.mlp = MLP(mlp_net_depth, mlp_net_width, mlp_net_depth_condition, mlp_net_width_condition,
                       mlp_skip_index, mlp_num_rgb_channels,mlp_xyz_dim, mlp_view_dim, appearance_dim,appearance_count)
        self.rgb_activation = torch.nn.Sigmoid()
        self.rgb_padding = rgb_padding
        self.density_activation = torch.nn.Softplus()

        self.vis = Visibility()

    def forward(self, rays:torch.tensor, randomized: bool, chunk_size = 1024):
        """The mip-NeRF Model.
        Args:
            rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
            randomized: bool, use randomized stratified sampling.
            chunk_size: int, chunk size for render
        Returns:
            ret: list, [*(rgb, distance, acc)]
        """

        ret = {}
        t_samples, weights = None, None
        stage = None
        for i_level in range(self.num_levels):
            if i_level == 0:
                # Stratified sampling along rays
                stage = 'coarse'
                t_samples, means_covs = sample_along_rays(
                    rays[:, :3], #.origins
                    rays[:,3:6], #.directions,
                    rays[:, 6:7], #.radii,
                    self.num_samples,
                    self.near,
                    self.far,
                    randomized,
                    self.disparity,
                )
            else:
                t_samples, means_covs = resample_along_rays(
                    rays[:, :3], #.origins
                    rays[:,3:6], #.directions,
                    rays[:, 6:7], #.radii,
                    t_samples,
                    weights,
                    randomized,
                    resample_padding=self.resample_padding,
                )
            if self.disable_integration:
                means_covs = (means_covs[0], torch.zeros_like(means_covs[1]))
            samples_enc = integrated_pos_enc(
                means_covs,
                0,
                self.max_deg_point,
            )  # samples_enc: [B, N, 2*3*L]  L:(max_deg_point - min_deg_point)

            # Point attribute predictions
            viewdirs = rays[:, 3:6] / torch.norm(rays[:, 3:6], dim=-1, keepdim=True)
            viewdirs_enc = pos_enc(
                viewdirs,
                min_deg=0,
                max_deg=self.deg_view,
                append_identity=True,
            )
            if self.use_exposure:
                exposure_enc = pos_enc(rays[:,7:8], 
                                        min_deg=0, 
                                        max_deg=self.deg_exposure, 
                                        append_identity=False)

            Batch, N_samples, _ = samples_enc.shape
            images_id = rays[:, -1:]
            samples_enc = samples_enc.view(-1, samples_enc.shape[-1])
            viewdirs_enc = viewdirs_enc.repeat(1, N_samples, 1).view(-1, viewdirs_enc.shape[-1])
            exposure_enc = exposure_enc.repeat(1, N_samples, 1).view(-1, exposure_enc.shape[-1]) if self.use_exposure else None
            images_id = images_id.repeat(1, N_samples).view(-1) if self.use_appearance else None
            raw_rgbs = []
            raw_densitys = []
            for i in range(0, samples_enc.shape[0], chunk_size):
                raw_rgb, raw_density = self.mlp(
                    samples_enc[i:i+chunk_size],
                    viewdirs_enc[i:i+chunk_size],
                    images_id[i:i+chunk_size] if self.use_appearance else None,
                    exposure_enc[i:i+chunk_size] if self.use_exposure else None,
                )
                raw_rgbs.append(raw_rgb)
                raw_densitys.append(raw_density)
            raw_rgbs = torch.cat(raw_rgbs, 0).reshape([Batch, N_samples, 3])
            raw_densitys = torch.cat(raw_densitys, 0).reshape([Batch, N_samples, 1])
            
            # Add noise to regularize the density predictions if needed.
            if randomized and (self.density_noise > 0):
                raw_densitys += self.density_noise * torch.randn(raw_densitys.shape, dtype=raw_densitys.dtype)

            # Volumetric rendering.
            rgb = self.rgb_activation(raw_rgbs)  # [B, N, 3]
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            density = self.density_activation(raw_densitys + self.density_bias)  # [B, N, 1]
            comp_rgb, depth, weights = volumetric_rendering(
                rgb,
                density,
                t_samples,
                rays[:, 3:6], #.directions,
                white_bkgd=False,
            )
            ret[f'rgb_{stage}'] = comp_rgb
            ret[f'depth_{stage}'] = depth
            ret[f'weights_{stage}'] = weights

        return ret