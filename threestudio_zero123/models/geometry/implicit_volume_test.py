from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import threestudio
from threestudio.models.geometry.base import (
    BaseGeometry,
    BaseImplicitGeometry,
    contract_to_unisphere,
)
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *
from threestudio.utils.misc import broadcast, get_rank


@threestudio.register("implicit-volume-test")
class ImplicitVolumeTest(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_input_dims: int = 3
        n_feature_dims: int = 3
        density_activation: Optional[str] = "softplus"
        density_bias: Union[float, str] = "blob_magic3d"
        density_blob_scale: float = 10.0
        density_blob_std: float = 0.5
        udf_blob_scale: float = 50.

        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        # pos_encoding_coarse_config: dict = field(
        #     default_factory=lambda: {
        #         "otype": "HashGrid",
        #         "n_levels": 8,
        #         "n_features_per_level": 2,
        #         "log2_hashmap_size": 19,
        #         "base_resolution": 16,
        #         "per_level_scale": 1.447269237440378,
        #     }
        # )
        # pos_encoding_fine_config: dict = field(
        #     default_factory=lambda: {
        #         "otype": "HashGrid",
        #         "n_levels": 8,
        #         "n_features_per_level": 2,
        #         "log2_hashmap_size": 19,
        #         "base_resolution": 308,
        #         "per_level_scale": 1.447269237440378,
        #     }
        # )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        normal_type: Optional[
            str
        ] = "finite_difference"  # in ['pred', 'finite_difference', 'finite_difference_laplacian']
        finite_difference_normal_eps: float = 0.01

        # automatically determine the threshold
        isosurface_threshold: Union[float, str] = 25.0

        fine_stage: Optional[bool] = False

        udf_guidance: Optional[bool] = False
        udf_ckpt_path: Optional[str] = ''
        udf_scale: float = 1.0

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.encoding = get_encoding(
            self.cfg.n_input_dims, self.cfg.pos_encoding_config
        )
        # self.encoding_coarse = get_encoding(
        #     self.cfg.n_input_dims, self.cfg.pos_encoding_coarse_config
        # )
        # self.encoding_fine = get_encoding(
        #     self.cfg.n_input_dims, self.cfg.pos_encoding_fine_config
        # )
        self.density_network = get_mlp(
            self.encoding.n_output_dims, 1, self.cfg.mlp_network_config
        )
        # self.density_network = get_mlp(
        #     self.encoding_coarse.n_output_dims+self.encoding_fine.n_output_dims, 1, self.cfg.mlp_network_config
        # )
        if self.cfg.n_feature_dims > 0:
            self.feature_network = get_mlp(
                self.encoding.n_output_dims,
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
            )
        # if self.cfg.n_feature_dims > 0:
        #     self.feature_network = get_mlp(
        #         self.encoding_coarse.n_output_dims+self.encoding_fine.n_output_dims,
        #         self.cfg.n_feature_dims,
        #         self.cfg.mlp_network_config,
        #     )
        if self.cfg.normal_type == "pred":
            self.normal_network = get_mlp(
                self.encoding.n_output_dims, 3, self.cfg.mlp_network_config
            )
        # if self.cfg.normal_type == "pred":
        #     self.normal_network = get_mlp(
        #         self.encoding_coarse.n_output_dims+self.encoding_fine.n_output_dims, 3, self.cfg.mlp_network_config
        #     )
        
        self.fine_stage = self.cfg.fine_stage

        # if self.cfg.udf_guidance:
        #     self.udf_guidance = True
        #     self.udf_ckpt_path = self.cfg.udf_ckpt_path
        #     print('udf')
        #     from neudf.models.fields import SDFNetwork, SingleVarianceNetwork
        #     self.neudf_udf_network = SDFNetwork(d_out=257,d_in=3,d_hidden=256,n_layers = 8,skip_in = [4],
        #                                         multires = 6,bias = 0.5,scale = 1.0,geometric_init = True,
        #                                         weight_norm = True).to(self.device)
        #     # self.neudf_udf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        #     # self.neudf_deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        #     checkpoint = torch.load(self.cfg.udf_ckpt_path, map_location=self.device)
        #     self.neudf_udf_network.load_state_dict(checkpoint['sdf_network_fine'])
        #     # self.neudf_deviation_network.load_state_dict(checkpoint['variance_network_fine'])

        self.udf_scale = self.cfg.udf_scale

    def get_activated_density(
        self, points: Float[Tensor, "*N Di"], density: Float[Tensor, "*N 1"]
    ) -> Tuple[Float[Tensor, "*N 1"], Float[Tensor, "*N 1"]]:
        density_bias: Union[float, Float[Tensor, "*N 1"]]
        if self.cfg.density_bias == "blob_dreamfusion":
            # pre-activation density bias
            density_bias = (
                self.cfg.density_blob_scale
                * torch.exp(
                    -0.5 * (points**2).sum(dim=-1) / self.cfg.density_blob_std**2
                )[..., None]
            )
        elif self.cfg.density_bias == "blob_magic3d":
            # pre-activation density bias
            density_bias = (
                self.cfg.density_blob_scale
                * (
                    1
                    - torch.sqrt((points**2).sum(dim=-1)) / self.cfg.density_blob_std
                )[..., None]
            )
        elif self.cfg.density_bias == "udf":
            density_bias = self.get_udf_bias(points)
        elif isinstance(self.cfg.density_bias, float):
            density_bias = self.cfg.density_bias
        else:
            raise ValueError(f"Unknown density bias {self.cfg.density_bias}")
        raw_density: Float[Tensor, "*N 1"] = density + density_bias
        density = get_activation(self.cfg.density_activation)(raw_density)
        return raw_density, density

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        grad_enabled = torch.is_grad_enabled()

        if output_normal and self.cfg.normal_type == "analytic":
            torch.set_grad_enabled(True)
            points.requires_grad_(True)

        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(
            points, self.bbox, self.unbounded
        )  # points normalized to (0, 1)

        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        density = self.density_network(enc).view(*points.shape[:-1], 1)
        raw_density, density = self.get_activated_density(points_unscaled, density)

        output = {
            "density": density,
        }

        if self.cfg.n_feature_dims > 0:
            features = self.feature_network(enc).view(
                *points.shape[:-1], self.cfg.n_feature_dims
            )
            output.update({"features": features})

        if output_normal:
            if (
                self.cfg.normal_type == "finite_difference"
                or self.cfg.normal_type == "finite_difference_laplacian"
            ):
                # TODO: use raw density
                eps = self.cfg.finite_difference_normal_eps
                if self.cfg.normal_type == "finite_difference_laplacian":
                    offsets: Float[Tensor, "6 3"] = torch.as_tensor(
                        [
                            [eps, 0.0, 0.0],
                            [-eps, 0.0, 0.0],
                            [0.0, eps, 0.0],
                            [0.0, -eps, 0.0],
                            [0.0, 0.0, eps],
                            [0.0, 0.0, -eps],
                        ]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 6 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    density_offset: Float[Tensor, "... 6 1"] = self.forward_density(
                        points_offset
                    )
                    normal = (
                        -0.5
                        * (density_offset[..., 0::2, 0] - density_offset[..., 1::2, 0])
                        / eps
                    )
                else:
                    offsets: Float[Tensor, "3 3"] = torch.as_tensor(
                        [[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 3 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    density_offset: Float[Tensor, "... 3 1"] = self.forward_density(
                        points_offset
                    )
                    normal = -(density_offset[..., 0::1, 0] - density) / eps
                normal = F.normalize(normal, dim=-1)
            elif self.cfg.normal_type == "pred":
                normal = self.normal_network(enc).view(*points.shape[:-1], 3)
                normal = F.normalize(normal, dim=-1)
            elif self.cfg.normal_type == "analytic":
                normal = -torch.autograd.grad(
                    density,
                    points_unscaled,
                    grad_outputs=torch.ones_like(density),
                    create_graph=True,
                )[0]
                normal = F.normalize(normal, dim=-1)
                if not grad_enabled:
                    normal = normal.detach()
            else:
                raise AttributeError(f"Unknown normal type {self.cfg.normal_type}")
            output.update({"normal": normal, "shading_normal": normal})

        torch.set_grad_enabled(grad_enabled)
        return output

    def forward_density(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)

        density = self.density_network(
            self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        ).reshape(*points.shape[:-1], 1)

        _, density = self.get_activated_density(points_unscaled, density)
        return density

    def forward_field(
        self, points: Float[Tensor, "*N Di"]
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        if self.cfg.isosurface_deformable_grid:
            threestudio.warn(
                f"{self.__class__.__name__} does not support isosurface_deformable_grid. Ignoring."
            )
        density = self.forward_density(points)
        return density, None

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        return -(field - threshold)

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        out.update(
            {
                "features": features,
            }
        )
        return out

    @staticmethod
    @torch.no_grad()
    def create_from(
        other: BaseGeometry,
        cfg: Optional[Union[dict, DictConfig]] = None,
        copy_net: bool = True,
        **kwargs,
    ) -> "ImplicitVolumeTest":
        if isinstance(other, ImplicitVolumeTest):
            instance = ImplicitVolumeTest(cfg, **kwargs)
            instance.encoding.load_state_dict(other.encoding.state_dict())
            # instance.encoding_coarse.load_state_dict(other.encoding_coarse.state_dict())
            # instance.encoding_fine.load_state_dict(other.encoding_fine.state_dict())
            instance.density_network.load_state_dict(other.density_network.state_dict())
            if copy_net:
                if (
                    instance.cfg.n_feature_dims > 0
                    and other.cfg.n_feature_dims == instance.cfg.n_feature_dims
                ):
                    instance.feature_network.load_state_dict(
                        other.feature_network.state_dict()
                    )
                if (
                    instance.cfg.normal_type == "pred"
                    and other.cfg.normal_type == "pred"
                ):
                    instance.normal_network.load_state_dict(
                        other.normal_network.state_dict()
                    )
            return instance
        else:
            raise TypeError(
                f"Cannot create {ImplicitVolumeTest.__name__} from {other.__class__.__name__}"
            )

    # def encoding(self, points):
    #     # feature_coarse = self.encoding_coarse(points)
    #     # feature_fine = self.encoding_fine(points)
    #     # if self.fine_stage:
    #     #     feature_coarse = feature_coarse.detach()
    #     # feature = torch.cat([feature_coarse, feature_fine], dim=-1)
    #     feature = self.encoding(points)
    #     return feature
    
    def initialize_shape(self) -> None:
        if not self.cfg.udf_guidance:
            self.neudf_udf_network = None
            return

        from neudf.models.fields import SDFNetwork, RenderingNetwork
        self.neudf_udf_network = SDFNetwork(d_out=257,d_in=3,d_hidden=256,n_layers = 8,skip_in = [4],
                                            multires = 6,bias = 0.5,scale = 1.0,geometric_init = True,
                                            weight_norm = True).to(self.device)
        # self.neudf_color_network = RenderingNetwork(d_feature=256,mode='normal_appr',d_in=9,d_out=3,d_hidden=256,
        #                                       n_layers=4,weight_norm=True,multires_view=4,squeeze_out=True).to(self.device)
        checkpoint = torch.load(self.cfg.udf_ckpt_path, map_location=self.device)
        self.neudf_udf_network.load_state_dict(checkpoint['sdf_network_fine'])
        # self.neudf_color_network.load_state_dict(checkpoint['color_network_fine'])

        # print(self.neudf_udf_network(torch.tensor([[0.,0.,0.]], dtype=torch.float32).to(self.device))[:, :1])
        # exit()

        # get_gt_density: Callable[[Float[Tensor, "N 3"]], Float[Tensor, "N 1"]]
        
        # def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
        #     udf = self.neudf_udf_network.sdf(points_rand).detach()
        #     density = self.udf2density(udf).to(points_rand)
        #     return density.reshape(-1, 1)
        # get_gt_density = func

        # def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
        #     # udf = self.neudf_udf_network(points_rand).detach()
        #     # density = self.udf2density(udf).to(points_rand)
        #     second_order_gradients = None
        #     gradients = self.neudf_udf_network.gradient(points_rand).squeeze()
        #     dirs = torch.zeros_like(points_rand)
        #     udf_nn_output = self.neudf_udf_network(points_rand)
        #     udf = udf_nn_output[:, :1]
        #     feature_vector = udf_nn_output[:, 1:]
        #     # print('udf:',points_rand[0], udf[0], self.neudf_udf_network.sdf(points_rand[:1]))
        #     color = self.neudf_color_network(points_rand, gradients, dirs, feature_vector, 
        #                                      udf, second_order_gradients).detach()
        #     density = self.udf2density(udf).to(points_rand)
        #     return density.reshape(-1, 1), color.reshape(-1, 3)
        # get_gt_density_color = func

        # # print(self.neudf_udf_network(torch.tensor([[0.,0.,0.],[0.,0.,0.]], dtype=torch.float32).to(self.device))[:, :1])
        # # print(func(torch.tensor([[0.,0.,0.],[0.,0.,0.]], dtype=torch.float32).to(self.device)))
        # # print(get_gt_density_color(torch.tensor([[0.,0.,0.],[0.,0.,0.]], dtype=torch.float32).to(self.device)))
        # # exit()

        # # Initialize SDF to a given shape when no weights are provided or force_shape_init is True
        # params = []
        # params += list(self.encoding.parameters())
        # params += list(self.feature_network.parameters())
        # params_mlp = list(self.density_network.parameters())
        # optim = torch.optim.Adam([{'params': params_mlp, 'lr': 1e-3}, {'params': params}], lr=1e-1)
        # # optim = torch.optim.Adam(self.parameters(), lr=1e-1)
        # from tqdm import tqdm

        # for _ in tqdm(
        #     range(1000),
        #     desc=f"Initializing UDF:",
        #     disable=get_rank() != 0,
        # ):
        #     points_rand = (
        #         torch.rand((10000, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0
        #     )
        #     # density_gt = get_gt_density(points_rand)
        #     density_gt, color_gt = get_gt_density_color(points_rand)

        #     # density_pred = self.forward_density(points_rand)

        #     # points_rand = points_rand[..., [1,0,2]]
        #     # points_rand[..., 0] = -points_rand[..., 0]

        #     points_unscaled = points_rand
        #     points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)

        #     # feature_coarse = self.encoding_coarse(points.reshape(-1, self.cfg.n_input_dims))
        #     # feature_fine = self.encoding_fine(points.reshape(-1, self.cfg.n_input_dims))
        #     # enc = torch.cat([feature_coarse, feature_fine], dim=-1)
        #     enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))

        #     density = self.density_network(
        #         enc
        #     ).reshape(*points.shape[:-1], 1)

        #     _, density_pred = self.get_activated_density(points_unscaled, density)

        #     color_pred = self.feature_network(enc).reshape(*points.shape[:-1], 3)

        #     # print(density_gt.max(), density_gt.min())
        #     # print(density_pred.max(), density_pred.min())
        #     loss_density = F.mse_loss(density_pred, density_gt.detach())
        #     loss_color = F.mse_loss(color_pred, color_gt.detach())
        #     print(loss_density.item(), loss_color.item(), density_gt.max().item(), density_gt.min().item(), density_pred.max().item(), density_pred.min().item())
        #     loss = loss_density * 10 + loss_color * 0
        #     optim.zero_grad()
        #     loss.backward()
        #     optim.step()

        # # explicit broadcast to ensure param consistency across ranks
        # for param in self.parameters():
        #     broadcast(param, src=0)

        # del self.neudf_udf_network
        # del self.neudf_color_network

    def udf2density(self, udf, k=500., scale=1.75, eps=0.005):
        udf = (udf-eps).clip(0., 1e5)
        density = scale * k*torch.exp(-udf*k)/(1+torch.exp(-udf*k))**2
        return density

    def get_udf_bias(self, points):
        if self.neudf_udf_network is not None:
            max_idx = 0
            # print(points.shape,min(points.shape[0], max_idx+10000))
            udf = []
            while max_idx < points.shape[0]:
                max_idx_new = min(points.shape[0], max_idx+100000)
                udf.append(self.neudf_udf_network.sdf(points[max_idx:max_idx_new]*self.udf_scale).detach())
                max_idx = max_idx_new
            udf = torch.cat(udf, dim=0)
            # udf = self.neudf_udf_network.sdf(points*self.udf_scale).detach()
            bias_raw = self.udf2density(udf, k=2000, scale=1.5, eps=0.005)
            # bias_mask = torch.sqrt((points**2).sum(dim=-1))<.5
            # bias = torch.zeros_like(bias_raw)
            # bias[bias_mask] = bias_raw[bias_mask]
            bias = torch.where(torch.sqrt((points**2).sum(dim=-1)).reshape(-1, 1)<0.5, bias_raw, 0.0)
            bias = bias + self.cfg.density_blob_scale*(1-torch.sqrt((points**2).sum(dim=-1))/self.cfg.density_blob_std).reshape(-1, 1)-(1-torch.exp(-udf*1.))*self.cfg.udf_blob_scale
        else:
            bias = self.cfg.density_blob_scale*(1-torch.sqrt((points**2).sum(dim=-1))/self.cfg.density_blob_std).reshape(-1, 1)
        return bias.detach()



