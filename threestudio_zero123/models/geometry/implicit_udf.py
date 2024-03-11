import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import BaseImplicitGeometry, contract_to_unisphere
from threestudio.models.mesh import Mesh
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.misc import broadcast, get_rank
from threestudio.utils.typing import *
from threestudio.utils.ops import get_activation


@threestudio.register("implicit-udf")
class ImplicitUDF(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_input_dims: int = 3
        n_feature_dims: int = 3
        # pos_encoding_config: dict = field(
        #     default_factory=lambda: {
        #         "otype": "HashGrid",
        #         "n_levels": 16,
        #         "n_features_per_level": 2,
        #         "log2_hashmap_size": 19,
        #         "base_resolution": 16,
        #         "per_level_scale": 1.447269237440378,
        #     }
        # )
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "Frequency",
                "n_frequencies": 10,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "softplus",
                "output_activation": "none",
                "n_neurons": 256,
                "n_hidden_layers": 10,
            }
        )
        normal_type: Optional[
            str
        ] = "pred"  # in ['pred', 'finite_difference', 'finite_difference_laplacian']
        finite_difference_normal_eps: Union[
            float, str
        ] = 0.005  # in [float, "progressive"]
        shape_init: Optional[str] = None
        shape_init_params: Optional[Any] = None
        shape_init_mesh_up: str = "+z"
        shape_init_mesh_front: str = "+x"
        force_shape_init: bool = False
        udf_bias: Union[float, str] = 0.0
        udf_blob_scale: Optional[float] = None
        udf_blob_std: Optional[float] = None

        isosurface_remove_outliers: bool = False

        udf_activation: Optional[str] = "softplus"

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.encoding = get_encoding(
            self.cfg.n_input_dims, self.cfg.pos_encoding_config
        )
        self.udf_network = get_mlp(
            self.encoding.n_output_dims, 1, self.cfg.mlp_network_config
        )

        if self.cfg.n_feature_dims > 0:
            self.feature_network = get_mlp(
                self.encoding.n_output_dims,
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
            )

        if self.cfg.normal_type == "pred":
            self.normal_network = get_mlp(
                self.encoding.n_output_dims, 3, self.cfg.mlp_network_config
            )
        if self.cfg.isosurface_deformable_grid:
            assert (
                self.cfg.isosurface_method == "mt"
            ), "isosurface_deformable_grid only works with mt"
            self.deformation_network = get_mlp(
                self.encoding.n_output_dims, 3, self.cfg.mlp_network_config
            )

        self.finite_difference_normal_eps: Optional[float] = None

    def initialize_shape(self) -> None:
        if self.cfg.shape_init is None and not self.cfg.force_shape_init:
            return

        # do not initialize shape if weights are provided
        if self.cfg.weights is not None and not self.cfg.force_shape_init:
            return

        if self.cfg.udf_bias != 0.0:
            threestudio.warn(
                "shape_init and udf_bias are both specified, which may lead to unexpected results."
            )

        get_gt_udf: Callable[[Float[Tensor, "N 3"]], Float[Tensor, "N 1"]]
        assert isinstance(self.cfg.shape_init, str)
        if self.cfg.shape_init == "sphere":
            assert isinstance(self.cfg.shape_init_params, float)
            radius = self.cfg.shape_init_params

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return torch.abs((points_rand**2).sum(dim=-1, keepdim=True).sqrt() - radius)

            get_gt_udf = func
        elif self.cfg.shape_init.startswith("mesh:"):
            assert isinstance(self.cfg.shape_init_params, float)
            mesh_path = self.cfg.shape_init[5:]
            if not os.path.exists(mesh_path):
                raise ValueError(f"Mesh file {mesh_path} does not exist.")

            import trimesh

            mesh = trimesh.load(mesh_path)

            # move to center
            centroid = mesh.vertices.mean(0)
            mesh.vertices = mesh.vertices - centroid

            # align to up-z and front-x
            dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
            dir2vec = {
                "+x": np.array([1, 0, 0]),
                "+y": np.array([0, 1, 0]),
                "+z": np.array([0, 0, 1]),
                "-x": np.array([-1, 0, 0]),
                "-y": np.array([0, -1, 0]),
                "-z": np.array([0, 0, -1]),
            }
            if (
                self.cfg.shape_init_mesh_up not in dirs
                or self.cfg.shape_init_mesh_front not in dirs
            ):
                raise ValueError(
                    f"shape_init_mesh_up and shape_init_mesh_front must be one of {dirs}."
                )
            if self.cfg.shape_init_mesh_up[1] == self.cfg.shape_init_mesh_front[1]:
                raise ValueError(
                    "shape_init_mesh_up and shape_init_mesh_front must be orthogonal."
                )
            z_, x_ = (
                dir2vec[self.cfg.shape_init_mesh_up],
                dir2vec[self.cfg.shape_init_mesh_front],
            )
            y_ = np.cross(z_, x_)
            std2mesh = np.stack([x_, y_, z_], axis=0).T
            mesh2std = np.linalg.inv(std2mesh)

            # scaling
            scale = np.abs(mesh.vertices).max()
            mesh.vertices = mesh.vertices / scale * self.cfg.shape_init_params
            mesh.vertices = np.dot(mesh2std, mesh.vertices.T).T

            from pysdf import SDF

            sdf = SDF(mesh.vertices, mesh.faces)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                # add a negative signed here
                # as in pysdf the inside of the shape has positive signed distance
                return torch.from_numpy(np.abs(-sdf(points_rand.cpu().numpy()))).to(
                    points_rand
                )[..., None]

            get_gt_udf = func

        else:
            raise ValueError(
                f"Unknown shape initialization type: {self.cfg.shape_init}"
            )

        # Initialize UDF to a given shape when no weights are provided or force_shape_init is True
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        from tqdm import tqdm

        for _ in tqdm(
            range(1000),
            desc=f"Initializing UDF to a(n) {self.cfg.shape_init}:",
            disable=get_rank() != 0,
        ):
            points_rand = (
                torch.rand((10000, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0
            )
            udf_gt = get_gt_udf(points_rand)
            udf_pred = self.forward_udf(points_rand)
            loss = F.mse_loss(udf_pred, udf_gt)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # explicit broadcast to ensure param consistency across ranks
        for param in self.parameters():
            broadcast(param, src=0)

    def get_shifted_udf(
        self, points: Float[Tensor, "*N Di"], udf: Float[Tensor, "*N 1"]
    ) -> Float[Tensor, "*N 1"]:
        udf_bias: Union[float, Float[Tensor, "*N 1"]]
        if isinstance(self.cfg.udf_bias, float):
            udf_bias = self.cfg.udf_bias
            udf = udf + udf_bias
        elif self.cfg.udf_bias == 'blob':
            udf_bias = (
                self.cfg.udf_blob_scale
                * (1 - torch.sqrt((points**2).sum(dim=-1)) / self.cfg.udf_blob_std)[..., None]
            )
            udf = get_activation(self.cfg.udf_activation)(udf_bias+udf)
        else:
            raise ValueError(f"Unknown udf bias {self.cfg.udf_bias}")
        return udf

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
        udf = self.udf_network(enc).view(*points.shape[:-1], 1)
        udf = self.get_shifted_udf(points_unscaled, udf)
        output = {"udf": udf}

        output.update({'density': self.udf2density(udf)})

        if self.cfg.n_feature_dims > 0:
            features = self.feature_network(enc).view(
                *points.shape[:-1], self.cfg.n_feature_dims
            )
            output.update({"features": features})

        if output_normal:
            if (
                self.cfg.normal_type == "finite_difference_udf"
            ):
                assert self.finite_difference_normal_eps is not None
                eps: float = self.finite_difference_normal_eps * 2
                # if self.cfg.normal_type == "finite_difference_udf":
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
                udf_offset: Float[Tensor, "... 6 1"] = self.forward_udf(
                    points_offset
                )
                udf_grad0 = (udf_offset[..., 0::2, 0] - udf) / eps
                udf_grad1 = (udf - udf_offset[..., 1::2, 0]) / eps
                udf_grad = torch.where(torch.abs(udf_grad0) > torch.abs(udf_grad1), udf_grad0, udf_grad1)
                normal = F.normalize(udf_grad, dim=-1)
            elif self.cfg.normal_type == "pred":
                normal = self.normal_network(enc).view(*points.shape[:-1], 3)
                normal = F.normalize(normal, dim=-1)
                udf_grad = normal
            elif self.cfg.normal_type == "analytic":
                udf_grad = -torch.autograd.grad(
                    udf,
                    points_unscaled,
                    grad_outputs=torch.ones_like(udf),
                    create_graph=True,
                )[0]
                normal = F.normalize(udf_grad, dim=-1)
                if not grad_enabled:
                    udf_grad = udf_grad.detach()
                    normal = normal.detach()
            else:
                raise AttributeError(f"Unknown normal type {self.cfg.normal_type}")
            output.update(
                {"normal": normal, "shading_normal": normal, "udf_grad": udf_grad}
            )
        return output

    def forward_udf(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox.to(points_unscaled), self.unbounded)

        udf = self.udf_network(
            self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        ).reshape(*points.shape[:-1], 1)
        udf = self.get_shifted_udf(points_unscaled, udf)
        return udf

    def forward_field(
        self, points: Float[Tensor, "*N Di"]
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        udf = self.udf_network(enc).reshape(*points.shape[:-1], 1)
        udf = self.get_shifted_udf(points_unscaled, udf)
        deformation: Optional[Float[Tensor, "*N 3"]] = None
        if self.cfg.isosurface_deformable_grid:
            deformation = self.deformation_network(enc).reshape(*points.shape[:-1], 3)
        return udf, deformation

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        return field - threshold

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

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        if (
            self.cfg.normal_type == "finite_difference"
            or self.cfg.normal_type == "finite_difference_laplacian"
            or self.cfg.normal_type == "finite_difference_udf"
        ):
            if isinstance(self.cfg.finite_difference_normal_eps, float):
                self.finite_difference_normal_eps = (
                    self.cfg.finite_difference_normal_eps
                )
            elif self.cfg.finite_difference_normal_eps == "progressive":
                # progressive finite difference eps from Neuralangelo
                # https://arxiv.org/abs/2306.03092
                hg_conf: Any = self.cfg.pos_encoding_config
                assert (
                    hg_conf.otype == "ProgressiveBandHashGrid"
                ), "finite_difference_normal_eps=progressive only works with ProgressiveBandHashGrid"
                current_level = min(
                    hg_conf.start_level
                    + max(global_step - hg_conf.start_step, 0) // hg_conf.update_steps,
                    hg_conf.n_levels,
                )
                grid_res = hg_conf.base_resolution * hg_conf.per_level_scale ** (
                    current_level - 1
                )
                grid_size = 2 * self.cfg.radius / grid_res
                if grid_size != self.finite_difference_normal_eps:
                    threestudio.info(
                        f"Update finite_difference_normal_eps to {grid_size}"
                    )
                self.finite_difference_normal_eps = grid_size
            else:
                raise ValueError(
                    f"Unknown finite_difference_normal_eps={self.cfg.finite_difference_normal_eps}"
                )

    def forward_density(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox.to(points_unscaled), self.unbounded)

        udf = self.udf_network(
            self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        ).reshape(*points.shape[:-1], 1)
        udf = self.get_shifted_udf(points_unscaled, udf)
        density = self.udf2density(udf)
        return density
    
    def udf2density(self, udf):
        inv_s = 64 
        sigma = 1/(1+torch.exp(-udf * inv_s))
        rho = sigma * (1 - sigma) * inv_s
        return rho