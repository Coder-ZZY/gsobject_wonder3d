import bisect
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *

from threestudio.data.image import (
    SingleImageDataModuleConfig,
    SingleImageDataBase,
    SingleImageIterableDataset,
    SingleImageDataset,
)


import os
import json
from PIL import Image

import torchvision.transforms.functional as TF

from glob import glob
import PIL.Image

def get_ortho_ray_directions_origins(W, H, use_pixel_centers=True):
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    i, j = torch.from_numpy(i), torch.from_numpy(j)

    origins = torch.stack([(i/W-0.5)*2, (j/H-0.5)*2, torch.zeros_like(i)], dim=-1) # W, H, 3
    directions = torch.stack([torch.zeros_like(i), torch.zeros_like(j), torch.ones_like(i)], dim=-1) # W, H, 3

    return origins, directions

def get_ortho_rays(origins, directions, c2w, keepdim=False):
    # Rotate ray directions from camera coordinate to the world coordinate
    # rays_d = directions @ c2w[:, :3].T # (H, W, 3) # slow?
    assert directions.shape[-1] == 3
    assert origins.shape[-1] == 3

    if directions.ndim == 2: # (N_rays, 3)
        assert c2w.ndim == 3 # (N_rays, 4, 4) / (1, 4, 4)
        rays_d = torch.matmul(c2w[:, :3, :3], directions[:, :, None]).squeeze()  # (N_rays, 3)
        rays_o = torch.matmul(c2w[:, :3, :3], origins[:, :, None]).squeeze()  # (N_rays, 3)
        rays_o = c2w[:,:3,3].expand(rays_d.shape) + rays_o  
    elif directions.ndim == 3: # (H, W, 3)
        if c2w.ndim == 2: # (4, 4)
            rays_d = torch.matmul(c2w[None, None, :3, :3], directions[:, :, :, None]).squeeze()  # (H, W, 3)
            rays_o = torch.matmul(c2w[None, None, :3, :3], origins[:, :, :, None]).squeeze()  # (H, W, 3)
            rays_o = c2w[None, None,:3,3].expand(rays_d.shape) + rays_o  
        elif c2w.ndim == 3: # (B, 4, 4)
            rays_d = torch.matmul(c2w[:,None, None, :3, :3], directions[None, :, :, :, None]).squeeze()  # # (B, H, W, 3)
            rays_o = torch.matmul(c2w[:,None, None, :3, :3], origins[None, :, :, :, None]).squeeze()  # # (B, H, W, 3)
            rays_o = c2w[:,None, None, :3,3].expand(rays_d.shape) + rays_o  

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d

def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True):
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    i, j = torch.from_numpy(i), torch.from_numpy(j)

    # directions = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1) # (H, W, 3)
    # opencv system
    directions = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1) # (H, W, 3)

    return directions

def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def camNormal2worldNormal(rot_c2w, camNormal):
    H,W,_ = camNormal.shape
    normal_img = np.matmul(rot_c2w[None, :, :], camNormal.reshape(-1,3)[:, :, None]).reshape([H, W, 3])

    return normal_img

def worldNormal2camNormal(rot_w2c, worldNormal):
    H,W,_ = worldNormal.shape
    normal_img = np.matmul(rot_w2c[None, :, :], worldNormal.reshape(-1,3)[:, :, None]).reshape([H, W, 3])

    return normal_img

def trans_normal(normal, RT_w2c, RT_w2c_target):

    normal_world = camNormal2worldNormal(np.linalg.inv(RT_w2c[:3,:3]), normal)
    normal_target_cam = worldNormal2camNormal(RT_w2c_target[:3,:3], normal_world)

    return normal_target_cam

def img2normal(img):
    return (img/255.)*2-1

def normal2img(normal):
    return np.uint8((normal*0.5+0.5)*255)

def norm_normalize(normal, dim=-1):

    normal = normal/(np.linalg.norm(normal, axis=dim, keepdims=True)+1e-6)

    return normal

def RT_opengl2opencv(RT):
     # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    R = RT[:3, :3]
    t = RT[:3, 3]

    R_bcam2cv = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)

    R_world2cv = R_bcam2cv @ R
    t_world2cv = R_bcam2cv @ t

    RT = np.concatenate([R_world2cv,t_world2cv[:,None]],1)

    return RT

def normal_opengl2opencv(normal):
    H,W,C = np.shape(normal)
    # normal_img = np.reshape(normal, (H*W,C))
    R_bcam2cv = np.array([1, -1, -1], np.float32)
    normal_cv = normal * R_bcam2cv[None, None, :]

    print(np.shape(normal_cv))

    return normal_cv

def inv_RT(RT):
    RT_h = np.concatenate([RT, np.array([[0,0,0,1]])], axis=0)
    RT_inv = np.linalg.inv(RT_h)

    return RT_inv[:3, :]


def load_a_prediction(root_dir, test_object, imSize, view_types, load_color=False, cam_pose_dir=None,
                         normal_system='front', erode_mask=True, camera_type='ortho', cam_params=None):

    all_images = []
    all_normals = []
    all_normals_world = []
    all_masks = []
    all_color_masks = []
    all_poses = []
    all_w2cs = []
    directions = []
    ray_origins = []

    RT_front = np.loadtxt(glob(os.path.join(cam_pose_dir, '*_%s_RT.txt'%( 'front')))[0])   # world2cam matrix
    RT_front_cv = RT_opengl2opencv(RT_front)   # convert normal from opengl to opencv
    for idx, view in enumerate(view_types):
        print(os.path.join(root_dir,test_object))
        normal_filepath = os.path.join(root_dir, test_object, 'normals_000_%s.png'%( view))
        # Load key frame
        if load_color:  # use bgr
            image =np.array(PIL.Image.open(normal_filepath.replace("normals", "rgb")).resize(imSize))[:, :, :3]

        normal = np.array(PIL.Image.open(normal_filepath).resize(imSize))
        mask = normal[:, :, 3]
        normal = normal[:, :, :3]

        color_mask = np.array(PIL.Image.open(os.path.join(root_dir,test_object, 'masked_colors/rgb_000_%s.png'%( view))).resize(imSize))[:, :, 3]
        invalid_color_mask = color_mask < 255*0.5
        threshold =  np.ones_like(image[:, :, 0]) * 250
        invalid_white_mask = (image[:, :, 0] > threshold) & (image[:, :, 1] > threshold) & (image[:, :, 2] > threshold)
        invalid_color_mask_final = invalid_color_mask & invalid_white_mask
        color_mask = (1 - invalid_color_mask_final) > 0

        # if erode_mask:
        #     kernel = np.ones((3, 3), np.uint8)
        #     mask = cv2.erode(mask, kernel, iterations=1)

        RT = np.loadtxt(os.path.join(cam_pose_dir, '000_%s_RT.txt'%( view)))  # world2cam matrix

        normal = img2normal(normal)

        normal[mask==0] = [0,0,0]
        mask = mask> (0.5*255)
        if load_color:
            all_images.append(image)
        
        all_masks.append(mask)
        all_color_masks.append(color_mask)
        RT_cv = RT_opengl2opencv(RT)   # convert normal from opengl to opencv
        all_poses.append(inv_RT(RT_cv))   # cam2world
        all_w2cs.append(RT_cv)

        # whether to 
        normal_cam_cv = normal_opengl2opencv(normal)

        if normal_system == 'front':
            print("the loaded normals are defined in the system of front view")
            normal_world = camNormal2worldNormal(inv_RT(RT_front_cv)[:3, :3], normal_cam_cv)
        elif normal_system == 'self':
            print("the loaded normals are in their independent camera systems")
            normal_world = camNormal2worldNormal(inv_RT(RT_cv)[:3, :3], normal_cam_cv)
        all_normals.append(normal_cam_cv)
        all_normals_world.append(normal_world)

        if camera_type == 'ortho':
            origins, dirs = get_ortho_ray_directions_origins(W=imSize[0], H=imSize[1])
        elif camera_type == 'pinhole':
            dirs = get_ray_directions(W=imSize[0], H=imSize[1],
                                                 fx=cam_params[0], fy=cam_params[1], cx=cam_params[2], cy=cam_params[3])
            origins = dirs # occupy a position
        else:
            raise Exception("not support camera type")
        ray_origins.append(origins)
        directions.append(dirs)
        
        
        if not load_color:
            all_images = [normal2img(x) for x in all_normals_world]


    return np.stack(all_images), np.stack(all_masks), np.stack(all_normals), \
        np.stack(all_normals_world), np.stack(all_poses), np.stack(all_w2cs), np.stack(ray_origins), np.stack(directions), np.stack(all_color_masks)


class OrthoDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.data_dir = self.config.root_dir
        self.object_name = self.config.scene
        self.scene = self.config.scene
        self.imSize = self.config.imSize
        self.load_color = True
        self.img_wh = [self.imSize[0], self.imSize[1]]
        self.w = self.img_wh[0]
        self.h = self.img_wh[1]
        self.camera_type = self.config.camera_type
        self.camera_params = self.config.camera_params  # [fx, fy, cx, cy]
        
        self.view_types = ['front', 'front_right', 'right', 'back', 'left', 'front_left']

        self.view_weights = torch.from_numpy(np.array(self.config.view_weights)).float().to(self.rank).view(-1)
        self.view_weights = self.view_weights.view(-1,1,1).repeat(1, self.h, self.w)

        if self.config.cam_pose_dir is None:
            self.cam_pose_dir = "./datasets/fixed_poses"
        else:
            self.cam_pose_dir = self.config.cam_pose_dir
            
        self.images_np, self.masks_np, self.normals_cam_np, self.normals_world_np, \
            self.pose_all_np, self.w2c_all_np, self.origins_np, self.directions_np, self.rgb_masks_np = load_a_prediction(
                self.data_dir, self.object_name, self.imSize, self.view_types,
                self.load_color, self.cam_pose_dir, normal_system='front', 
                camera_type=self.camera_type, cam_params=self.camera_params)

        self.has_mask = True
        self.apply_mask = self.config.apply_mask

        self.all_c2w = torch.from_numpy(self.pose_all_np)
        self.all_images = torch.from_numpy(self.images_np) / 255.
        self.all_fg_masks = torch.from_numpy(self.masks_np)
        self.all_rgb_masks = torch.from_numpy(self.rgb_masks_np)
        self.all_normals_world = torch.from_numpy(self.normals_world_np)
        self.origins = torch.from_numpy(self.origins_np)
        self.directions = torch.from_numpy(self.directions_np)

        self.directions = self.directions.float().to(self.rank)
        self.origins = self.origins.float().to(self.rank)
        self.all_rgb_masks = self.all_rgb_masks.float().to(self.rank)
        self.all_c2w, self.all_images, self.all_fg_masks, self.all_normals_world = \
            self.all_c2w.float().to(self.rank), \
            self.all_images.float().to(self.rank), \
            self.all_fg_masks.float().to(self.rank), \
            self.all_normals_world.float().to(self.rank)
        

class OrthoDataset(Dataset, OrthoDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class OrthoIterableDataset(IterableDataset, OrthoDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


class OrthoDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = OrthoIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = OrthoDataset(self.config, self.config.get('val_split', 'train'))
        if stage in [None, 'test']:
            self.test_dataset = OrthoDataset(self.config, self.config.get('test_split', 'test'))
        if stage in [None, 'predict']:
            self.predict_dataset = OrthoDataset(self.config, 'train')    

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       






@dataclass
class SingleImageWonder3DDataModuleConfig:
    # height and width should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 96
    width: Any = 96
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    default_elevation_deg: float = 0.0
    default_azimuth_deg: float = -180.0
    default_camera_distance: float = 1.2
    default_fovy_deg: float = 60.0
    image_path: str = ""
    use_random_camera: bool = True
    random_camera: dict = field(default_factory=dict)
    rays_noise_scale: float = 2e-3
    batch_size: int = 1
    requires_depth: bool = False
    requires_normal: bool = False

    wonder3d_dir: str = ''
    wonder3d_scene: str = ''


class SingleImageWonder3DIterableDataset(IterableDataset, SingleImageDataBase, Updateable):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)
        # self.setup_wonder3d(data_dir=cfg.wonder3d_dir, scene=cfg.wonder3d_scene)
        # self.ref_images = self.get_ref_images()




        # print('zero123', self.rays_o[0,0,127,:], self.rays_d[0,0,127,:])
        # print('wonder3d', self.get_wonder3d(torch.arange(6))['rays_o'][:,0,256-1,:], 
        #     self.get_wonder3d(torch.arange(6))['rays_d'][:,0,256-1,:])
        # exit()





    def collate(self, batch) -> Dict[str, Any]:
        batch = {
            "rays_o": self.rays_o,
            "rays_d": self.rays_d,
            "mvp_mtx": self.mvp_mtx,
            "camera_positions": self.camera_position,
            "light_positions": self.light_position,
            "elevation": self.elevation_deg,
            "azimuth": self.azimuth_deg,
            "camera_distances": self.camera_distance,
            "rgb": self.rgb,
            "ref_depth": self.depth,
            "ref_normal": self.normal,
            "mask": self.mask,
            "height": self.cfg.height,
            "width": self.cfg.width,
            # 'wonder3d': self.get_wonder3d(torch.randint(6, (2,)))
        }
        if self.cfg.use_random_camera:
            batch["random_camera"] = self.random_pose_generator.collate(None)

        return batch

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.update_step_(epoch, global_step, on_load_weights)
        self.random_pose_generator.update_step(epoch, global_step, on_load_weights)

    def __iter__(self):
        while True:
            yield {}

    def setup_wonder3d(self, data_dir, scene):
        self.wonder3d = wonder3d(data_dir, scene)

    def get_wonder3d(self, idx):
        return self.wonder3d.collate(idx)

    def get_ref_images(self):
        pass

class wonder3d():
    def __init__(self, data_dir, scene):
        # self.config = config
        # self.split = split
        self.rank = get_rank()

        self.data_dir = data_dir
        self.object_name = scene
        self.scene = scene
        self.imSize = [256, 256]
        self.load_color = True
        self.img_wh = [self.imSize[0], self.imSize[1]]
        self.w = self.img_wh[0]
        self.h = self.img_wh[1]
        self.camera_type = 'ortho'
        self.camera_params = None  # [fx, fy, cx, cy]
        
        self.view_types = ['front', 'front_right', 'right', 'back', 'left', 'front_left']

        self.view_weights = torch.from_numpy(np.array([1.0, 0.8, 0.2, 1.0, 0.4, 0.7])).float().to(self.rank).view(-1)
        self.view_weights = self.view_weights.view(-1,1,1).repeat(1, self.h, self.w)

        self.cam_pose_dir = "./load/wonder3d/fixed_poses"
            
        self.images_np, self.masks_np, self.normals_cam_np, self.normals_world_np, \
            self.pose_all_np, self.w2c_all_np, self.origins_np, self.directions_np, self.rgb_masks_np = load_a_prediction(
                self.data_dir, self.object_name, self.imSize, self.view_types,
                self.load_color, self.cam_pose_dir, normal_system='front', 
                camera_type=self.camera_type, cam_params=self.camera_params)

        self.has_mask = True
        self.apply_mask = True

        self.all_c2w = torch.from_numpy(self.pose_all_np)
        self.all_images = torch.from_numpy(self.images_np) / 255.
        self.all_fg_masks = torch.from_numpy(self.masks_np)
        self.all_rgb_masks = torch.from_numpy(self.rgb_masks_np)
        self.all_normals_world = torch.from_numpy(self.normals_world_np)
        self.origins = torch.from_numpy(self.origins_np)
        self.directions = torch.from_numpy(self.directions_np)

        # self.directions = self.directions.float().to(self.rank)
        # self.origins = self.origins.float().to(self.rank)
        # self.all_rgb_masks = self.all_rgb_masks.float().to(self.rank)
        # self.all_c2w, self.all_images, self.all_fg_masks, self.all_normals_world = \
        #     self.all_c2w.float().to(self.rank), \
        #     self.all_images.float().to(self.rank), \
        #     self.all_fg_masks.float().to(self.rank), \
        #     self.all_normals_world.float().to(self.rank)
    
    def prepare_rays_a_view(self, img_idx):
        """
        Generate random rays at world space from one camera.
        """
        img_idx = img_idx.to(self.rank)
        tx = torch.linspace(0, self.w - 1, self.w)
        ty = torch.linspace(0, self.h - 1, self.h)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)

        pixels_x = pixels_x.reshape(-1).int()
        pixels_y = pixels_y.reshape(-1).int()
        # color = self.all_images[img_idx]
        # mask = self.all_rgb_masks[img_idx]      # batch_size, 3
        # normal = self.all_normals_world[img_idx]      # batch_size, 3
        
        q = torch.stack([(pixels_x / self.w-0.5)*2, (pixels_y / self.h-0.5)*2, torch.zeros_like(pixels_y)], dim=-1).float()  # batch_size, 3
        v = torch.stack([torch.zeros_like(pixels_y), torch.zeros_like(pixels_y), torch.ones_like(pixels_y)], dim=-1).float()
        
        rays_v = v / torch.linalg.norm(v, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.all_c2w.float().to(self.rank)[img_idx.to(self.rank), None, :3, :3].to(self.rank), rays_v[:, :, None].to(self.rank)).squeeze()  # batch_size, 3
        
        rays_o = torch.matmul(self.all_c2w.float().to(self.rank)[img_idx, None, :3, :3], q[:, :, None].to(self.rank)).squeeze()  # batch_size, 3
        rays_o = self.all_c2w.float().to(self.rank)[img_idx, None, :3, 3].expand(rays_v.shape).to(self.rank) + rays_o.to(self.rank) # batch_size, 3
        
        # return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, None], normal, cosines[:, None]], dim=-1)   # batch_size, 10
        rays_o = rays_o[..., [1,0,2]]
        rays_o[..., 0] = -rays_o[..., 0]
        rays_v = rays_v[..., [1,0,2]]
        rays_v[..., 0] = -rays_v[..., 0]
        rays_o = rays_o.reshape(-1,256,256,3).permute((0,2,1,3))
        rays_v = rays_v.reshape(-1,256,256,3).permute((0,2,1,3))
        return rays_o, rays_v

    def collate(self, idx):
        rays_o, rays_d = self.prepare_rays_a_view(idx)
        rays_o = rays_o*0.5
        return {
            'rays_o': rays_o.reshape(-1,256,256,3), 
            'rays_d': rays_d.reshape(-1,256,256,3), 
            'idx': idx, 
            'images': self.all_images[idx], 
            'masks': self.all_rgb_masks[idx]
        }

class SingleImageWonder3DDataset(Dataset, SingleImageDataBase):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)
        # self.setup_wonder3d(data_dir=cfg.wonder3d_dir, scene=cfg.wonder3d_scene)

    def __len__(self):
        return len(self.random_pose_generator)

    def __getitem__(self, index):
        # return self.random_pose_generator[index]
        out = self.random_pose_generator[index]
        # out.update({'wonder3d': self.get_wonder3d(torch.arange(6))})
        return out
        # if index == 0:
        #     return {
        #         'rays_o': self.rays_o[0],
        #         'rays_d': self.rays_d[0],
        #         'mvp_mtx': self.mvp_mtx[0],
        #         'camera_positions': self.camera_position[0],
        #         'light_positions': self.light_position[0],
        #         'elevation': self.elevation_deg[0],
        #         'azimuth': self.azimuth_deg[0],
        #         'camera_distances': self.camera_distance[0],
        #         'rgb': self.rgb[0],
        #         'depth': self.depth[0],
        #         'mask': self.mask[0]
        #     }
        # else:
        #     return self.random_pose_generator[index - 1]
    
    def setup_wonder3d(self, data_dir, scene):
        self.wonder3d = wonder3d(data_dir, scene)

    def get_wonder3d(self, idx):
        return self.wonder3d.collate(idx)


@register("single-image-wonder3d-datamodule")
class SingleImageWonder3DDataModule(pl.LightningDataModule):
    cfg: SingleImageWonder3DDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(SingleImageWonder3DDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            # self.train_dataset = SingleImageIterableDataset(self.cfg, "train")
            self.train_dataset = SingleImageWonder3DIterableDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = SingleImageWonder3DDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = SingleImageWonder3DDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset, num_workers=0, batch_size=batch_size, collate_fn=collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)



