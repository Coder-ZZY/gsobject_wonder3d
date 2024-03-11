#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import sys
import os.path as osp
from typing import NamedTuple, Optional

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import torch
import torch.nn.functional as F
import math
from scene.colmap_loader import (qvec2rotmat, read_extrinsics_binary,
                                 read_extrinsics_text, read_intrinsics_binary,
                                 read_intrinsics_text, read_points3D_binary,
                                 read_points3D_text)
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, getWorld2View2, transform_pcd
from utils.image_utils import load_meshlab_file
from utils.camera_utils import transform_cams, CameraInfo, generate_ellipse_path_from_camera_infos


# from threestudio.utils.typing import *


class SingleImageDreamUDFDataModuleConfig:
    # height and width should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height= 96
    width= 96
    # resolution_milestones: List[int] = field(default_factory=lambda: [])
    default_elevation_deg: float = 0.0
    default_azimuth_deg: float = -180.0
    default_camera_distance: float = 1.2
    default_fovy_deg: float = 60.0
    image_path: str = ""
    use_random_camera: bool = True
    # random_camera: dict = field(default_factory=dict)
    rays_noise_scale: float = 2e-3
    batch_size: int = 1
    requires_depth: bool = False
    requires_normal: bool = False


# class SingleImageDreamUDFDataBase:
#     def setup(self, cfg, split):
#         self.split = split
#         self.rank = get_rank()
#         self.cfg: SingleImageDreamUDFDataModuleConfig = cfg

#         if self.cfg.use_random_camera:
#             random_camera_cfg = parse_structured(
#                 RandomCameraDataModuleConfig, self.cfg.get("random_camera", {})
#             )
#             if split == "train":
#                 self.random_pose_generator = RandomCameraIterableDataset(
#                     random_camera_cfg
#                 )
#             else:
#                 self.random_pose_generator = RandomCameraDataset(
#                     random_camera_cfg, split
#                 )

#         elevation_deg = torch.FloatTensor([self.cfg.default_elevation_deg])#0
#         azimuth_deg = torch.FloatTensor([0,45,90,180,270,315.])
#         camera_distance = torch.FloatTensor([self.cfg.default_camera_distance])#1.2

#         elevation = elevation_deg * math.pi / 180
#         azimuth = azimuth_deg * math.pi / 180
#         camera_position: Float[Tensor, "6 3"] = torch.stack(
#             [
#                 camera_distance * torch.cos(elevation) * torch.cos(azimuth),
#                 camera_distance * torch.cos(elevation) * torch.sin(azimuth),
#                 camera_distance * torch.sin(elevation) * torch.zeros_like(azimuth),
#             ],
#             dim=-1,
#         )

#         center: Float[Tensor, "6 3"] = torch.zeros_like(camera_position)
#         up: Float[Tensor, "1 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None]

#         light_position: Float[Tensor, "6 3"] = camera_position
#         lookat: Float[Tensor, "6 3"] = F.normalize(center - camera_position, dim=-1)
#         right: Float[Tensor, "6 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
#         up = F.normalize(torch.cross(right, lookat), dim=-1)
#         self.c2w: Float[Tensor, "6 3 4"] = torch.cat(
#             [torch.stack([right, up, -lookat], dim=-1), camera_position[:, :, None]],
#             dim=-1,
#         )

#         self.camera_position = camera_position
#         self.light_position = light_position
#         self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
#         self.camera_distance = camera_distance
#         self.fovy = torch.deg2rad(torch.FloatTensor([self.cfg.default_fovy_deg]))

#         self.heights: List[int] = (
#             [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
#         )
#         self.widths: List[int] = (
#             [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
#         )
#         assert len(self.heights) == len(self.widths)
#         self.resolution_milestones: List[int]
#         if len(self.heights) == 1 and len(self.widths) == 1:
#             if len(self.cfg.resolution_milestones) > 0:
#                 threestudio.warn(
#                     "Ignoring resolution_milestones since height and width are not changing"
#                 )
#             self.resolution_milestones = [-1]
#         else:
#             assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
#             self.resolution_milestones = [-1] + self.cfg.resolution_milestones

#         self.directions_unit_focals = [
#             get_ray_directions(H=height, W=width, focal=1.0)
#             for (height, width) in zip(self.heights, self.widths)
#         ]
#         self.focal_lengths = [
#             0.5 * height / torch.tan(0.5 * self.fovy) for height in self.heights
#         ]

#         self.height: int = self.heights[0]
#         self.width: int = self.widths[0]
#         self.directions_unit_focal = self.directions_unit_focals[0]
#         self.focal_length = self.focal_lengths[0]
#         self.set_rays()
#         self.load_images()
#         self.prev_height = self.height

#     def set_rays(self):
#         # get directions by dividing directions_unit_focal by focal length
#         directions: Float[Tensor, "6 H W 3"] = self.directions_unit_focal[None]
#         directions[:, :, :, :2] = directions[:, :, :, :2] / self.focal_length

#         rays_o, rays_d = get_rays(
#             directions, self.c2w, keepdim=True, noise_scale=self.cfg.rays_noise_scale
#         )

#         # proj_mtx: Float[Tensor, "4 4"] = get_projection_matrix(
#         #     self.fovy, self.width / self.height, 0.1, 100.0
#         # )  # FIXME: hard-coded near and far
#         # mvp_mtx: Float[Tensor, "4 4"] = get_mvp_matrix(self.c2w, proj_mtx)

#         self.rays_o, self.rays_d = rays_o, rays_d
#         # self.mvp_mtx = mvp_mtx

#     def load_images(self):
#         # load image
#         image_path = '/'.join(self.cfg.image_path.split('/')[:-1])
#         image_names = [
#             'rgb_000_front.png', 
#             'rgb_000_front_right.png', 
#             'rgb_000_right.png', 
#             'rgb_000_back.png', 
#             'rgb_000_left.png', 
#             'rgb_000_front_left.png', 
#             ]
#         assert os.path.exists(
#             self.cfg.image_path
#         ), f"Could not find image {self.cfg.image_path}!"

#         self.rgb = []
#         self.mask = []
#         for i in range(6):
#             rgba = cv2.cvtColor(
#                 cv2.imread(f'{image_path}/{image_names[i]}', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
#             )
#             rgba = (
#                 cv2.resize(
#                     rgba, (self.width, self.height), interpolation=cv2.INTER_AREA
#                 ).astype(np.float32)
#                 / 255.0
#             )

#             rgb = rgba[..., :3]
#             self.rgb.append(
#                 torch.from_numpy(rgb).unsqueeze(0).contiguous().to(self.rank)
#             )
#             self.mask.append(
#                 torch.from_numpy(rgba[..., 3:] > 0.5).unsqueeze(0).to(self.rank)
#             )
#         print(
#             f"[INFO] single image dataset: load image {self.cfg.image_path} {self.rgb[0].shape}"
#         )

#         self.depth = None
#         self.normal = None

#     def get_all_images(self):
#         return self.rgb[0]

#     def update_step_(self, epoch: int, global_step: int, on_load_weights: bool = False):
#         size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
#         self.height = self.heights[size_ind]
#         if self.height == self.prev_height:
#             return

#         self.prev_height = self.height
#         self.width = self.widths[size_ind]
#         self.directions_unit_focal = self.directions_unit_focals[size_ind]
#         self.focal_length = self.focal_lengths[size_ind]
#         threestudio.debug(f"Training height: {self.height}, width: {self.width}")
#         self.set_rays()
#         self.load_images()









class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    render_cameras: Optional[list] = None

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, extra_opts=None):
    cam_infos = []

    # direct load resized images, not the original ones
    if extra_opts.resolution in [1, 2, 4, 8]:
        tmp_images_folder = images_folder + f'_{str(extra_opts.resolution)}' if extra_opts.resolution != 1 else images_folder
        if not osp.exists(tmp_images_folder):
            print(f"The {tmp_images_folder} is not found, use original resolution images")
        else:
            print(f"Using resized images in {tmp_images_folder}...")
            images_folder = tmp_images_folder
    else:
        print("use original resolution images")

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{},idx={},key={},name={}".format(idx+1, len(cam_extrinsics),idx, key, cam_extrinsics[key].name))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        # print(R)
        # print(T)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE": 
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = osp.join(images_folder, osp.basename(extr.name))
        image_name = osp.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        ### load masks
        mask_path_png = osp.join(osp.dirname(images_folder), "masks", osp.basename(
            image_path).replace(osp.splitext(osp.basename(image_path))[-1], '.png'))

        if osp.exists(mask_path_png) and hasattr(extra_opts, "use_mask") and extra_opts.use_mask:
            mask = cv2.imread(mask_path_png, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            mask = mask.astype(np.float32) / 255.0
        else:
            mask = None

        mono_depth = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, 
                              width=width, height=height, mask=mask, mono_depth=mono_depth)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readColmapCamerasWonder3D(cam_extrinsics, cam_intrinsics, images_folder, extra_opts=None):
    cam_infos = []
    cfg= SingleImageDreamUDFDataModuleConfig()

    # # direct load resized images, not the original ones
    # if extra_opts.resolution in [1, 2, 4, 8]:
    #     tmp_images_folder = images_folder + f'_{str(extra_opts.resolution)}' if extra_opts.resolution != 1 else images_folder
    #     if not osp.exists(tmp_images_folder):
    #         print(f"The {tmp_images_folder} is not found, use original resolution images")
    #     else:
    #         print(f"Using resized images in {tmp_images_folder}...")
    #         images_folder = tmp_images_folder
    # else:
    #     print("use original resolution images")

    image_names = [
            'obj6.jpg', 
            'obj7.jpg', 
            'obj8.jpg', 
            'obj9.jpg', 
            'obj10.jpg', 
            'obj11.jpg', 
            ]


    for uid in range(0,6):
        # sys.stdout.write('\r')
        # # the exact output you're looking for:
        # sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        # sys.stdout.flush()

        # extr = cam_extrinsics[90]
        # # print(cam_intrinsics)
        # intr = cam_intrinsics[extr.camera_id]
        # height = intr.height
        # width = intr.width

        # uid = intr.id
        # R = np.transpose(qvec2rotmat(extr.qvec))
        # T = np.array(extr.tvec)

        # if intr.model=="SIMPLE_PINHOLE":
        #     focal_length_x = intr.params[0]
        #     FovY = focal2fov(focal_length_x, height)
        #     FovX = focal2fov(focal_length_x, width)
        # elif intr.model=="PINHOLE": 
        #     focal_length_x = intr.params[0]
        #     focal_length_y = intr.params[1]
        #     FovY = focal2fov(focal_length_y, height)
        #     FovX = focal2fov(focal_length_x, width)
        # else:
        #     assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"




        elevation_deg = torch.FloatTensor([cfg.default_elevation_deg])
        azimuth_deg = torch.FloatTensor([0,45,90,180,270,315.])
        camera_distance = torch.FloatTensor([cfg.default_camera_distance])

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180
        camera_position= torch.stack(
            [
                camera_distance * torch.cos(elevation) * torch.cos(azimuth),
                camera_distance * torch.cos(elevation) * torch.sin(azimuth),
                camera_distance * torch.sin(elevation) * torch.zeros_like(azimuth),
            ],
            dim=-1,
        )

        center= torch.zeros_like(camera_position)
        up= torch.as_tensor([0, 0, 1], dtype=torch.float32)[None]

        light_position = camera_position
        lookat= F.normalize(center - camera_position, dim=-1)
        right = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_position[:, :, None]],
            dim=-1,
        )
        print(c2w.shape)
        print(c2w[0])
        c2wau = np.zeros((6, 4, 4))
        c2wau[:,:3, :] = c2w
        c2wau[:,3, :3] = 0
        c2wau[:,3, 3] = 1.0
        print(c2wau.shape)
        print(c2wau[0])
        # quit()
        c2wau = torch.tensor(c2wau)
        w2c = c2wau
        w2ct= torch.zeros(6,4,4)
        for i in range(0,6):
            w2c[i]=torch.linalg.inv(w2c[i])
            w2ct[i] = w2c[i].t()
            
        # print(w2c)
        Rs = w2ct[:,:3,:3]
        Ts = w2c[:,3,:3]
        print(Ts)
        R=np.array(Rs[uid])
        T=np.array(Ts[uid])

        FovY = torch.deg2rad(torch.FloatTensor([cfg.default_fovy_deg]))
        FovX = FovY
        
        heights= (
            [cfg.height] if isinstance(cfg.height, int) else cfg.height
        )
        widths= (
            [cfg.width] if isinstance(cfg.width, int) else cfg.width
        )
        assert len(heights) == len(widths)

        # resolution_milestones: List[int]
        # if len(heights) == 1 and len(widths) == 1:
        #     if len(cfg.resolution_milestones) > 0:
        #         threestudio.warn(
        #             "Ignoring resolution_milestones since height and width are not changing"
        #         )
        #     resolution_milestones = [-1]
        # else:
        #     assert len(heights) == len(cfg.resolution_milestones) + 1
        #     resolution_milestones = [-1] + cfg.resolution_milestones

        # directions_unit_focals = [
        #     get_ray_directions(H=height, W=width, focal=1.0)
        #     for (height, width) in zip(heights, widths)
        # ]
        # focal_lengths = [
        #     0.5 * height / torch.tan(0.5 * fovy) for height in heights
        # ]

        height: int = heights[0]
        width: int = widths[0]
        # directions_unit_focal = directions_unit_focals[0]
        # focal_length = focal_lengths[0]
        # set_rays()
        # load_images()
        # prev_height = height

        # print(images_folder)

        # print(image_names[uid])
        image_path = osp.join(images_folder, image_names[uid])
        image_name = osp.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        ### load masks
        mask_path_png = osp.join(osp.dirname(images_folder), "masks", osp.basename(
            image_path).replace(osp.splitext(osp.basename(image_path))[-1], '/0-new.png'))
        # print(mask_path_png)

        if osp.exists(mask_path_png) and hasattr(extra_opts, "use_mask") and extra_opts.use_mask:
            mask = cv2.imread(mask_path_png, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            mask = mask.astype(np.float32) / 255.0
        else:
            mask = None
                
        mono_depth = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, 
                              width=width, height=height, mask=mask, mono_depth=mono_depth)
        cam_infos.append(cam_info)
        uid = uid+1
    
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, extra_opts=None):
    try:
        cameras_extrinsic_file = osp.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = osp.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = osp.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = osp.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=osp.join(path, reading_dir), extra_opts=extra_opts)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    render_cam_infos = generate_ellipse_path_from_camera_infos(cam_infos)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = osp.join(path, "sparse/0/points3D.ply")
    bin_path = osp.join(path, "sparse/0/points3D.bin")
    txt_path = osp.join(path, "sparse/0/points3D.txt")
    if not osp.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    if hasattr(extra_opts, 'sparse_view_num') and extra_opts.sparse_view_num > 0: # means sparse setting
        assert eval == False
        assert osp.exists(osp.join(path, f"sparse_{str(extra_opts.sparse_view_num)}.txt")), "sparse_id.txt not found!"
        ids = np.loadtxt(osp.join(path, f"sparse_{str(extra_opts.sparse_view_num)}.txt"), dtype=np.int32)
        ids_test = np.loadtxt(osp.join(path, f"sparse_test.txt"), dtype=np.int32)
        test_cam_infos = [train_cam_infos[i] for i in ids_test]
        train_cam_infos = [train_cam_infos[i] for i in ids]
        print("Sparse view, only {} images are used for training, others are used for eval.".format(len(ids)))

    # NOTE in sparse condition, we may use random points to initialize the gaussians
    if hasattr(extra_opts, 'init_pcd_name'):
        if extra_opts.init_pcd_name == 'origin':
            pass # None just skip, use better init.
        elif extra_opts.init_pcd_name == 'random':
            raise NotImplementedError
        else:
            # use specific pointcloud, direct load it
            pcd = fetchPly(osp.join(path, extra_opts.init_pcd_name if extra_opts.init_pcd_name.endswith(".ply") 
                                        else extra_opts.init_pcd_name + ".ply"))


    if hasattr(extra_opts, 'transform_the_world') and extra_opts.transform_the_world:
        """
            a experimental feature, we use the transform matrix to transform the pointcloud and the camera poses
        """
        assert osp.exists(osp.join(path, "pcd_transform.txt")), "pcd_transform.txt not found!"
        print("*"*10 , "The world is transformed!!!", "*"*10)
        MLMatrix44 = load_meshlab_file(osp.join(path, "pcd_transform.txt"))
        # this is a 4x4 matrix for transform the pointcloud, new_pc_xyz = (MLMatrix44 @ (homo_xyz.T)).T
        # First, we transform the input pcd, only accept BasicPCD
        assert isinstance(pcd, BasicPointCloud)
        pcd = transform_pcd(pcd, MLMatrix44)
        # then, we need to rotate all the camera poses
        train_cam_infos = transform_cams(train_cam_infos, MLMatrix44)
        test_cam_infos = transform_cams(test_cam_infos, MLMatrix44) if len(test_cam_infos) > 0 else []

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           render_cameras=render_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readColmapSceneInfoWonder3D(path, images, eval, llffhold=8, extra_opts=None):
    # try:
    #     cameras_extrinsic_file = osp.join(path, "sparse/0", "images.bin")
    #     cameras_intrinsic_file = osp.join(path, "sparse/0", "cameras.bin")
    #     cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    #     cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    # except:
    #     cameras_extrinsic_file = osp.join(path, "sparse/0", "images.txt")
    #     cameras_intrinsic_file = osp.join(path, "sparse/0", "cameras.txt")
    #     cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    #     cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCamerasWonder3D(cam_extrinsics=None, cam_intrinsics=None, images_folder=osp.join(path, reading_dir), extra_opts=extra_opts)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    render_cam_infos = generate_ellipse_path_from_camera_infos(cam_infos)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = osp.join(path, "sparse/0/points3D.ply")
    bin_path = osp.join(path, "sparse/0/points3D.bin")
    txt_path = osp.join(path, "sparse/0/points3D.txt")
    # if not osp.exists(ply_path):
        # print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        # try:
        #     xyz, rgb, _ = read_points3D_binary(bin_path)
        # except:
        #     xyz, rgb, _ = read_points3D_text(txt_path)
        # storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    if hasattr(extra_opts, 'sparse_view_num') and extra_opts.sparse_view_num > 0: # means sparse setting
        assert eval == False
        assert osp.exists(osp.join(path, f"sparse_{str(extra_opts.sparse_view_num)}.txt")), "sparse_id.txt not found!"
        ids = np.loadtxt(osp.join(path, f"sparse_{str(extra_opts.sparse_view_num)}.txt"), dtype=np.int32)
        ids_test = np.loadtxt(osp.join(path, f"sparse_test.txt"), dtype=np.int32)
        test_cam_infos = [train_cam_infos[i] for i in ids_test]
        train_cam_infos = [train_cam_infos[i] for i in ids]
        print("Sparse view, only {} images are used for training, others are used for eval.".format(len(ids)))

    # NOTE in sparse condition, we may use random points to initialize the gaussians
    if hasattr(extra_opts, 'init_pcd_name'):
        if extra_opts.init_pcd_name == 'origin':
            pass # None just skip, use better init.
        elif extra_opts.init_pcd_name == 'random':
            raise NotImplementedError
        else:
            # use specific pointcloud, direct load it
            pcd = fetchPly(osp.join(path, extra_opts.init_pcd_name if extra_opts.init_pcd_name.endswith(".ply") 
                                        else extra_opts.init_pcd_name + ".ply"))


    if hasattr(extra_opts, 'transform_the_world') and extra_opts.transform_the_world:
        """
            a experimental feature, we use the transform matrix to transform the pointcloud and the camera poses
        """
        assert osp.exists(osp.join(path, "pcd_transform.txt")), "pcd_transform.txt not found!"
        print("*"*10 , "The world is transformed!!!", "*"*10)
        MLMatrix44 = load_meshlab_file(osp.join(path, "pcd_transform.txt"))
        # this is a 4x4 matrix for transform the pointcloud, new_pc_xyz = (MLMatrix44 @ (homo_xyz.T)).T
        # First, we transform the input pcd, only accept BasicPCD
        assert isinstance(pcd, BasicPointCloud)
        pcd = transform_pcd(pcd, MLMatrix44)
        # then, we need to rotate all the camera poses
        train_cam_infos = transform_cams(train_cam_infos, MLMatrix44)
        test_cam_infos = transform_cams(test_cam_infos, MLMatrix44) if len(test_cam_infos) > 0 else []

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           render_cameras=render_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info







sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfoWonder3D,
    "Wonder3D":readColmapSceneInfoWonder3D
}
