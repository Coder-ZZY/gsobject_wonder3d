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

import sys,os
import os.path as osp
from typing import NamedTuple, Optional
from glob import glob
import cv2,torch,math
import torch.nn.functional as F
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from scene.colmap_loader import (qvec2rotmat, read_extrinsics_binary,
                                 read_extrinsics_text, read_intrinsics_binary,
                                 read_intrinsics_text, read_points3D_binary,
                                 read_points3D_text)
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, getWorld2View2, transform_pcd
from utils.image_utils import load_meshlab_file
from utils.camera_utils import transform_cams, CameraInfo, generate_ellipse_path_from_camera_infos
from utils.console_utils import *

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    render_cameras: Optional[list[CameraInfo]] = None
class SingleImageDreamUDFDataModuleConfig:
    # height and width should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    default_height= 96
    default_width= 96
    # resolution_milestones: List[int] = field(default_factory=lambda: [])
    default_elevation_deg = torch.FloatTensor([0.])
    default_azimuth_deg = torch.FloatTensor([0,45,90,180,270,315.])
    default_camera_distance = torch.FloatTensor([1.2])
    default_fovy_deg: float = torch.FloatTensor([60.])
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
    if extra_opts.resolution in [2, 4, 8]:
        tmp_images_folder = images_folder + f'_{str(extra_opts.resolution)}' if extra_opts.resolution != 1 else images_folder
        if not osp.exists(tmp_images_folder):
            print_error(f"The {tmp_images_folder} is not found, use original resolution images")
        else:
            print_info(f"Using resized images in {tmp_images_folder}...")
            images_folder = tmp_images_folder
    else:
        print_info("use original resolution images")

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

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
def readWonder3DCameras(images_folder,extra_opts=None,extension=".jpg"):
    cam_infos = []
    image_list = sorted(glob(os.path.join(images_folder, f"*{extension}")))
    # direct load resized images, not the original ones
    if extra_opts.resolution in [2, 4, 8]:
        tmp_images_folder = images_folder + f'_{str(extra_opts.resolution)}' if extra_opts.resolution != 1 else images_folder
        if not osp.exists(tmp_images_folder):
            print(f"The {tmp_images_folder} is not found, use original resolution images")
        else:
            print(f"Using resized images in {tmp_images_folder}...")
            images_folder = tmp_images_folder
    else:
        print("use original resolution images")
    cfg = SingleImageDreamUDFDataModuleConfig()
    elevation = cfg.default_elevation_deg * math.pi / 180
    azimuth = cfg.default_azimuth_deg * math.pi / 180
    camera_position = torch.stack(
        [
            cfg.default_camera_distance * torch.cos(elevation) * torch.cos(azimuth),
            cfg.default_camera_distance * torch.cos(elevation) * torch.sin(azimuth),
            cfg.default_camera_distance * torch.sin(elevation) * torch.zeros_like(azimuth),
        ],
        dim=-1,
    )
    center= torch.zeros_like(camera_position)
    up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None]

    lookat= F.normalize(center - camera_position, dim=-1)
    right= F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    #[6,3,4]
    c2ws = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_position[:, :, None]],
        dim=-1,
    )
    fovY = torch.deg2rad(torch.FloatTensor([cfg.default_fovy_deg]))
    fovX = fovY
    print_info(f"camera number: {len(image_list)}")
    for idx in range(0,len(image_list)):
        image_path = image_list[idx]
        image_name = osp.basename(image_path).split(".")[0]
        image = np.array(Image.open(image_path))
        width, height = image.shape[1], image.shape[0]
        c2w = np.vstack((c2ws[idx].cpu().numpy(),np.array([[0,0,0,1]])))
        c2w[:3,2] = -1 * c2w[:3,2]
        w2c = np.linalg.inv(c2w)
        R = w2c[:3,:3].transpose()
        T = w2c[:3,3]
        #load mask png
        mask_path_png = osp.join(osp.dirname(images_folder), "masks", osp.basename(image_path).replace(osp.splitext(osp.basename(image_path))[-1], '.png'))
        if osp.exists(mask_path_png) and hasattr(extra_opts, "use_mask") and extra_opts.use_mask:
            mask = cv2.imread(mask_path_png, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            mask = mask.astype(np.float32) / 255.0
        else:
            mask = None
        mono_depth = None
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=fovY, FovX=fovX, image=image,
                              image_path=image_path, image_name=image_name, 
                              width=width, height=height, mask=mask, mono_depth=mono_depth)
        cam_infos.append(cam_info)
    return cam_infos

    
def readWonder3DSceneInfo(path,eval,llffhold=8,extra_opts=None,extension=".png",images = None):
    reading_dir = "images"
    #read camera infos
    cam_infos_unsorted = readWonder3DCameras(images_folder=osp.join(path, reading_dir), extra_opts=extra_opts)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    #TODO:what this?
    render_cam_infos = generate_ellipse_path_from_camera_infos(cam_infos)
    nerf_normalization = getNerfppNorm(train_cam_infos)
    #read ply infos, if exist
    # ply_path = osp.join(path, "sparse/0/points3D.ply")
    # bin_path = osp.join(path, "sparse/0/points3D.bin")
    # txt_path = osp.join(path, "sparse/0/points3D.txt")
    # if not osp.exists(ply_path):
    #     try:
    #         print_info("trying to convert point3d.bin to .ply, will happen only the first time you open the scene.")
    #         xyz, rgb, _ = read_points3D_binary(bin_path)
    #     except:
    #         try:
    #             xyz, rgb, _ = read_points3D_text(txt_path)
    #             storePly(ply_path, xyz, rgb)
    #         except:
    #             print_warning("No point cloud found in the dataset")
    #             xyz,rgb = None,None
    pcd = None
    ply_path = ""
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
            ply_path = osp.join(path, extra_opts.init_pcd_name if extra_opts.init_pcd_name.endswith(".ply") else extra_opts.init_pcd_name + ".ply")
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
    "Colmap": readColmapSceneInfo,
    "Wonder3D": readWonder3DSceneInfo
}
