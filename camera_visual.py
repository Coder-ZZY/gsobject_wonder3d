from turtle import position
import numpy as np
import open3d as o3d
import os,argparse,json

from sympy import N, li
def cameras_reader(cam_file_path):
    cam_list = json.load(open(cam_file_path))
    intrinsics = []
    poses = []
    positions = []
    for cam in cam_list:
        id  = cam['id']
        image_size = (cam['width'],cam['height'])
        position = np.array(cam["position"])
        rotation = np.array(cam["rotation"])
        fy = cam["fy"]
        fx = cam["fx"]
        # 构建外参矩阵
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = position
        t_after_R = -pose[:3,:3].T @ pose[:3,3]
        t_after_R = t_after_R.reshape(1,3)
        pose[:3, :3] = pose[:3, :3].T
        pose[:3,3] = t_after_R
        # 构建内参矩阵
        intrinsic = np.array([[fx, 0, cam['width'] / 2],
                              [0, fy, cam['height'] / 2],
                              [0, 0, 1]])
        intrinsics.append(intrinsic)
        poses.append(pose)
        positions.append(position.T)
    return poses,intrinsics,image_size,positions
def get_camera_frustum(img_size, K, W2C, frustum_length=0.5, color=[0., 1., 0.]):
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / K[0, 0]) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / K[1, 1]) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],                          # frustum origin
                               [-half_w, -half_h, frustum_length],    # top-left image corner
                               [half_w, -half_h, frustum_length],     # top-right image corner
                               [half_w, half_h, frustum_length],      # bottom-right image corner
                               [-half_w, half_h, frustum_length]])    # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    C2W = np.linalg.inv(W2C)
    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    return frustum_points, frustum_lines, frustum_colors
def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)
    return lineset
def visualize_cameras(intrinsics, poses, sphere_radius=1, positions= None,img_size=(1920, 1080) , camera_size=0.1, geometry_file=None, geometry_type='mesh'):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))
    # geometry_file = None
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
    things_to_draw = [ coord_frame]
    frustums = []
    # colmap相机
    for intrinsic, pose in zip(intrinsics, poses):     # [intrinsics, poses]
        color = [0, 1, 0] # 绿色
        frustums.append(get_camera_frustum(img_size, intrinsic, pose, frustum_length=camera_size, color=color))

    cameras = frustums2lineset(frustums)
    things_to_draw.append(cameras)
    lines = []
    for i in range(len(positions)-1):
        lines.append([i,i+1])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(positions)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    things_to_draw.append(line_set)
    if geometry_file is not None:
        if geometry_type == 'mesh':
            geometry = o3d.io.read_triangle_mesh(geometry_file)
            geometry.compute_vertex_normals()
        elif geometry_type == 'pointcloud':
            geometry = o3d.io.read_point_cloud(geometry_file)
        else:
            raise Exception('Unknown geometry_type: ', geometry_type)

        things_to_draw.append(geometry)

    o3d.visualization.draw_geometries(things_to_draw)
def main():
    parser = argparse.ArgumentParser(description='visual cameras')
    parser.add_argument('-m','--model_path', type=str, default="", help='path to models')
    parser.add_argument('-s','--source_path', type=str, default="", help='path to data')
    parser.add_argument('--vis_geo', action='store_true', help='visualize geometry')
    args = parser.parse_args()
    if os.path.exists(os.path.join(args.source_path,'train_cameras.json')):
        poses,intrinsics,image_size,positions = cameras_reader(os.path.join(args.source_path,'train_cameras.json'))
        visualize_cameras(intrinsics,poses,img_size=image_size,positions=positions,geometry_file=os.path.join("visual_hull.ply"),geometry_type='pointcloud')
if __name__ == '__main__':
    main()