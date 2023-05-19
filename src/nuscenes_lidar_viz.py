"""viz lidar nuscenes."""

import json
from typing import Dict
from xml.etree.ElementTree import PI
# from matplotlib.pyplot import box
import open3d as o3d
from open3d import geometry
# import pcl
import numpy as np
import math
import json 
import os
import torch
def get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar):
    center_lidar = [center_lidar[0], center_lidar[1], center_lidar[2]]#x,y,z

    lidar_r = np.matrix(
        [[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]]
    )
    l, w, h = obj_size
    center_lidar[2] = center_lidar[2] - h / 2
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = lidar_r * corners_3d_lidar + np.matrix(center_lidar).T

    return corners_3d_lidar.T

def read_label_bboxes(gt_boxes):
    boxes = []
    for box in gt_boxes:
        obj_size = box[3:6]
        yaw_lidar = box[6:7]
        center_lidar = box[0:3]

        box = get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar)
        boxes.append(np.matrix.tolist(box)) 

    return boxes

def draw_pcd_box(pcd,linesets):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(pcd)
    for i in range(len(linesets)): 
        vis.add_geometry(linesets[i])

    #点云渲染
    opt = vis.get_render_option()
    opt.point_size = 1  #点云大小
    opt.background_color = np.asarray([0, 0, 0])       #点云背景色

    vis.run()
    vis.destroy_window() 

def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (torch.Tensor | np.ndarray): The value to be converted.
        offset (float, optional): Offset to set the value range.
            Defaults to 0.5.
        period ([type], optional): Period of the value. Defaults to np.pi.

    Returns:
        (torch.Tensor | np.ndarray): Value in the range of
            [-offset * period, (1-offset) * period]
    """
    limited_val = val - np.floor(val / period + offset) * period
    return limited_val

def draw_points(points,
                 vis,
                 points_size=2,
                 point_color=(0.5, 0.5, 0.5),
                 mode='xyz'):
    """Draw points on visualizer.

    Args:
        points (numpy.array | torch.tensor, shape=[N, 3+C]):
            points to visualize.
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        points_size (int, optional): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float], optional): the color of points.
            Default: (0.5, 0.5, 0.5).
        mode (str, optional):  indicate type of the input points,
            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.

    Returns:
        tuple: points, color of each point.
    """
    vis.get_render_option().point_size = points_size  # set points size
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    points = points.copy()
    pcd = geometry.PointCloud()
    if mode == 'xyz':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = np.tile(np.array(point_color), (points.shape[0], 1))
    elif mode == 'xyzrgb':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = points[:, 3:6]
        # normalize to [0, 1] for open3d drawing
        if not ((points_colors >= 0.0) & (points_colors <= 1.0)).all():
            points_colors /= 255.0
    else:
        raise NotImplementedError

    pcd.colors = o3d.utility.Vector3dVector(points_colors)
    vis.add_geometry(pcd)

    return pcd, points_colors

def draw_bboxes(bbox3d,
                 vis,
                 points_colors=None,
                 pcd=None,
                 bbox_color=(0, 1, 0),
                 points_in_box_color=(1, 0, 0),
                 rot_axis=2,
                 center_mode='lidar_middle',
                 mode='xyz'):
    """Draw bbox on visualizer and change the color of points inside bbox3d.

    Args:
        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
            3d bbox (x, y, z, x_size, y_size, z_size, yaw) to visualize.
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        points_colors (numpy.array): color of each points.
        pcd (:obj:`open3d.geometry.PointCloud`, optional): point cloud.
            Default: None.
        bbox_color (tuple[float], optional): the color of bbox.
            Default: (0, 1, 0).
        points_in_box_color (tuple[float], optional):
            the color of points inside bbox3d. Default: (1, 0, 0).
        rot_axis (int, optional): rotation axis of bbox. Default: 2.
        center_mode (bool, optional): indicate the center of bbox is
            bottom center or gravity center. available mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str, optional):  indicate type of the input points,
            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.
    """
    if isinstance(bbox3d, torch.Tensor):
        bbox3d = bbox3d.cpu().numpy()
    bbox3d = bbox3d.copy()

    in_box_color = np.array(points_in_box_color)
    for i in range(len(bbox3d)):
        cat = bbox3d[i, 9]
        if cat > 7:
            continue
        center = bbox3d[i, 0:3]
        dim = bbox3d[i, 3:6]
        yaw = bbox3d[i, 6]
        yaw = yaw + np.pi / 2
        yaw = limit_period(yaw, period=np.pi * 2)
        yaw_matrix = np.zeros(3)
        yaw_matrix[rot_axis] = yaw

        mask = np.isnan(bbox3d[i, 7:9])

        print(i, ":", bbox3d[i, 9], ", ", bbox3d[i, 6], "- >", yaw, ", ", bbox3d[i, 7:9], "->" ,mask)
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw_matrix)

        if center_mode == 'lidar_bottom':
            center[rot_axis] += dim[
                rot_axis] / 2  # bottom center to gravity center
        elif center_mode == 'camera_bottom':
            center[rot_axis] -= dim[
                rot_axis] / 2  # bottom center to gravity center
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)

        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color(bbox_color)
        # draw bboxes on visualizer
        vis.add_geometry(line_set)

        # change the color of points which are in box
        if pcd is not None and mode == 'xyz':
            indices = box3d.get_point_indices_within_bounding_box(pcd.points)
            points_colors[indices] = in_box_color

    # update points colors
    if pcd is not None:
        pcd.colors = o3d.utility.Vector3dVector(points_colors)
        vis.update_geometry(pcd)

def show_gt_boxes(meta:Dict):
    # sample_token = meta[""]
    #读取3d box数据
    path = "/home/tzrobot/wangqin/horizon/BEV/centerpoint/data"

    points = np.load(os.path.join(path, str(meta["index"])+"_points.npy")).reshape([-1, 5])
    gt_boxes = np.load(os.path.join(path, str(meta["index"])+"_gt_boxes.npy")).reshape([-1, 10])

    # draw_pcd_box(pcd,line_set)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])  # create coordinate frame
    vis.add_geometry(mesh_frame)

    point_color=(0.5, 0.5, 0.5)
    pcd, points_colors = draw_points(points, vis)
    draw_bboxes(gt_boxes, vis, points_colors, pcd=pcd)
    vis.run()
    vis.destroy_window()

for i in range(50):
    show_gt_boxes(dict(index=i))



def _parse_data(self, raw_data):

        lidar_info = raw_data[b"lidar"]
        metadata_info = raw_data[b"metadata"]
        calib_info = raw_data[b"calib"]
        camera_info = raw_data[b"cam"]

        res = {
            "lidar": {
                "type": "lidar",
                "points": np.array(
                    lidar_info[b"points"], dtype=np.float32
                ).reshape((-1, self.num_point_feature)),
                "annotations": {
                    "boxes": np.array(
                        lidar_info[b"annotations"][b"boxes"], dtype=np.float32
                    ),
                    "names": np.array(
                        [
                            n.decode("utf-8")
                            for n in lidar_info[b"annotations"][b"names"]
                        ]
                    ),
                },
            },
            "metadata": {
                # "image_prefix": self.root_dir,
                "num_point_features": self.num_point_feature,
                # annotation info
                "image_idx": metadata_info[b"image_idx"],
                "image_shape": metadata_info[b"image_shape"],
                "token": metadata_info[b"token"].decode("utf-8"),
                "name": np.array(
                    [n.decode("utf-8") for n in metadata_info[b"name"]]
                ),  # noqa
                "truncated": metadata_info[b"truncated"],
                "occluded": metadata_info[b"occluded"],
                "alpha": metadata_info[b"alpha"],
                "bbox": metadata_info[b"bbox"],
                "dimensions": metadata_info[b"dimensions"],
                "location": metadata_info[b"location"],
                "rotation_y": metadata_info[b"rotation_y"],
            },
            "calib": {
                "P0": np.array(calib_info[b"P0"], dtype=np.float32).reshape(
                    (4, 4)
                ),
                "P1": np.array(calib_info[b"P1"], dtype=np.float32).reshape(
                    (4, 4)
                ),
                "P2": np.array(calib_info[b"P2"], dtype=np.float32).reshape(
                    (4, 4)
                ),
                "P3": np.array(calib_info[b"P3"], dtype=np.float32).reshape(
                    (4, 4)
                ),
                "R0_rect": np.array(
                    calib_info[b"R0_rect"], dtype=np.float32
                ).reshape((4, 4)),
                "Tr_velo_to_cam": np.array(
                    calib_info[b"Tr_velo_to_cam"], dtype=np.float32
                ).reshape((4, 4)),
                "Tr_imu_to_velo": np.array(
                    calib_info[b"Tr_imu_to_velo"], dtype=np.float32
                ).reshape((4, 4)),
            },
            "cam": {
                "annotations": {
                    "bbox": np.array(
                        camera_info[b"annotations"][b"boxes"], dtype=np.float32
                    ),
                    "names": np.array(
                        [
                            n.decode("utf-8")
                            for n in camera_info[b"annotations"][b"names"]
                        ]
                    ),
                }
            },
            "mode": raw_data[b"mode"].decode("utf-8"),
        }
        return res