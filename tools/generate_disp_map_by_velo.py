import _init_path
import argparse
import os
import cv2

import numpy as np
import tqdm

from utils.kitti_util import Calibration


def generate_depth_map_from_velo(pc_velo, height, width, calib):
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < width - 1) & (pts_2d[:, 0] >= 0) & \
               (pts_2d[:, 1] < height - 1) & (pts_2d[:, 1] >= 0)
    fov_inds = fov_inds & (pc_velo[:, 0] > 2)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)
    depth_map = np.zeros((height, width)) - 1
    # imgfov_pts_2d = np.round(imgfov_pts_2d * 256).astype(int)
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        depth_map[int(imgfov_pts_2d[i, 1]), int(imgfov_pts_2d[i, 0])] = depth
    return depth_map


# def generate_disp_map_from_velo(pc_velo, height, width, calib):
#     depth_map = generate_depth_map_from_velo(pc_velo, height, width, calib)
#     baseline = calib.b_x_3 - calib.b_x_2
#     disp_map = (calib.f_u_2 * baseline) / depth_map
#     disp_map[disp_map < 0] = 0
#     return disp_map

def generate_disp_map_from_depthmap(depth_map, calib):
    baseline = calib.b_x_3 - calib.b_x_2
    disp_map = (calib.f_u_2 * baseline) / depth_map
    disp_map[disp_map < 0] = -1
    return disp_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Disparity')
    parser.add_argument('--data_path', type=str, default='../data/KITTI/object/testing')
    parser.add_argument('--split_file', type=str, default='../data/KITTI/ImageSets/test.txt')
    args = parser.parse_args()

    assert os.path.isdir(args.data_path)
    lidar_dir = args.data_path + '/velodyne/'
    calib_dir = args.data_path + '/calib/'
    image_dir = args.data_path + '/image_2/'
    disparity_dir = args.data_path + '/disp_map/'
    # depthmap_dir = args.data_path + '/depth_map/'

    assert os.path.isdir(lidar_dir)
    assert os.path.isdir(calib_dir)
    assert os.path.isdir(image_dir)

    if not os.path.isdir(disparity_dir):
        os.makedirs(disparity_dir)

    lidar_files = [x for x in os.listdir(lidar_dir) if x[-3:] == 'bin']
    lidar_files = sorted(lidar_files)

    assert os.path.isfile(args.split_file)
    with open(args.split_file, 'r') as f:
        file_names = [x.strip() for x in f.readlines()]

    for fn in tqdm.tqdm(lidar_files):
        predix = fn[:-4]
        if predix not in file_names:
            continue
        calib_file = '{}/{}.txt'.format(calib_dir, predix)
        calib = Calibration(calib_file)
        # load point cloud
        lidar = np.fromfile(lidar_dir + '/' + fn, dtype=np.float32).reshape((-1, 4))[:, :3]
        image_file = '{}/{}.png'.format(image_dir, predix)
        image = cv2.imread(image_file)
        height, width = image.shape[:2]
        depth = generate_depth_map_from_velo(lidar, height, width, calib)
        # np.save(depthmap_dir + '/' + predix, depth)
        disp = generate_disp_map_from_depthmap(depth, calib)
        np.save(disparity_dir + '/' + predix, disp)
