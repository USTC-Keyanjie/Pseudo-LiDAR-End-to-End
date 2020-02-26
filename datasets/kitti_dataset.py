""" Helper class and functions for loading KITTI objects

"""
from __future__ import print_function

import os
import numpy as np
import torch.utils.data as data
from PIL import Image

import libs.pointrcnn.calibration as calibration
import libs.pointrcnn.kitti_utils as kitti_utils


class Kitti_2015_Dataset(data.Dataset):
    """Load and parse object data into a usable format."""

    def __init__(self, root_dir, list_root, mode='train'):
        """root_dir contains training and testing folders"""

        super(Kitti_2015_Dataset, self).__init__()
        self.mode = mode
        assert self.mode in ['TRAIN', 'VAL', 'TRAINVAL', 'TEST'], "Unknown mode Error!"
        self.split = "testing" if mode == "TEST" else "training"

        if self.mode == 'TRAIN':
            self.num_sample = 3712
        elif self.mode == 'VAL':
            self.num_sample = 3769
        elif self.mode == 'TRAINVAL':
            self.num_sample = 7481
        elif self.mode == 'TEST':
            self.num_sample = 7518
        else:
            raise NotImplementedError

        file_list = os.path.join(list_root, self.mode.lower() + '.txt')
        self.image_idx_list = [x.strip() for x in open(file_list).readlines()]
        assert len(self.image_idx_list) == self.num_sample, \
            f"file_list length error!\n" \
            f"length of file_list should be {self.num_sample}, but get {len(self.image_idx_list)}"

        self.root_dir = os.path.join(root_dir, self.split)
        self.image_2_dir = os.path.join(self.root_dir, 'image_2')
        self.image_3_dir = os.path.join(self.root_dir, 'image_3')
        self.pseudo_lidar_dir = os.path.join(self.root_dir, 'pseudo_lidar')
        self.lidar_dir = os.path.join(self.root_dir, 'velodyne')
        self.calib_dir = os.path.join(self.root_dir, 'calib')
        self.label_dir = os.path.join(self.root_dir, 'label_2')
        self.plane_dir = os.path.join(self.root_dir, 'planes')
        self.disp_dir = os.path.join(self.root_dir, 'disp_map')

    def get_image_2(self, idx):
        # assert False, 'DO NOT USE cv2 NOW, AVOID DEADLOCK'
        import cv2
        # cv2.setNumThreads(0)  # for solving deadlock when switching epoch
        img_file = os.path.join(self.image_2_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return cv2.imread(img_file)  # (H, W, 3) BGR mode

    def get_image_3(self, idx):
        # assert False, 'DO NOT USE cv2 NOW, AVOID DEADLOCK'
        import cv2
        # cv2.setNumThreads(0)  # for solving deadlock when switching epoch
        img_file = os.path.join(self.image_3_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return cv2.imread(img_file)  # (H, W, 3) BGR mode

    '''
    def get_pseudo_lidar(self, idx):
        lidar_file = os.path.join(self.pseudo_lidar_dir, '%06d.npy' % idx)
        assert os.path.exists(lidar_file)
        return np.load(lidar_file)

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '%06d.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    '''

    def get_image_shape(self, idx):
        img_file = os.path.join(self.image_2_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        im = Image.open(img_file)
        width, height = im.size
        return height, width, 3

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return kitti_utils.get_objects_from_label(label_file)

    def get_road_plane(self, idx):
        plane_file = os.path.join(self.plane_dir, '%06d.txt' % idx)
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    def get_disp_map(self, idx):
        disp_file = os.path.join(self.disp_dir, '%06d.npy' % idx)
        return np.load(disp_file)

    def __len__(self):
        return self.num_sample

    def __getitem__(self, index):
        raise NotImplementedError
