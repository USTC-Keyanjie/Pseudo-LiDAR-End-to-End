import torch
import torch.nn as nn

from models.GANet import GANet
from models.VPNet import VPNet
from models.depth_to_pts import Depth_to_pts
from models.PointRCNN.net.point_rcnn import PointRCNN
from models.PointRCNN.config import cfg

import numpy as np


class MainNet(nn.Module):
    def __init__(self,
                 num_classes='Car',
                 max_high=1,
                 maxdisp=192,
                 use_xyz=True,
                 mode='TRAIN',
                 npoints=16384):
        super(MainNet, self).__init__()
        if cfg.GANET.ENABLED:
            self.GANet = GANet(maxdisp)

        if cfg.RPN.ENABLED or cfg.RCNN.ENABLED:
            self.VPNet = VPNet()

            scale_list = [1, 0.75, 0.5, 0.25]
            self.Depth_to_pts = Depth_to_pts(scale_list, max_high, npoints)
            self.PointRCNN = PointRCNN(num_classes=num_classes, use_xyz=use_xyz, mode=mode)

    def forward(self, data):

        with torch.set_grad_enabled((not cfg.GANET.FIXED) and self.training):
            if cfg.GANET.FIXED:
                self.GANet.eval()
            data = self.GANet(data)

        # disp to depth
        f_bls = data['f_bls']
        disp2 = data['disp2']
        depth_map = f_bls.unsqueeze(-1).repeat(1, disp2.shape[1], disp2.shape[2]) / disp2
        data['depth_map'] = depth_map

        DBH_x, DBH_y = self.VPNet(data)
        data['DBH_x'], data['DBH_y'] = DBH_x, DBH_y

        pts_rect = self.Depth_to_pts(data)
        data['pts_rect'] = pts_rect

        # data = self.PointRCNN(data)
        return data
