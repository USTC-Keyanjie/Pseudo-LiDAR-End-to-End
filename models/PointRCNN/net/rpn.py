import math

import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from libs.pointrcnn.Delaunay import Delaunay
from models.PointRCNN.rpn.proposal_layer import ProposalLayer
import libs.pointnet2_lib.pointnet2.pytorch_utils as pt_utils
import libs.pointrcnn.loss_utils as loss_utils
from models.PointRCNN.config import cfg
import importlib
import libs.pointrcnn.kitti_utils as kitti_utils
import models.PointRCNN.net.pointnet2_msg


def boxes3d_to_corners3d_torch(boxes3d, flip=False):
    """
    :param boxes3d: (B, N, 7) [x, y, z, h, w, l, ry]
    :return: corners_rotated: (B, N, 8, 3)
    """
    bs, n, _ = boxes3d.shape
    h, w, l, ry = boxes3d[:, :, 3:4], boxes3d[:, :, 4:5], boxes3d[:, :, 5:6], boxes3d[:, :, 6:7]  # (B, N, 1)
    if flip:
        ry = ry + math.pi
    centers = boxes3d[:, :, 0:3]  # (B, N, 3)
    zeros = torch.cuda.FloatTensor(bs, n, 1).fill_(0)
    ones = torch.cuda.FloatTensor(bs, n, 1).fill_(1)

    x_corners = torch.cat([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2.], dim=-1)  # (B, N, 8)
    y_corners = torch.cat([zeros, zeros, zeros, zeros, -h, -h, -h, -h], dim=-1)  # (B, N, 8)
    z_corners = torch.cat([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dim=-1)  # (B, N, 8)
    corners = torch.cat((x_corners.unsqueeze(dim=2), y_corners.unsqueeze(dim=2), z_corners.unsqueeze(dim=2)), dim=2) # (B, N, 3, 8)

    cosa, sina = torch.cos(ry), torch.sin(ry)  # (B, N, 1)
    raw_1 = torch.cat([cosa, zeros, sina], dim=-1)
    raw_2 = torch.cat([zeros, ones, zeros], dim=-1)
    raw_3 = torch.cat([-sina, zeros, cosa], dim=-1)
    R = torch.cat((raw_1.unsqueeze(dim=2), raw_2.unsqueeze(dim=2), raw_3.unsqueeze(dim=2)), dim=2)  # (B, N, 3, 3)

    corners_rotated = torch.matmul(R, corners)  # (B, N, 3, 8)
    corners_rotated = corners_rotated + centers.unsqueeze(dim=-1).expand(-1, -1, -1, 8)  # (B, N, 3, 8)
    corners_rotated = corners_rotated.permute(0, 1, 3, 2)  # (B, N, 8, 3)
    return corners_rotated


def enlarge_box3d(boxes3d, extra_width):
    """
    :param boxes3d: (B, N, 7) [x, y, z, h, w, l, ry]
    """
    large_boxes3d = boxes3d.clone()
    large_boxes3d[:, :, 3:6] += extra_width * 2
    large_boxes3d[:, :, 1] += extra_width
    return large_boxes3d


def generate_rpn_labels(input_data):
    pts_rect = input_data['pts_rect']
    gt_boxes3d = input_data['gt_boxes3d']

    # ---------
    # for test
    # pts = pts_rect.cpu().numpy()[0]
    # np.save("/home/kyj/workspaces/PointRCNN/data/temp/%06d.npy" % input_data['sample_id'].item(),
    #         np.concatenate((pts, np.ones((pts.shape[0], 1), dtype='float32')), axis=1))
    # ---------

    bs, num, _ = pts_rect.shape
    rpn_cls_label = torch.zeros(bs, num, device='cuda').long()
    rpn_reg_label = torch.zeros(bs, num, 7, device='cuda').float()  # dx, dy, dz, ry, h, w, l
    gt_corners = boxes3d_to_corners3d_torch(gt_boxes3d, flip=False)
    extend_gt_boxes3d = enlarge_box3d(gt_boxes3d, extra_width=0.2)
    extend_gt_corners = boxes3d_to_corners3d_torch(extend_gt_boxes3d, flip=False)

    pts_rect_numpy = pts_rect.cpu().detach().numpy()
    gt_corners_numpy = gt_corners.cpu().detach().numpy()
    extend_gt_corners_numpy = extend_gt_corners.cpu().detach().numpy()

    for b in range(gt_corners.shape[0]):
        for n in range(gt_corners.shape[1]):
            fg_pt_flag = kitti_utils.in_hull(pts_rect_numpy[b], gt_corners_numpy[b, n])

            # enlarge the bbox3d, ignore nearby points
            fg_enlarge_flag = kitti_utils.in_hull(pts_rect_numpy[b], extend_gt_corners_numpy[b, n])
            ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)

            fg_pt_flag = torch.from_numpy(fg_pt_flag)
            ignore_flag = torch.from_numpy(ignore_flag)
            fg_pts_rect = pts_rect[b, fg_pt_flag]
            rpn_cls_label[b, fg_pt_flag] = 1
            rpn_cls_label[b, ignore_flag] = -1

            # pixel offset of object center
            center3d = gt_boxes3d[b, n, 0:3].clone()  # (x, y, z)
            center3d[1] -= gt_boxes3d[b, n, 3] / 2  # ????? 为什么y要减去h/2
            rpn_reg_label[b, fg_pt_flag, 0:3] = center3d - fg_pts_rect  # Now y is the true center of 3d box 20180928

            # size and angle encoding
            rpn_reg_label[b, fg_pt_flag, 3] = gt_boxes3d[b, n, 3]  # h
            rpn_reg_label[b, fg_pt_flag, 4] = gt_boxes3d[b, n, 4]  # w
            rpn_reg_label[b, fg_pt_flag, 5] = gt_boxes3d[b, n, 5]  # l
            rpn_reg_label[b, fg_pt_flag, 6] = gt_boxes3d[b, n, 6]  # ry

    return rpn_cls_label, rpn_reg_label


class RPN(nn.Module):
    def __init__(self, use_xyz=True, mode='TRAIN'):
        super().__init__()
        self.training_mode = (mode == 'TRAIN')

        MODEL = importlib.import_module(cfg.RPN.BACKBONE)
        self.backbone_net = MODEL.get_model(input_channels=int(cfg.RPN.USE_INTENSITY), use_xyz=use_xyz)

        # classification branch
        cls_layers = []
        pre_channel = cfg.RPN.FP_MLPS[0][-1]
        for k in range(0, cfg.RPN.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RPN.CLS_FC[k], bn=cfg.RPN.USE_BN))
            pre_channel = cfg.RPN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, 1, activation=None))
        if cfg.RPN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(cfg.RPN.DP_RATIO))
        self.rpn_cls_layer = nn.Sequential(*cls_layers)

        # regression branch
        per_loc_bin_num = int(cfg.RPN.LOC_SCOPE / cfg.RPN.LOC_BIN_SIZE) * 2
        if cfg.RPN.LOC_XZ_FINE:
            reg_channel = per_loc_bin_num * 4 + cfg.RPN.NUM_HEAD_BIN * 2 + 3
        else:
            reg_channel = per_loc_bin_num * 2 + cfg.RPN.NUM_HEAD_BIN * 2 + 3
        reg_channel += 1  # reg y

        reg_layers = []
        pre_channel = cfg.RPN.FP_MLPS[0][-1]
        for k in range(0, cfg.RPN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RPN.REG_FC[k], bn=cfg.RPN.USE_BN))
            pre_channel = cfg.RPN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RPN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RPN.DP_RATIO))
        self.rpn_reg_layer = nn.Sequential(*reg_layers)

        if cfg.RPN.LOSS_CLS == 'DiceLoss':
            self.rpn_cls_loss_func = loss_utils.DiceLoss(ignore_target=-1)
        elif cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
            self.rpn_cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RPN.FOCAL_ALPHA[0],
                                                                               gamma=cfg.RPN.FOCAL_GAMMA)
        elif cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':
            self.rpn_cls_loss_func = F.binary_cross_entropy
        else:
            raise NotImplementedError

        self.proposal_layer = ProposalLayer(mode=mode)
        self.init_weights()

    def init_weights(self):
        if cfg.RPN.LOSS_CLS in ['SigmoidFocalLoss']:
            pi = 0.01
            nn.init.constant_(self.rpn_cls_layer[2].conv.bias, -np.log((1 - pi) / pi))

        nn.init.normal_(self.rpn_reg_layer[-1].conv.weight, mean=0, std=0.001)

    def forward(self, input_data):
        """
        :param input_data: dict (point_cloud)
        :return:
        """
        pts_input = input_data['pts_rect']
        backbone_xyz, backbone_features = self.backbone_net(pts_input)  # (B, N, 3), (B, C, N)

        rpn_cls_label, rpn_reg_label = generate_rpn_labels(input_data)

        rpn_cls = self.rpn_cls_layer(backbone_features).transpose(1, 2).contiguous()  # (B, N, 1)
        rpn_reg = self.rpn_reg_layer(backbone_features).transpose(1, 2).contiguous()  # (B, N, C)

        ret_dict = {'rpn_cls': rpn_cls, 'rpn_reg': rpn_reg,
                    'backbone_xyz': backbone_xyz, 'backbone_features': backbone_features,
                    'rpn_cls_label': rpn_cls_label, 'rpn_reg_label': rpn_reg_label}

        return ret_dict
