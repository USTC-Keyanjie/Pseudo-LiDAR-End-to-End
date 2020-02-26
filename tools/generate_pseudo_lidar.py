from __future__ import print_function

import _init_path
import argparse
import os

import numpy as np
import torch.nn.parallel
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import logging
import re
import glob
import time
from tensorboardX import SummaryWriter

# from datasets.kitti_dataset import Kitti_2015_dataset
from datasets.kitti_vpnet_dataset import Kitti_VPNet_Dataset as Dataset
from models.MainNet import MainNet
from models.PointRCNN.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
from libs.pointrcnn.bbox_transform import decode_bbox_target
import libs.pointrcnn.kitti_utils as kitti_utils
import libs.pointrcnn.iou3d.iou3d_utils as iou3d_utils
from models.PointRCNN.net.point_rcnn import PointRCNN
import utils.train_utils.train_utils as train_utils
from libs.pointrcnn.bbox_transform import decode_bbox_target
from tools.kitti_object_eval_python.evaluate import evaluate as kitti_evaluate


# Training settings
# TODO: add mgpus opt
parser = argparse.ArgumentParser(description='arg parser')

# ckpt
parser.add_argument('--ckpt', type=str, default='')
parser.add_argument('--ganet_ckpt', type=str, default='GANet.pth')
parser.add_argument('--pointrcnn_ckpt', type=str, default="")
parser.add_argument('--rpn_ckpt', type=str, default="")
parser.add_argument('--rcnn_ckpt', type=str, default="")

# args
parser.add_argument('--npoints', type=int, default=16384, help='sampled to the number of points')

# super args
parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')

parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument('--save_result', type=bool, default=True, help='save results?')
parser.add_argument('--test', action='store_true', default=False, help='evaluate without ground truth')

# mode
parser.add_argument('--mode', default="TRAINVAL", type=str, help='select mode in "TRAIN", "VAL", "TRAINVAL" or "TEST"')

# root path
parser.add_argument('--cfg_file', default='cfgs/default.yaml', type=str, help='specify the config for training')
parser.add_argument('--ckpt_root', default="../checkpoints/", type=str, help="ckpt root")
parser.add_argument('--data_root', default="../data/KITTI/object/", type=str, help="data root")
parser.add_argument('--list_root', default="../data/KITTI/ImageSets/", type=str, help="training list")
parser.add_argument('--log_root', default="../log/", type=str, help="log file")
parser.add_argument('--output_root', default='../outputs/', type=str, help="location to save result")

args = parser.parse_args()
print(args)


def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def save_kitti_format(sample_id, calib, bbox3d, kitti_output_dir, scores, img_shape):
    corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
    img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

    img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
    img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
    box_valid_mask = np.logical_and(img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8)

    kitti_output_file = os.path.join(kitti_output_dir, '%06d.txt' % sample_id)
    with open(kitti_output_file, 'w') as f:
        for k in range(bbox3d.shape[0]):
            if box_valid_mask[k] == 0:
                continue
            x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
            beta = np.arctan2(z, x)
            alpha = -np.sign(beta) * np.pi / 2 + beta + ry

            print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' %
                  (cfg.CLASSES, alpha, img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2], img_boxes[k, 3],
                   bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2],
                   bbox3d[k, 6], scores[k]), file=f)


if __name__ == "__main__":
    # load cfg file
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]

    mode = args.mode

    # whether use cuda
    cuda = args.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    # anchor size
    MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()

    # create logger
    os.makedirs(args.log_root, exist_ok=True)
    log_file_path = os.path.join(args.log_root, 'log_eval_one.txt')
    logger = create_logger(log_file_path)

    # load dataset
    dataset = Dataset(root_dir=args.data_root,
                      list_root=args.list_root,
                      mode=mode,
                      npoints=args.npoints,
                      classes='Car',
                      logger=logger
                      )
    data_loader = DataLoader(dataset=dataset,
                             num_workers=args.workers,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=dataset.collate_batch
                             )

    print("==> Building model...")
    model = MainNet(num_classes=dataset.num_class, maxdisp=cfg.GANET.MAX_DISP, npoints=args.npoints, mode=mode)
    print("==> Building success!")
    if cuda:
        model = torch.nn.DataParallel(model).cuda()

    # load ganet ckpt
    print("==> Loading checkpoint...")
    ckpt_path = os.path.join(args.ckpt_root, "all", args.ckpt)
    if os.path.isfile(ckpt_path):
        train_utils.load_checkpoint(model.module,
                                    ckpt_file=ckpt_path,
                                    optimizer=None,
                                    logger=logger)

    total_keys = model.state_dict().keys().__len__()
    ganet_ckpt_path = os.path.join(args.ckpt_root, "GANet", args.ganet_ckpt)
    if os.path.isfile(ganet_ckpt_path):
        train_utils.load_part_ckpt(model.module,
                                   ckpt_file=ganet_ckpt_path,
                                   logger=logger,
                                   total_keys=total_keys)

    # load pointrcnn ckpt

    pointrcnn_ckpt_path = os.path.join(args.ckpt_root, "pointrcnn", "all", args.pointrcnn_ckpt)
    rpn_ckpt_path = os.path.join(args.ckpt_root, "pointrcnn", "rpn", args.rpn_ckpt)
    rcnn_ckpt_path = os.path.join(args.ckpt_root, "pointrcnn", "rcnn", args.rcnn_ckpt)

    if os.path.isfile(pointrcnn_ckpt_path):
        train_utils.load_part_ckpt(model.module,
                                   ckpt_file=pointrcnn_ckpt_path,
                                   logger=logger,
                                   total_keys=total_keys)
    elif os.path.isfile(rpn_ckpt_path):
        train_utils.load_part_ckpt(model.module,
                                   ckpt_file=rpn_ckpt_path,
                                   logger=logger,
                                   total_keys=total_keys)
    elif os.path.isfile(rcnn_ckpt_path):
        train_utils.load_part_ckpt(model.module,
                                   ckpt_file=rcnn_ckpt_path,
                                   logger=logger,
                                   total_keys=total_keys)

    result_dir = args.output_root
    final_output_dir = os.path.join(result_dir, 'final_result', 'data')
    os.makedirs(final_output_dir, exist_ok=True)

    if args.save_result:
        roi_output_dir = os.path.join(result_dir, 'roi_result', 'data')
        refine_output_dir = os.path.join(result_dir, 'refine_result', 'data')
        rpn_output_dir = os.path.join(result_dir, 'rpn_result', 'data')
        os.makedirs(rpn_output_dir, exist_ok=True)
        os.makedirs(roi_output_dir, exist_ok=True)
        os.makedirs(refine_output_dir, exist_ok=True)

    logger.info('---- EPOCH %s JOINT EVALUATION ----' % 'no_number')
    logger.info('==> Output file: %s' % result_dir)
    model.eval()

    thresh_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    total_recalled_bbox_list, total_gt_bbox = [0] * 5, 0
    total_roi_recalled_bbox_list = [0] * 5
    cnt = final_total = total_cls_acc = total_cls_acc_refined = total_rpn_iou = 0

    model.eval()
    progress_bar = tqdm(total=len(data_loader), leave=True, desc='eval')
    for iteration, data_dict in enumerate(data_loader):
        sample_id = data_dict['sample_id']
        left, right = data_dict['left'], data_dict['right']
        batch_size = data_dict['P2'].shape[0]

        if mode != 'TEST':
            gt_boxes3d = data_dict['gt_boxes3d']

        input_data = {key: torch.from_numpy(val).contiguous().cuda(non_blocking=True).float()
                      for key, val in data_dict.items()}

        # model inference
        with torch.no_grad():
            ret_dict = model(input_data)

        pts = ret_dict
        pts = pts.cpu().numpy()
        for i in range(pts.shape[0]):
            print(input_data['sample_id'][i].item())
            np.save("../data/KITTI/object/training/pseudo_lidar/%06d.npy" % input_data['sample_id'][i].item(),
                    np.concatenate((pts[i], np.ones((pts.shape[1], 1), dtype='float32')), axis=1))
