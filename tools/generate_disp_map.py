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
parser.add_argument('--ganet_ckpt', type=str, default='GANet.pth')

# super args
parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')

parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument('--mgpus', action='store_true', default=False, help='whether to use multiple gpu')
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
parser.add_argument('--output_dir', default='../outputs/', type=str, help="location to save result")

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


if __name__ == "__main__":
    # load cfg file
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]

    cfg.GANET.ENABLED = True
    cfg.RPN.ENABLED = False
    cfg.RCNN.ENABLED = False

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
        model = model.cuda()
        if args.mgpus:
            model = torch.nn.DataParallel(model).cuda()

    print("==> Loading checkpoint...")
    ganet_ckpt_path = os.path.join(args.ckpt_root, "GANet", args.ganet_ckpt)

    # load ganet ckpt
    if os.path.isfile(ganet_ckpt_path):
        train_utils.load_checkpoint(model,
                                    ckpt_file=ganet_ckpt_path,
                                    optimizer=None,
                                    logger=logger)

    result_dir = args.output_dir
    logger.info('---- Generating disp map ----')
    logger.info('==> Output file: %s' % result_dir)

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

        disp_map = ret_dict['disp2']
        disp_map = disp_map.cpu().numpy()

        for i in range(disp_map.shape[0]):
            np.save(result_dir + '%06d.npy' % sample_id[i], disp_map[i])
            print(sample_id[i])
