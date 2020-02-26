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
parser.add_argument('--mode', default="VAL", type=str, help='select mode in "TRAIN", "VAL", "TRAINVAL" or "TEST"')

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


    print("==> Loading checkpoint...")
    ckpt_path = os.path.join(args.ckpt_root, "all", args.ckpt)
    ganet_ckpt_path = os.path.join(args.ckpt_root, "GANet", args.ganet_ckpt)
    pointrcnn_ckpt_path = os.path.join(args.ckpt_root, "pointrcnn", "all", args.pointrcnn_ckpt)
    rpn_ckpt_path = os.path.join(args.ckpt_root, "pointrcnn", "rpn", args.rpn_ckpt)
    rcnn_ckpt_path = os.path.join(args.ckpt_root, "pointrcnn", "rcnn", args.rcnn_ckpt)

    if os.path.isfile(ckpt_path):
        train_utils.load_checkpoint(model.module,
                                    ckpt_file=ckpt_path,
                                    optimizer=None,
                                    logger=logger)
    else:
        total_keys = model.state_dict().keys().__len__()
        # load ganet ckpt
        if os.path.isfile(ganet_ckpt_path):
            train_utils.load_part_ckpt(model.module,
                                       ckpt_file=ganet_ckpt_path,
                                       logger=logger,
                                       total_keys=total_keys)
        # load pointrcnn ckpt
        if os.path.isfile(pointrcnn_ckpt_path):
            train_utils.load_part_ckpt(model.module,
                                       ckpt_file=pointrcnn_ckpt_path,
                                       logger=logger,
                                       total_keys=total_keys)
        elif os.path.isfile(rpn_ckpt_path) and os.path.isfile(rcnn_ckpt_path):
            train_utils.load_part_ckpt(model.module,
                                       ckpt_file=rpn_ckpt_path,
                                       logger=logger,
                                       total_keys=total_keys)
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

        roi_scores_raw = ret_dict['roi_scores_raw']  # (B, M)
        roi_boxes3d = ret_dict['rois']  # (B, M, 7)
        seg_result = ret_dict['seg_result'].long()  # (B, N)

        rcnn_cls = ret_dict['rcnn_cls'].view(batch_size, -1, ret_dict['rcnn_cls'].shape[1])
        rcnn_reg = ret_dict['rcnn_reg'].view(batch_size, -1, ret_dict['rcnn_reg'].shape[1])  # (B, M, C)

        # bounding box regression
        anchor_size = MEAN_SIZE
        if cfg.RCNN.SIZE_RES_ON_ROI:
            assert False

        pred_boxes3d = decode_bbox_target(roi_boxes3d.view(-1, 7), rcnn_reg.view(-1, rcnn_reg.shape[-1]),
                                          anchor_size=anchor_size,
                                          loc_scope=cfg.RCNN.LOC_SCOPE,
                                          loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                          num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                          get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                                          loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                                          get_ry_fine=True).view(batch_size, -1, 7)

        # scoring
        if rcnn_cls.shape[2] == 1:
            raw_scores = rcnn_cls  # (B, M, 1)

            norm_scores = torch.sigmoid(raw_scores)
            pred_classes = (norm_scores > cfg.RCNN.SCORE_THRESH).long()
        else:
            pred_classes = torch.argmax(rcnn_cls, dim=1).view(-1)
            cls_norm_scores = F.softmax(rcnn_cls, dim=1)
            raw_scores = rcnn_cls[:, pred_classes]
            norm_scores = cls_norm_scores[:, pred_classes]

        # evaluation
        recalled_num = gt_num = rpn_iou = 0
        if not args.test:
            if not cfg.RPN.FIXED:
                rpn_cls_label, rpn_reg_label = ret_dict['rpn_cls_label'], ret_dict['rpn_reg_label']
                # rpn_cls_label = torch.from_numpy(rpn_cls_label).cuda(non_blocking=True).long()

            gt_boxes3d = data_dict['gt_boxes3d']

            for k in range(batch_size):
                # calculate recall
                cur_gt_boxes3d = gt_boxes3d[k]
                tmp_idx = cur_gt_boxes3d.__len__() - 1

                while tmp_idx >= 0 and cur_gt_boxes3d[tmp_idx].sum() == 0:
                    tmp_idx -= 1

                if tmp_idx >= 0:
                    cur_gt_boxes3d = cur_gt_boxes3d[:tmp_idx + 1]

                    cur_gt_boxes3d = torch.from_numpy(cur_gt_boxes3d).cuda(non_blocking=True).float()
                    iou3d = iou3d_utils.boxes_iou3d_gpu(pred_boxes3d[k], cur_gt_boxes3d)
                    gt_max_iou, _ = iou3d.max(dim=0)
                    refined_iou, _ = iou3d.max(dim=1)

                    for idx, thresh in enumerate(thresh_list):
                        total_recalled_bbox_list[idx] += (gt_max_iou > thresh).sum().item()
                    recalled_num += (gt_max_iou > 0.7).sum().item()
                    gt_num += cur_gt_boxes3d.shape[0]
                    total_gt_bbox += cur_gt_boxes3d.shape[0]

                    # original recall
                    iou3d_in = iou3d_utils.boxes_iou3d_gpu(roi_boxes3d[k], cur_gt_boxes3d)
                    gt_max_iou_in, _ = iou3d_in.max(dim=0)

                    for idx, thresh in enumerate(thresh_list):
                        total_roi_recalled_bbox_list[idx] += (gt_max_iou_in > thresh).sum().item()

                if not cfg.RPN.FIXED:
                    fg_mask = rpn_cls_label > 0
                    correct = ((seg_result == rpn_cls_label) & fg_mask).sum().float()
                    union = fg_mask.sum().float() + (seg_result > 0).sum().float() - correct
                    rpn_iou = correct / torch.clamp(union, min=1.0)
                    total_rpn_iou += rpn_iou.item()

        disp_dict = {'mode': mode, 'recall': '%d/%d' % (total_recalled_bbox_list[3], total_gt_bbox)}
        progress_bar.set_postfix(disp_dict)
        progress_bar.update()

        if args.save_result:
            # save roi and refine results
            roi_boxes3d_np = roi_boxes3d.cpu().numpy()
            pred_boxes3d_np = pred_boxes3d.cpu().numpy()
            roi_scores_raw_np = roi_scores_raw.cpu().numpy()
            raw_scores_np = raw_scores.cpu().numpy()

            rpn_cls_np = ret_dict['rpn_cls'].cpu().numpy()
            rpn_xyz_np = ret_dict['backbone_xyz'].cpu().numpy()
            seg_result_np = seg_result.cpu().numpy()
            output_data = np.concatenate((rpn_xyz_np, rpn_cls_np.reshape(batch_size, -1, 1),
                                          seg_result_np.reshape(batch_size, -1, 1)), axis=2)

            for k in range(batch_size):
                cur_sample_id = sample_id[k]
                calib = dataset.get_calib(cur_sample_id)
                image_shape = dataset.get_image_shape(cur_sample_id)
                save_kitti_format(cur_sample_id, calib, roi_boxes3d_np[k], roi_output_dir,
                                  roi_scores_raw_np[k], image_shape)
                save_kitti_format(cur_sample_id, calib, pred_boxes3d_np[k], refine_output_dir,
                                  raw_scores_np[k], image_shape)

                output_file = os.path.join(rpn_output_dir, '%06d.npy' % cur_sample_id)
                np.save(output_file, output_data.astype(np.float32))

        # scores thresh
        inds = norm_scores > cfg.RCNN.SCORE_THRESH

        for k in range(batch_size):
            cur_inds = inds[k].view(-1)
            if cur_inds.sum() == 0:
                continue

            pred_boxes3d_selected = pred_boxes3d[k, cur_inds]
            raw_scores_selected = raw_scores[k, cur_inds]
            norm_scores_selected = norm_scores[k, cur_inds]

            # NMS thresh
            # rotated nms
            boxes_bev_selected = kitti_utils.boxes3d_to_bev_torch(pred_boxes3d_selected)
            keep_idx = iou3d_utils.nms_gpu(boxes_bev_selected, raw_scores_selected, cfg.RCNN.NMS_THRESH).view(-1)
            pred_boxes3d_selected = pred_boxes3d_selected[keep_idx]
            scores_selected = raw_scores_selected[keep_idx]
            pred_boxes3d_selected, scores_selected = pred_boxes3d_selected.cpu().numpy(), scores_selected.cpu().numpy()

            cur_sample_id = sample_id[k]
            calib = dataset.get_calib(cur_sample_id)
            final_total += pred_boxes3d_selected.shape[0]
            image_shape = dataset.get_image_shape(cur_sample_id)
            save_kitti_format(cur_sample_id, calib, pred_boxes3d_selected, final_output_dir, scores_selected,
                              image_shape)

    progress_bar.close()
    # dump empty files
    split_file = os.path.join(dataset.root_dir, '..', '..', 'ImageSets', dataset.mode.lower() + '.txt')
    split_file = os.path.abspath(split_file)
    image_idx_list = [x.strip() for x in open(split_file).readlines()]
    empty_cnt = 0
    for k in range(image_idx_list.__len__()):
        cur_file = os.path.join(final_output_dir, '%s.txt' % image_idx_list[k])
        if not os.path.exists(cur_file):
            with open(cur_file, 'w') as temp_f:
                pass
            empty_cnt += 1
            logger.info('empty_cnt=%d: dump empty file %s' % (empty_cnt, cur_file))

    ret_dict = {'empty_cnt': empty_cnt}

    logger.info('-------------------performance of epoch %s---------------------' % 'no_number')
    logger.info(str(datetime.now()))

    avg_rpn_iou = (total_rpn_iou / max(cnt, 1.0))
    avg_cls_acc = (total_cls_acc / max(cnt, 1.0))
    avg_cls_acc_refined = (total_cls_acc_refined / max(cnt, 1.0))
    avg_det_num = (final_total / max(len(dataset), 1.0))
    logger.info('final average detections: %.3f' % avg_det_num)
    logger.info('final average rpn_iou refined: %.3f' % avg_rpn_iou)
    logger.info('final average cls acc: %.3f' % avg_cls_acc)
    logger.info('final average cls acc refined: %.3f' % avg_cls_acc_refined)
    ret_dict['rpn_iou'] = avg_rpn_iou
    ret_dict['rcnn_cls_acc'] = avg_cls_acc
    ret_dict['rcnn_cls_acc_refined'] = avg_cls_acc_refined
    ret_dict['rcnn_avg_num'] = avg_det_num

    for idx, thresh in enumerate(thresh_list):
        cur_roi_recall = total_roi_recalled_bbox_list[idx] / max(total_gt_bbox, 1.0)
        logger.info('total roi bbox recall(thresh=%.3f): %d / %d = %f' % (thresh, total_roi_recalled_bbox_list[idx],
                                                                          total_gt_bbox, cur_roi_recall))
        ret_dict['rpn_recall(thresh=%.2f)' % thresh] = cur_roi_recall

    for idx, thresh in enumerate(thresh_list):
        cur_recall = total_recalled_bbox_list[idx] / max(total_gt_bbox, 1.0)
        logger.info('total bbox recall(thresh=%.3f): %d / %d = %f' % (thresh, total_recalled_bbox_list[idx],
                                                                      total_gt_bbox, cur_recall))
        ret_dict['rcnn_recall(thresh=%.2f)' % thresh] = cur_recall

    if cfg.TEST.SPLIT != 'test':
        logger.info('Averate Precision:')
        name_to_class = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        ap_result_str, ap_dict = kitti_evaluate(dataset.label_dir, final_output_dir, label_split_file=split_file,
                                                current_class=name_to_class[cfg.CLASSES])
        logger.info(ap_result_str)
        ret_dict.update(ap_dict)

    logger.info('result is saved to: %s' % result_dir)
