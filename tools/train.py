import os
from functools import partial

import _init_path
import argparse

import logging

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from datasets.kitti_vpnet_dataset import Kitti_VPNet_Dataset
from models.MainNet import MainNet
from models.PointRCNN.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
from models.PointRCNN.net import train_functions
from utils.train_utils.fastai_optim import OptimWrapper
from utils.train_utils import learning_schedules_fastai as lsf
import utils.train_utils.train_utils as train_utils

parser = argparse.ArgumentParser(description='arg parser')

# ckpt
parser.add_argument('--ckpt', type=str, default=None)

# train args
parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
parser.add_argument("--ckpt_save_interval", type=int, default=5, help="number of training epochs")
parser.add_argument("--eval_frequency", type=int, default=1, help="number of training epochs")
parser.add_argument('--train_with_eval', action='store_true', default=False, help='whether to train with evaluation')

# model args
parser.add_argument('--npoints', type=int, default=16384, help='sampled to the number of points')
parser.add_argument('--crop_height', type=int, default=384, help="crop height")
parser.add_argument('--crop_width', type=int, default=1248, help="crop width")
parser.add_argument('--max_disp', type=int, default=192, help="max disp")

# super args
parser.add_argument('--lr', type=float, default=0.002, help='Learning Rate. Default=0.001')
parser.add_argument("--batch_size", type=int, default=1, help="batch size for training")
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train for")

parser.add_argument('--cuda', type=int, default=1, help='use cuda? Default=True')
parser.add_argument('--mgpus', action='store_true', default=False, help='whether to use multiple gpu')
parser.add_argument('--seed', type=int, default=158, help='random seed to use. Default=158')

# mode
parser.add_argument('--mode', default="TRAIN", type=str, help='select mode in "TRAIN", "VAL", "TRAINVAL" or "TEST"')
parser.add_argument("--train_mode", type=str, default='rpn', required=True, help="specify the training mode")

# root path
parser.add_argument('--cfg_file', default='cfgs/default.yaml', type=str, help='specify the config for training')
parser.add_argument('--data_root', default="../data/KITTI/object/", type=str, help="data root")
parser.add_argument('--output_dir', default=None, type=str, help="location to save result")
parser.add_argument('--list_root', default="../data/KITTI/ImageSets/", type=str, help="training list")
parser.add_argument('--ckpt_root', default="../checkpoints/", type=str, help="ckpt root")
parser.add_argument('--log_root', default="../log/", type=str, help="log file")
parser.add_argument("--gt_database", type=str,
                    default='../data/KITTI/object/training/gt_database/train_gt_database_3level_Car.pkl',
                    help='generated gt database for augmentation')

# parser.add_argument('--shift', type=int, default=0, help='random shift of left image. Default=0')

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


def create_dataloader(logger):
    train_dataset = Kitti_VPNet_Dataset(root_dir=args.data_root,
                                        list_root=args.list_root,
                                        mode='TRAIN',
                                        crop_size=[args.crop_height, args.crop_width],
                                        npoints=args.npoints,
                                        classes='Car',
                                        logger=logger
                                        )
    train_data_loader = DataLoader(dataset=train_dataset,
                                   num_workers=args.workers,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   collate_fn=train_dataset.collate_batch
                                   )
    if args.train_with_eval:
        val_dataset = Kitti_VPNet_Dataset(root_dir=args.data_root,
                                          list_root=args.list_root,
                                          mode='VAL',
                                          npoints=args.npoints,
                                          classes='Car',
                                          logger=logger
                                          )
        val_data_loader = DataLoader(dataset=val_dataset,
                                     num_workers=args.workers,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     collate_fn=val_dataset.collate_batch
                                     )
    else:
        val_data_loader = None

    return train_data_loader, val_data_loader


def create_optimizer(model):
    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                              momentum=cfg.TRAIN.MOMENTUM)
    elif cfg.TRAIN.OPTIMIZER == 'adam_onecycle':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=cfg.TRAIN.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )

        if cfg.RPN.ENABLED and cfg.RPN.FIXED:
            for param in model.rpn.parameters():
                param.requires_grad = False
    else:
        raise NotImplementedError

    return optimizer


def create_scheduler(optimizer, total_steps, last_epoch):
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg.TRAIN.DECAY_STEP_LIST:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg.TRAIN.LR_DECAY
        return max(cur_decay, cfg.TRAIN.LR_CLIP / cfg.TRAIN.LR)

    def bnm_lmbd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg.TRAIN.BN_DECAY_STEP_LIST:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg.TRAIN.BN_DECAY
        return max(cfg.TRAIN.BN_MOMENTUM * cur_decay, cfg.TRAIN.BNM_CLIP)

    if cfg.TRAIN.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = lsf.OneCycle(
            optimizer, total_steps, cfg.TRAIN.LR, list(cfg.TRAIN.MOMS), cfg.TRAIN.DIV_FACTOR, cfg.TRAIN.PCT_START
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

    bnm_scheduler = train_utils.BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    return lr_scheduler, bnm_scheduler


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 400:
        lr = 0.002
    else:
        lr = 0.002 * 0.1
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]

    if args.train_mode == 'ganet':
        cfg.GANET.ENABLED = True
        cfg.RPN.ENABLED = False
        cfg.RCNN.ENABLED = False
        root_result_dir = os.path.join('../', 'outputs', 'ganet', cfg.TAG)
    elif args.train_mode == 'rpn':
        cfg.GANET.ENABLED = cfg.GANET.FIXED = True
        cfg.RPN.ENABLED = True
        cfg.RCNN.ENABLED = False
        root_result_dir = os.path.join('../', 'outputs', 'rpn', cfg.TAG)
    elif args.train_mode == 'rcnn':
        cfg.GANET.ENABLED = False
        cfg.RPN.ENABLED = True
        cfg.RCNN.ENABLED = True
        root_result_dir = os.path.join('../', 'outputs', 'rcnn', cfg.TAG)
    elif args.train_mode == 'rcnn_offline':
        cfg.GANET.ENABLED = False
        cfg.RPN.ENABLED = False
        cfg.RCNN.ENABLED = True
        root_result_dir = os.path.join('../', 'outputs', 'rcnn', cfg.TAG)
    elif args.train_mode == 'all':
        cfg.GANET.ENABLED = True
        cfg.RPN.ENABLED = True
        cfg.RCNN.ENABLED = True
        root_result_dir = os.path.join('../', 'outputs', 'all', cfg.TAG)
    else:
        raise NotImplementedError

    if args.output_dir is not None:
        root_result_dir = args.output_dir
    os.makedirs(root_result_dir, exist_ok=True)

    log_file = os.path.join(root_result_dir, 'log_train.txt')
    logger = create_logger(log_file)
    logger.info('**********************Start logging**********************')

    # log to file
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))

    save_config_to_file(cfg, logger=logger)

    # copy important files to backup
    backup_dir = os.path.join(root_result_dir, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp *.py %s/' % backup_dir)
    os.system('cp ../models/PointRCNN/net/*.py %s/' % backup_dir)
    os.system('cp ../datasets/kitti_vpnet_dataset.py %s/' % backup_dir)

    # tensorboard log
    tb_log = SummaryWriter(log_dir=os.path.join(root_result_dir, 'tensorboard'))

    # create dataloader & network & optimizer
    train_loader, val_loader = create_dataloader(logger)
    model = MainNet(num_classes=train_loader.dataset.num_class, use_xyz=True, mode='TRAIN')

    optimizer = create_optimizer(model)
    # optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999))

    # whether use cuda
    cuda = args.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    if cuda:
        model = model.cuda()
        if args.mgpus:
            model = torch.nn.DataParallel(model).cuda()

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1

    print("==> Loading checkpoint...")
    total_keys = model.state_dict().keys().__len__()
    pure_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    if args.ckpt is not None:
        ckpt_path = os.path.join(args.ckpt_root, args.ckpt)
        it, start_epoch = train_utils.load_checkpoint(pure_model, ckpt_path, optimizer, logger)
        last_epoch = start_epoch + 1

    # if args.ganet_ckpt is not None:
    #     ganet_ckpt_path = os.path.join(args.ckpt_root, "GANet", args.ganet_ckpt)
    #     train_utils.load_part_ckpt(pure_model, ganet_ckpt_path, logger, total_keys)
    #
    # if args.pointrcnn_ckpt is not None:
    #     pointrcnn_ckpt_path = os.path.join(args.ckpt_root, "pointrcnn", "all", args.pointrcnn_ckpt)
    #     train_utils.load_part_ckpt(pure_model, pointrcnn_ckpt_path, logger, total_keys)

    lr_scheduler, bnm_scheduler = create_scheduler(optimizer,
                                                   total_steps=len(train_loader) * args.epochs,
                                                   last_epoch=last_epoch)

    if cfg.TRAIN.LR_WARMUP and cfg.TRAIN.OPTIMIZER != 'adam_onecycle':
        lr_warmup_scheduler = train_utils.CosineWarmupLR(optimizer,
                                                         T_max=cfg.TRAIN.WARMUP_EPOCH * len(train_loader),
                                                         eta_min=cfg.TRAIN.WARMUP_MIN)
    else:
        lr_warmup_scheduler = None

    # start training
    logger.info('**********************Start training**********************')
    ckpt_dir = os.path.join(root_result_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    trainer = train_utils.Trainer(
        model,
        train_functions.model_joint_fn_decorator(),
        optimizer,
        ckpt_dir=ckpt_dir,
        lr_scheduler=lr_scheduler,
        bnm_scheduler=bnm_scheduler,
        model_fn_eval=train_functions.model_joint_fn_decorator(),
        tb_log=tb_log,
        eval_frequency=args.eval_frequency,
        lr_warmup_scheduler=lr_warmup_scheduler,
        warmup_epoch=cfg.TRAIN.WARMUP_EPOCH,
        grad_norm_clip=cfg.TRAIN.GRAD_NORM_CLIP
    )

    trainer.train(
        it,
        start_epoch,
        args.epochs,
        train_loader,
        val_loader,
        ckpt_save_interval=args.ckpt_save_interval,
        lr_scheduler_each_iter=(cfg.TRAIN.OPTIMIZER == 'adam_onecycle')
    )

    logger.info('**********************End training**********************')
