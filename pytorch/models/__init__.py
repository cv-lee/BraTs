import os
import sys

import torch
from torch.optim import Adam, SGD

from .unet import *
from .pspnet import *
from .deeplab import *

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import *
import config


def load_model(args, class_num, mode):

    # Device Init
    device = config.device

    # Model Init
    if args.model == 'unet':
        net = UNet(args.in_channel, class_num, drop_rate=args.drop_rate)
    elif args.model == 'pspnet_squeeze':
        net = pspnet_squeeze()
    elif args.model == 'pspnet_res18':
        net = pspnet_res18()
    elif args.model == 'pspnet_res34':
        net = pspnet_res34()
    elif args.model == 'pspnet_res50':
        net = pspnet_res50()
    elif args.model == 'deeplab':
        net = Deeplab_V3_Plus()
    else:
        raise ValueError('args.model ERROR')

    # Optimizer Init
    if mode == 'train':
        resume = args.resume
        #optimizer = Adam(net.parameters(), lr=args.lr)
        optimizer = SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif mode == 'test':
        resume = True
        optimizer = None
    else:
        raise ValueError('load_model mode ERROR')

    # Model Load
    if resume:
        checkpoint = Checkpoint(net, optimizer)
        checkpoint.load(os.path.join(args.ckpt_root, args.model+'.tar'))
        best_score = checkpoint.best_score
        start_epoch = checkpoint.epoch+1
    else:
        best_score = 0
        start_epoch = 1

    if device == 'cuda':
        net.cuda()
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark=True

    return net, optimizer, best_score, start_epoch
