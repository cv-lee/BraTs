import os
import sys

import torch
from torch.optim import Adam, SGD

from .unet import *
from .pspnet import *


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import *
import config


def load_model(args, class_num, mode):

    device = config.device

    if args.model == 'unet':
        model = UNet(class_num)
    elif args.model == 'pspnet':
        #model = PSPNet(sizes=(1,2,3,6), psp_size=2048, deep_features_size=1024,
        #               backend='resnet50')
        model = PSPNet(sizes=(1,2,3,6), psp_size=512, deep_features_size=256,
                       backend='resnet18')
    else:
        raise ValueError('args.model ERROR')

    if mode == 'train':
        resume = args.resume
        #optimizer = Adam(model.parameters(), lr=args.lr)
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif mode == 'test':
        resume = True
        optimizer = None
    else:
        raise ValueError('load_model mode ERROR')

    if resume:
        checkpoint = Checkpoint(model, optimizer)
        checkpoint.load(args.ckpt_path)
        best_score = checkpoint.best_score
        start_epoch = checkpoint.epoch+1
    else:
        best_score = 0
        start_epoch = 1

    if device == 'cuda':
        model.cuda()
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark=True

    return model, optimizer, best_score, start_epoch
