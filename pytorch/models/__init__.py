import os
import sys

import torch
from torch.optim import Adam

from .unet import *

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import *
import config


def load_model(args, class_num, mode):

    device = config.device
    model = UNet(class_num)

    if mode == 'train':
        resume = args.resume
        optimizer = Adam(model.parameters(), lr=args.lr)
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
