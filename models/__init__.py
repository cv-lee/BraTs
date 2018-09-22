import os
import sys

import torch
from torch.optim import Adam

from .unet import *
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import *


def load_model(args, class_num, mode):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(class_num)
    if mode == 'TRAIN':
        optimizer = Adam(model.parameters(), lr=args.lr)
    elif mode == 'TEST':
        optimizer = None
    else:
        raise ValueError('InValid Flag in load_model')

    if args.resume:
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

