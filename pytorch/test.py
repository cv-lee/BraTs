import shutil
import argparse
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torch.optim import SGD
from torchvision import transforms
from torch.utils.data import DataLoader

import config
from dataset import *
from models import *
from utils import *


def test(args):

    # Device Init
    device = config.device
    cudnn.benchmark = True

    # Data Load
    testloader = data_loader(args, mode='test')
    # Model Load
    model, optimizer, best_score, start_epoch = \
        load_model(args, class_num=config.class_num, mode='test')

    # Loss Init
    unet.eval()
    for idx, (inputs, paths) in enumerate(testloader):
        inputs = inputs.to(device)
        outputs = unet(inputs)
        save_img(args, inputs, outputs, paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=bool, default=True,
                        help="Model Trianing resume.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="The batch size to load the data")
    parser.add_argument("--data", type=str, default="complete",
                        help="Label data type.")
    parser.add_argument("--img_root", type=str, default="../data/ImagesTr",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--output_root", type=str, default="./output/prediction",
                        help="The directory containing the results.")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoint/unet.tar",
                        help="The directory containing the training label datgaset")
    args = parser.parse_args()

    test(args)
