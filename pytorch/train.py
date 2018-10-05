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


def train(args):

    # Device Init
    device = config.device
    cudnn.benchmark = True

    # Data Load
    trainloader = data_loader(args, mode='train')
    validloader = data_loader(args, mode='valid')

    # Model Load
    model, optimizer, best_score, start_epoch =\
        load_model(args, class_num=config.class_num, mode='train')

    for epoch in range(start_epoch, args.epochs + 1):
        # Train Model
        print('\nEpoch: {}\n<Train>\n'.format(epoch))
        model.train(True)
        loss = 0
        lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        for idx, (inputs, targets, paths) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            batch_loss = dice_coef(outputs, targets)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += float(batch_loss)
            progress_bar(idx, len(trainloader), 'Loss: %.5f, Dice-Coef: %.5f'
                         %((loss/(idx+1)), (1-(loss/(idx+1)))))

        # Validate Model
        print('\n<Validation>\n')
        model.eval()
        loss = 0
        for idx, (inputs, targets, paths) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            batch_loss = dice_coef(outputs, targets)
            loss += float(batch_loss)
            progress_bar(idx, len(trainloader), 'Loss: %.5f, Dice-Coef: %.5f'
                         %((loss/(idx+1)), (1-(loss/(idx+1)))))
        loss /= (idx+1)
        score = 1 - loss
        if score > best_score:
            checkpoint = Checkpoint(model, optimizer, epoch, score)
            checkpoint.save(args.ckpt_path)
            best_score = score
            print("Saving...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=bool, default=False,
                        help="Model Trianing resume.")
    parser.add_argument("--batch_size", type=int, default=52,
                        help="The batch size to load the data")
    parser.add_argument("--epochs", type=int, default=150,
                        help="The training epochs to run.")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate to use in training")
    parser.add_argument("--data", type=str, default="complete",
                        help="Label data type.")
    parser.add_argument("--img_root", type=str, default="../data/train/image_FLAIR",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--label_root", type=str, default="../data/train/label",
                        help="The directory containing the training label datgaset")
    parser.add_argument("--output_root", type=str, default="./output/prediction",
                        help="The directory containing the result predictions")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoint/model.tar",
                        help="The directory containing the training label datgaset")
    args = parser.parse_args()

    train(args)
