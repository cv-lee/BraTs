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

from dataset import *
from models import *
from utils import *


def train(args):

    # Device Init
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn.benchmark = True

    # Data Load
    trainloader = data_loader(args, mode='TRAIN')
    testloader = data_loader(args, mode='TEST')
    # Model Load
    unet, optimizer, best_score, start_epoch = load_model(args, class_num=2, mode='TRAIN')

    # Loss Init
    for epoch in range(start_epoch, args.epochs + 1):
        # Train Model
        print('\nEpoch: {}\n'.format(epoch))
        unet.train(True)
        loss = 0
        lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        for idx, (input, target, input_path) in enumerate(trainloader):
            input, target = input.to(device), target.to(device)
            #target = (target.type(torch.cuda.LongTensor)).max(1)[1]
            #weights = torch.FloatTensor([0.2,1.0]).cuda()
            output = unet(input)
            batch_loss = dice_coef(output, target)
            #loss_ = F.cross_entropy(output, target, weight=weights)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += float(batch_loss)
            progress_bar(idx, len(trainloader), 'Loss: %.5f, Dice-Coef: %.5f'
                         %((loss/(idx+1)), (1-(loss/(idx+1)))))
        loss /= (idx+1)
        score = 1 - loss
        if score > best_score:
            checkpoint = Checkpoint(unet, optimizer, epoch, score)
            checkpoint.save(args.ckpt_path)
            best_score = score
            print("Saving...")

        unet.eval()
        for idx, (input, input_path) in enumerate(testloader):
            if idx == 250:
                break
            input = input.to(device)
            output = unet(input)
            save_img(args, input, output, input_path)

        '''
        # Validate Model
        unet.eval()
        loss = 0
        for idx, (input, target, input_path) in enumerate(trainloader):
            input, target = input.to(device), target.to(device)
            output = unet(input)
            loss += float(dice_coef(output, target))
            progress_bar(idx, len(trainloader), 'Loss: %.5f, Dice-Coef: %.5f'
                         %((loss/(idx+1)), (1-(loss/(idx+1)))))


        score = loss/(idx+1)
        if score > best_score:
            checkpoint = Checkpoint(unet, optimizer, epoch, score)
            checkpoint.save(args.ckpt_path)
            best_score = score
            print("Saving...")
        '''

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
    parser.add_argument("--data", type=str, default="ENHANCING",
                        help="Label data type.")
    parser.add_argument("--img_root", type=str, default="../data/train/image_T1C",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--label_root", type=str, default="../data/train/label",
                        help="The directory containing the training label datgaset")
    parser.add_argument("--output_root", type=str, default="./output/prediction",
                        help="The directory containing the result predictions")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoint/unet.tar",
                        help="The directory containing the training label datgaset")
    args = parser.parse_args()

    train(args)
