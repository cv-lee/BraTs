import argparse
import logging
import pdb

import torch
import torch.backends.cudnn as cudnn

from config import *
from dataset import *
from models import *
from utils import *


def train(args):

    # Variables Init
    device = config.device
    cudnn.benchmark = True
    get_logger()

    # Data Load
    trainloader = data_loader(args, mode='train')
    validloader = data_loader(args, mode='valid')

    # Model Load
    model, optimizer, best_score, start_epoch =\
        load_model(args, class_num=config.class_num, mode='train')

    for epoch in range(start_epoch, start_epoch+args.epochs):

        # Train Model
        print('\nEpoch: {}\n<Train>\n'.format(epoch))
        model.train(True)
        loss = 0
        lr = args.lr * (0.5 ** (epoch // 3))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        torch.set_grad_enabled(True)
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
        log_msg = '\n'.join(['Epoch: %d  Loss: %.5f,  Dice-Coef:  %.5f'\
                         %(epoch, loss/(idx+1), 1-(loss/(idx+1)))])
        logging.info(log_msg)

        # Validate Model
        print('\n<Validation>\n')
        model.eval()
        loss = 0
        torch.set_grad_enabled(False)
        for idx, (inputs, targets, paths) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            #outputs = post_process(args, inputs, outputs, save=False)
            batch_loss = dice_coef(outputs, targets, backprop=False)
            loss += float(batch_loss)
            progress_bar(idx, len(validloader), 'Loss: %.5f, Dice-Coef: %.5f'
                         %((loss/(idx+1)), (1-(loss/(idx+1)))))
        log_msg = '\n'.join(['Epoch: %d  Loss: %.5f,  Dice-Coef:  %.5f'
                        %(epoch, loss/(idx+1), 1-(loss/(idx+1)))])
        logging.info(log_msg)

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
    parser.add_argument("--epochs", type=int, default=10,
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
