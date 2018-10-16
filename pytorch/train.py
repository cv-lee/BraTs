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

    # Variables and logger Init
    device = config.device
    cudnn.benchmark = True
    get_logger()

    # Data Load
    trainloader = data_loader(args, mode='train')
    validloader = data_loader(args, mode='valid')

    # Model Load
    net, optimizer, best_score, start_epoch =\
        load_model(args, class_num=config.class_num, mode='train')
    log_msg = '\n'.join(['%s Train Start'%(args.model)])
    logging.info(log_msg)

    for epoch in range(start_epoch, start_epoch+args.epochs):

        # Train Model
        print('\n\n\nEpoch: {}\n<Train>'.format(epoch))
        net.train(True)
        loss = 0
        lr = args.lr * (0.5 ** (epoch // 4))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        torch.set_grad_enabled(True)
        for idx, (inputs, targets, paths) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            if type(outputs) == tuple:
                outputs = outputs[0]
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
        print('\n\n<Validation>')
        net.eval()
        for module in net.module.modules():
            if isinstance(module, torch.nn.modules.Dropout2d):
                module.train(True)
            elif isinstance(module, torch.nn.modules.Dropout):
                module.train(True)
            else:
                pass
        loss = 0
        torch.set_grad_enabled(False)
        for idx, (inputs, targets, paths) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            if type(outputs) == tuple:
                outputs = outputs[0]
            #outputs = post_process(args, inputs, outputs, save=False)
            batch_loss = dice_coef(outputs, targets, backprop=False)
            loss += float(batch_loss)
            progress_bar(idx, len(validloader), 'Loss: %.5f, Dice-Coef: %.5f'
                         %((loss/(idx+1)), (1-(loss/(idx+1)))))
        log_msg = '\n'.join(['Epoch: %d  Loss: %.5f,  Dice-Coef:  %.5f'
                        %(epoch, loss/(idx+1), 1-(loss/(idx+1)))])
        logging.info(log_msg)

        # Save Model
        loss /= (idx+1)
        score = 1 - loss
        if score > best_score:
            checkpoint = Checkpoint(net, optimizer, epoch, score)
            checkpoint.save(os.path.join(args.ckpt_root, args.model+'.tar'))
            best_score = score
            print("Saving...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=bool, default=False,
                        help="Model Trianing resume.")
    parser.add_argument("--model", type=str, default='pspnet_res50',
                        help="Model Name (unet, pspnet_squeeze, pspnet_res50,\
                        pspnet_res34, pspnet_res50, deeplab)")
    parser.add_argument("--in_channel", type=int, default=1,
                        help="A number of images to use for input")
    parser.add_argument("--batch_size", type=int, default=80,
                        help="The batch size to load the data")
    parser.add_argument("--epochs", type=int, default=30,
                        help="The training epochs to run.")
    parser.add_argument("--drop_rate", type=float, default=0.1,
                        help="Drop-out Rate")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate to use in training")
    parser.add_argument("--data", type=str, default="complete",
                        help="Label data type.")
    parser.add_argument("--img_root", type=str, default="../../data/train/image_FLAIR",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--label_root", type=str, default="../../data/train/label",
                        help="The directory containing the training label datgaset")
    parser.add_argument("--output_root", type=str, default="./output/prediction",
                        help="The directory containing the result predictions")
    parser.add_argument("--ckpt_root", type=str, default="./checkpoint",
                        help="The directory containing the checkpoint files")
    args = parser.parse_args()

    train(args)
