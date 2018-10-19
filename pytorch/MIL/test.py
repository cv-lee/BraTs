import pdb
import argparse
import torch
import torch.backends.cudnn as cudnn

from config import *
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
    net, _, _, _ = load_model(args, mode='test')

    # Loss Init
    net.eval()
    torch.set_grad_enabled(False)
    aleatoric_total = 0
    epistemic_total = 0

    for idx, (input1, input2, input3, input4, path) in enumerate(testloader):
        batch_size = input1.shape[0]
        input1, input2, input3, input4 =\
            input1.to(device), input2.to(device), input3.to(device), input4.to(device)
        for i in range(args.iter_num):
            output = net(input1, input2, input3, input4)
            output = output.cpu().detach().numpy()
            output = np.expand_dims(output,axis=0)
            if i == 0:
                outputs = output
            else:
                outputs = np.concatenate((outputs, output), axis=0)

        img = input1.cpu().detach().numpy()
        prediction = (np.mean(outputs, axis=0)).argmax(axis=1)
        aleatoric = np.mean(outputs*(1-outputs), axis=0)[:,1,:,:]
        epistemic = (np.mean(outputs**2, axis=0) - np.mean(outputs, axis=0)**2)[:,1,:,:]
        post_process(args, img, prediction, aleatoric, epistemic, path,
                     crf=False, erode=False)
        print(idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default='pspnet_res34', # Need to be fixed
                        help="Model Name")
    parser.add_argument("--iter_num", type=int, default=30, # Need to be fixed
                        help="The number of iteration for uncertainty model")
    parser.add_argument("--batch_size", type=int, default=155, # Need to be fixed
                        help="The batch size to load the data")
    parser.add_argument("--data", type=str, default="complete",
                        help="Label data type.")
    parser.add_argument("--img_root1", type=str, default="../../data/train/image_FLAIR",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--img_root2", type=str, default="../../data/train/image_T1",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--img_root3", type=str, default="../../data/train/image_T1C",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--img_root4", type=str, default="../../data/train/image_T2",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--output_root", type=str, default="./output/prediction",
                        help="The directory containing the results.")
    parser.add_argument("--ckpt_root", type=str, default="./checkpoint",
                        help="The directory containing the trained model checkpoint")
    args = parser.parse_args()

    test(args)
