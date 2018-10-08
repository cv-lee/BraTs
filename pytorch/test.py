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
    model, _, _, _ = load_model(args, class_num=config.class_num, mode='test')

    # Loss Init
    model.eval()
    torch.set_grad_enabled(False)
    for idx, (inputs, paths) in enumerate(testloader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        post_process(args, inputs, outputs, paths)
        print(idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_size", type=int, default=155, # Need to be fixed
                        help="The batch size to load the data")
    parser.add_argument("--data", type=str, default="complete",
                        help="Label data type.")
    parser.add_argument("--img_root", type=str, default="../data/train/image_FLAIR",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--output_root", type=str, default="./output/prediction",
                        help="The directory containing the results.")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoint/model.tar",
                        help="The directory containing the training label datgaset")
    args = parser.parse_args()

    test(args)
