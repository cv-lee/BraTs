import pdb
import argparse

from data import *
from unet import *

def train(args):

    # Data Load
    trainset = trainGenerator(args)

    # Model Load
    model = unet()
    model_checkpoint = ModelCheckpoint(args.ckpt_path,
                                       monitor='loss',verbose=1,
                                       save_best_only=True)
    model.fit_generator(trainset, steps_per_epoch=2000,
                        epochs=args.epoch,callbacks=[model_checkpoint])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume model checkpoint')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='batch size.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='starting learning_rate')
    parser.add_argument('--epoch', type=int, default=10,
                        help='number of epochs.')
    parser.add_argument('--data', type=str, default='enhancing',
                        help='MRI Label data to train')
    parser.add_argument('--train_root', type=str,
                        default='../data/train/keras',
                        help='the directory containing the train dataset.')
    parser.add_argument('--image_folder1', type=str,
                        default='flair1',
                        help='the directory containing the trian image dataset.')
    parser.add_argument('--image_folder2', type=str,
                        default='t1c_2',
                        help='the directory containing the trian image dataset.')
    parser.add_argument('--label_folder1', type=str,
                        default='label1',
                        help='the directory containing the train label dataset.')
    parser.add_argument('--label_folder2', type=str,
                        default='label2_t1c',
                        help='the directory containing the train label dataset.')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint/unet.hdf5',
                        help='The directory containing the generative image.')
    args = parser.parse_args()

    train(args)
