import argparse

from data import *
from unet import *


def train(args):

    # Data Load
    trainset = dataset(args, mode='train')
    validset = dataset(args, mode='valid')

    # Model Load
    model = unet(args)
    model_checkpoint = ModelCheckpoint(args.ckpt_path, monitor='val_loss',
                                       verbose=2, save_best_only=True)
    # Model Train
    model.fit_generator(trainset, steps_per_epoch=500, shuffle=True, epochs=args.epoch,
                        validation_data=validset, validation_steps=2000,
                        callbacks=[model_checkpoint], workers=16)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='starting learning_rate')
    parser.add_argument('--epoch', type=int, default=20,
                        help='number of epochs.')
    parser.add_argument('--data', type=str, default='complete',
                        help='MRI Label data to train')
    parser.add_argument('--image_root', type=str, default='../data/train/keras',
                        help='the root directory containing the image dataset.')
    parser.add_argument('--image_folder1', type=str, default='flair1',
                        help='the directory containing the image dataset(whole)')
    parser.add_argument('--image_folder2', type=str, default='flair2',
                        help='the directory containing the image dataset(ratio>n%)')
    parser.add_argument('--label_root', type=str, default='../data/train/keras',
                        help='the root directory containing the label dataset.')
    parser.add_argument('--label_folder1', type=str, default='label1',
                        help='the directory containing the train label dataset(whole).')
    parser.add_argument('--label_folder2', type=str, default='label2',
                        help='the directory containing the train label dataset(ratio>n%).')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint/unet.hdf5',
                        help='The directory containing the generative image.')
    args = parser.parse_args()

    train(args)
