import pdb
import argparse
import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import backend as K

from data import *
from unet import *

def test(args):

    # Data Load
    testset = testGenerator(args)

    # Model Load
    model = unet(args)
    model.load_weights(args.ckpt_path)
    results = model.predict_generator(testset, 1, verbose=1)
    save_img(args, results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume model checkpoint')
    parser.add_argument('--batch_size', type=int, default=155,
                        help='batch size.')
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='output threshold')
    parser.add_argument('--data', type=str, default='complete',
                        help='MRI Label data to train')
    parser.add_argument('--test_img_root', type=str,
                        default='../data/train/image_FLAIR',
                        help='the directory containing the train dataset.')
    parser.add_argument('--test_label_root', type=str,
                        default='../data/train/label')
    parser.add_argument('--image_folder', type=str,
                        default='BRATS_387',
                        help='the directory containing the trian image dataset.')
    parser.add_argument('--label_folder', type=str,
                        default='BRATS_387',
                        help='the directory containing the train label dataset.')
    parser.add_argument('--output_root', type=str,
                        default='./output')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint/unet.hdf5',
                        help='The directory containing the generative image.')
    args = parser.parse_args()

    test(args)
