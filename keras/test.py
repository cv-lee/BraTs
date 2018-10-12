import argparse

from data import *
from unet import *


def test(args):

    # Data Load
    testset = dataset(args, mode='test')

    # Model Load
    model = unet(args)
    model.load_weights(args.ckpt_path)

    # Model Test
    results = model.predict_generator(testset, steps=1, verbose=1)

    # Save predictions
    save_result(args, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch_size', type=int, default=155,
                        help='batch size.')
    parser.add_argument('--data', type=str, default='complete',
                        help='MRI Label data to train')
    parser.add_argument('--image_root', type=str, default='../data/train/image_FLAIR',
                        help='the root directory containing the image dataset.')
    parser.add_argument('--label_root', type=str, default='../data/train/label',
                        help='the root directory containing the label dataset')
    parser.add_argument('--image_folder1', type=str, default='BRATS_074',
                        help='the directory containing the image dataset.')
    parser.add_argument('--label_folder1', type=str, default='BRATS_074',
                        help='the directory containing the label dataset.')
    parser.add_argument('--output_root', type=str, default='./output',
                        help='the directory to save results')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint/unet.hdf5',
                        help='The directory containing the segmentation model checkpoint.')
    args = parser.parse_args()

    test(args)
