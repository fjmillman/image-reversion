import argparse
import os
import numpy as np

import tensorflow as tf

from model import GAN
from utils import check_folder

def parse_args():
    """
    Parse the arguments for configuration
    """
    description = "Tensorflow implementation of GAN for reverting image adjustments"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--dataset_name', dest='dataset_name', default='enhancements', help='Name of the dataset')
    parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='Number of epoch')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='Number of images in a batch')
    parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='Number of images used to train')
    parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='Crop size')
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='Initial learning rate for ADAM')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='Momentum term of adam')
    parser.add_argument('--phase', dest='phase', default='train', help='Training or testing the GAN')
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='Where the models are saved')
    parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='Where the samples are saved')
    parser.add_argument('--test_dir', dest='test_dir', default='./test', help='Where the test samples are saved')
    parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100.0, help='Weight on L1 term in objective function')

    return parser.parse_args()

def main(_):
    args = parse_args()

    check_folder(args.checkpoint_dir)
    check_folder(args.sample_dir)
    check_folder(args.test_dir)

    with tf.Session() as sess:
        model = GAN(
            sess,
            image_size=args.fine_size,
            batch_size=args.batch_size,
            output_size=args.fine_size,
            dataset_name=args.dataset_name,
            checkpoint_dir=args.checkpoint_dir,
            sample_dir=args.sample_dir,
            test_dir=args.test_dir,
            beta1=args.beta1,
            learning_rate=args.lr,
            epoch=args.epoch,
            train_size=args.train_size
        )

        if args.phase == 'train':
            model.train()
        else:
            model.test()

if __name__ == '__main__':
    tf.app.run()
