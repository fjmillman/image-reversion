from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model import GAN
from utils import *

import argparse
import tensorflow as tf

flags = tf.flags

flags.DEFINE_integer('batch_size', 1, 'The number of images in each batch.')
flags.DEFINE_integer('max_epochs', 200, 'The number of training epochs.')

flags.DEFINE_integer('ngf', 64, 'The number of generator filters in the first convolution layer.')
flags.DEFINE_integer('ndf', 64, 'The number of discriminator filters in the first convolution layer.')

flags.DEFINE_float('lr', 0.0002, 'The initial learning rate for ADAM.')
flags.DEFINE_float('beta1', 0.5, 'The momentum term of ADAM.')
flags.DEFINE_float('l1_weight', 100.0, 'The weight on the L1 term for the generator gradient.')
flags.DEFINE_float('gan_weight', 1.0, 'The weight on the GAN term for the generator gradient.')

flags.DEFINE_integer('progress_freq', 50, 'The number of steps to take before displaying progress.')
flags.DEFINE_integer('save_freq', 5000, 'The number of steps to take before saving the model.')

FLAGS = flags.FLAGS


def parse_args():
    """
    Parse the arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True, help="directory containing the input images")
    parser.add_argument("--mode", required=True, choices=["train", "test"], help="operation that will be used")
    parser.add_argument("--output_dir", required=True, help="directory where the output images will be saved")
    parser.add_argument("--checkpoint", default=None, help="checkpoint to resume training from or use for testing")

    return parser.parse_args()


def main():
    # Parse the arguments from the command line
    args = parse_args()

    # Create output directory if it does not exist
    check_folder(args.output_dir)

    # Ensure checkpoint exists before testing
    if args.mode == "test" and args.checkpoint is None:
        raise Exception("Checkpoint is required for test mode")

    # Load the images from the input directory
    paths, inputs, targets, steps_per_epoch = load_images(args.input_dir, FLAGS.batch_size)

    # Initialise the GAN before running
    model = GAN(args.input_dir, args.output_dir, args.checkpoint, paths, inputs, targets, FLAGS.batch_size,
                steps_per_epoch, FLAGS.ngf, FLAGS.ndf, FLAGS.lr, FLAGS.beta1, FLAGS.l1_weight, FLAGS.gan_weight)

    sv = tf.train.Supervisor(logdir=None, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        # Train or test the initialised GAN based on the chosen mode
        if args.mode == "train":
            model.train(sv, sess, FLAGS.max_epochs, FLAGS.progress_freq, FLAGS.save_freq)
        else:
            model.test(sess)


if __name__ == '__main__':
    main()
