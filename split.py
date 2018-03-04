"""
Adapted from github.com/affinelayer/pix2pix-tensorflow/blob/master/tools/split.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import argparse
import glob
import os


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", type=str, required=True, help="path to folder containing images")
    parser.add_argument("--train_frac", type=float, default=0.75, help="percentage of images to use for training set")
    parser.add_argument("--test_frac", type=float, default=0.25, help="percentage of images to use for test set")
    parser.add_argument("--sort", action="store_true", help="if set, sort the images instead of shuffling them")

    return parser.parse_args()


def main():
    random.seed(0)

    args = parse_arguments()

    files = glob.glob(os.path.join(args.dir, "*.png"))
    files.sort()

    assignments = []
    assignments.extend(["train"] * int(args.train_frac * len(files)))
    assignments.extend(["test"] * int(args.test_frac * len(files)))
    assignments.extend(["val"] * int(len(files) - len(assignments)))

    if not args.sort:
        random.shuffle(assignments)

    for name in ["train", "val", "test"]:
        if name in assignments:
            d = os.path.join(args.dir, name)
            if not os.path.exists(d):
                os.makedirs(d)

    print(len(files), len(assignments))

    for inpath, assignment in zip(files, assignments):
        outpath = os.path.join(args.dir, assignment, os.path.basename(inpath))
        print(inpath, "->", outpath)
        os.rename(inpath, outpath)


if __name__ == '__main__':
    main()
