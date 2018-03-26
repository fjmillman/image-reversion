"""
Adapted from github.com/affinelayer/pix2pix-tensorflow/blob/master/tools/process.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import check_folder

import os
import time
import argparse
import numpy as np
import tfimage as im
import tensorflow as tf


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True, help="path to folder containing images")
    parser.add_argument("--output_dir", required=True, help="output path")
    parser.add_argument("--operation", required=True, choices=["resize", "combine"])
    parser.add_argument("--pad", action="store_true", help="pad instead of crop for resize operation")
    parser.add_argument("--size", type=int, default=256, help="size to use for resize operation")
    parser.add_argument("--b_dir", type=str, help="path to folder containing B images for combine operation")

    return parser.parse_args()


def resize(src, pad, new_size):
    height, width, _ = src.shape
    dst = src
    if height != width:
        if pad:
            size = max(height, width)
            # Pad to correct ratio
            oh = (size - height) // 2
            ow = (size - width) // 2
            dst = im.pad(image=dst, offset_height=oh, offset_width=ow, target_height=size, target_width=size)
        else:
            # Crop to correct ratio
            size = min(height, width)
            oh = (height - size) // 2
            ow = (width - size) // 2
            dst = im.crop(image=dst, offset_height=oh, offset_width=ow, target_height=size, target_width=size)

    assert(dst.shape[0] == dst.shape[1])

    size, _, _ = dst.shape
    if size > new_size:
        dst = im.downscale(images=dst, size=[new_size, new_size])
    elif size < new_size:
        dst = im.upscale(images=dst, size=[new_size, new_size])

    return dst


def combine(src, src_path, b_dir):
    if b_dir is None:
        raise Exception("missing b_dir")

    # Find corresponding file in b_dir, could have a different extension
    basename, _ = os.path.splitext(os.path.basename(src_path))
    for ext in [".png", ".jpg"]:
        sibling_path = os.path.join(b_dir, basename + ext)
        if os.path.exists(sibling_path):
            sibling = im.load(sibling_path)
            break
    else:
        raise Exception("could not find sibling image for " + src_path)

    # Make sure that dimensions are correct
    height, width, _ = src.shape
    if height != sibling.shape[0] or width != sibling.shape[1]:
        raise Exception("differing sizes")

    # Convert both images to RGB if necessary
    if src.shape[2] == 1:
        src = im.grayscale_to_rgb(images=src)

    if sibling.shape[2] == 1:
        sibling = im.grayscale_to_rgb(images=sibling)

    # Remove alpha channel
    if src.shape[2] == 4:
        src = src[:,:,:3]

    if sibling.shape[2] == 4:
        sibling = sibling[:,:,:3]

    return np.concatenate([src, sibling], axis=1)


def process(src_path, dst_path, operation, pad, size, b_dir):
    src = im.load(src_path)

    if operation == "resize":
        dst = resize(src, pad, size)
    elif operation == "combine":
        dst = combine(src, src_path, b_dir)
    else:
        raise Exception("invalid operation")

    im.save(dst, dst_path)


start = None
num_complete = 0
total = 0


def complete():
    global num_complete, rate, last_complete

    num_complete += 1
    now = time.time()
    elapsed = now - start
    rate = num_complete / elapsed
    if rate > 0:
        remaining = (total - num_complete) / rate
    else:
        remaining = 0

    print(f"Progress: {num_complete}/{total} complete - Rate: {rate:0.2f} images/sec - Time:"
          f"{elapsed // 60}m{elapsed % 60}s elapsed and {remaining // 60}m{remaining % 60}s remaining")

    last_complete = now


def main():
    args = parse_arguments()

    check_folder(args.output_dir)

    src_paths = []
    dst_paths = []

    skipped = 0
    for src_path in im.find(args.input_dir):
        name, _ = os.path.splitext(os.path.basename(src_path))
        dst_path = os.path.join(args.output_dir, name + ".png")
        if os.path.exists(dst_path):
            skipped += 1
        else:
            src_paths.append(src_path)
            dst_paths.append(dst_path)

    print(f"skipping {skipped} files that already exist")

    global total
    total = len(src_paths)

    print(f"processing {total} files")

    global start
    start = time.time()

    with tf.Session() as sess:
        for src_path, dst_path in zip(src_paths, dst_paths):
            process(src_path, dst_path, args.operation, args.pad, args.size, args.b_dir)
            complete()


if __name__ == '__main__':
    main()
