from __future__ import division
from skimage import io
from skimage.measure import compare_mse as mse, compare_ssim as ssim

import os
import glob
import argparse


def parse_arguments():
    """
    Parse the arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir_a", type=str, required=True, help="path to folder containing first set of images")
    parser.add_argument("--image_dir_b", type=str, required=True, help="path to folder containing second set of images")
    parser.add_argument("--output_dir", type=str, required=True, help="path to folder to write the results")

    return parser.parse_args()


def read_image(image_path):
    """
    Read the image from the given image path and transform to return a numpy array
    """
    return io.imread(image_path) / 127.5 - 1.


def get_images(image_paths):
    """
    Get the paired images from the given image paths
    """
    images = []

    for (image_path_a, image_path_b) in image_paths:
        image_a = read_image(image_path_a)
        image_b = read_image(image_path_b)
        images.append((image_a, image_b))

    return images


def run_metrics(images):
    """
    Test the given collection of image pairs with a set of measures and return the results
    """
    mse_results = list()
    ssim_results = list()

    for (image_a, image_b) in images:
        mse_results.append(mse(image_a, image_b))
        ssim_results.append(ssim(image_a, image_b, multichannel=True, data_range=image_a.max() - image_b.min()))

    return list(zip(mse_results, ssim_results))


def write_metrics(metrics, output_dir):
    """
    Write the metric results to a text file
    """
    file = open(output_dir + '/metrics.txt', 'w')

    for (mse_result, ssim_result) in metrics:
        file.write("MSE: {:.6f} - SSIM: {:.6f}\n".format(mse_result, ssim_result))


def main():
    args = parse_arguments()

    if args.image_dir_a is None or not os.path.exists(args.image_dir_a):
        raise Exception("image_dir_a does not exist")

    if args.image_dir_b is None or not os.path.exists(args.image_dir_b):
        raise Exception("image_dir_b does not exist")

    image_paths_a = sorted(glob.glob(os.path.join(args.image_dir_a, "*.png")))

    if len(image_paths_a) == 0:
        raise Exception("image_dir_a contains no image files")

    image_paths_b = sorted(glob.glob(os.path.join(args.image_dir_b, "*.png")))

    if len(image_paths_b) != len(image_paths_a):
        raise Exception("image_dir_b must contain the same number of image files as image_dir_a")

    image_paths = list(zip(image_paths_a, image_paths_b))

    images = get_images(image_paths)
    metrics = run_metrics(images)
    write_metrics(metrics, args.output_dir)


if __name__ == '__main__':
    main()
