from __future__ import division
from skimage import io
from skimage.measure import compare_mse as mse, compare_ssim as ssim

import os
import glob
import pylab
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_arguments():
    """
    Parse the arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir_a", type=str, required=True, help="path to folder containing first set of images")
    parser.add_argument("--image_dir_b", type=str, required=True, help="path to folder containing second set of images")
    parser.add_argument("--output_dir", type=str, required=True, help="path to folder to write the results")
    parser.add_argument("--operation", type=str, required=True, choices=["metrics", "heatmaps"], help="run metrics or generate heatmaps")

    return parser.parse_args()


def check_folder(path_dir):
    """
    Checks if directory exists and creates one if not
    """
    if not os.path.isdir(path_dir):
        os.makedirs(path_dir)


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

    for (image_a, (image_b)) in images:
        mse_results.append(mse(image_a, image_b))
        ssim_results.append(ssim(image_a, image_b, multichannel=True, data_range=image_a.max() - image_b.min()))

    return list(zip(mse_results, ssim_results))


def write_metrics(metrics, output_dir):
    """
    Write the metric results to a text file
    """
    file = open(output_dir + '/metrics.txt', 'w')

    mse_average = 0
    ssim_average = 0
    for i, (mse_result, ssim_result) in enumerate(metrics):
        mse_average += mse_result
        ssim_average += ssim_result

    mse_average = mse_average / len(metrics)
    file.write(f"MSE Average: {mse_average:.5f}\n")

    ssim_average = ssim_average / len(metrics)
    file.write(f"SSIM Average: {ssim_average:.5f}\n")


def generate_heatmaps(images, output_dir):
    """
    Generate heatmaps for each image pair to show difference
    """
    for ((image_a_name, image_a), (image_b_name, image_b)) in images:
        error_r = np.fabs(np.subtract(image_b[:, :, 0], image_a[:, :, 0]))
        error_g = np.fabs(np.subtract(image_b[:, :, 1], image_a[:, :, 1]))
        error_b = np.fabs(np.subtract(image_b[:, :, 2], image_a[:, :, 2]))

        image = np.maximum(np.maximum(error_r, error_g), error_b)
        image_plot = plt.imshow(image)
        image_plot.set_cmap('binary')
        plt.axis('off')

        filename = f"{image_b_name}-heatmap.png"
        out_path = os.path.join(output_dir, filename)
        pylab.savefig(out_path, bbox_inches='tight', pad_inches=0)


def main():
    args = parse_arguments()

    if args.image_dir_a is None or not os.path.exists(args.image_dir_a):
        raise Exception("image_dir_a does not exist")

    if args.image_dir_b is None or not os.path.exists(args.image_dir_b):
        raise Exception("image_dir_b does not exist")

    check_folder(args.output_dir)

    image_paths_a = sorted(glob.glob(os.path.join(args.image_dir_a, "*.png")))

    if len(image_paths_a) == 0:
        raise Exception("image_dir_a contains no image files")

    image_paths_b = sorted(glob.glob(os.path.join(args.image_dir_b, "*.png")))

    if len(image_paths_b) != len(image_paths_a):
        raise Exception("image_dir_b must contain the same number of image files as image_dir_a")

    image_paths = list(zip(image_paths_a, image_paths_b))

    images = get_images(image_paths)

    if args.operation == "metrics":
        metrics = run_metrics(images)
        write_metrics(metrics, args.output_dir)
    elif args.operation == "heatmaps":
        generate_heatmaps(images, args.output_dir)
    else:
        raise Exception("operation must be 'metrics' or 'heatmaps'")


if __name__ == '__main__':
    main()
