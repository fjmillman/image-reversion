from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import math
import random
import tensorflow as tf


def check_folder(path_dir):
    """
    Checks if directory exists and creates one if not
    """
    if not os.path.isdir(path_dir):
        os.makedirs(path_dir)


def transform(image, seed, scale_size, crop_size):
    """
    Process image to be inputted to model
    """
    image = tf.image.resize_images(image, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)
    offset = tf.cast(tf.floor(tf.random_uniform([2], 0, scale_size - crop_size + 1, seed=seed)), dtype=tf.int32)

    return tf.image.crop_to_bounding_box(image, offset[0], offset[1], crop_size, crop_size)


def convert(image):
    """
    Convert image type
    """
    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)


def pre_process(image):
    """
    Scale pixels of a given image to [-1, 1]
    """
    # [0, 1] => [-1, 1]
    return (image * 2) - 1


def de_process(image):
    """
    Scale pixels of a given image to [0, 1]
    """
    # [-1, 1] => [0, 1]
    return (image + 1) / 2


def load_images(input_dir, batch_size, scale_size, crop_size):
    """
    Load images from the given input directory
    """
    if input_dir is None or not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(input_dir, "*.png"))

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    path_queue = tf.train.string_input_producer(input_paths, shuffle=True)
    reader = tf.WholeFileReader()
    paths, contents = reader.read(path_queue)
    raw_image = tf.image.decode_png(contents)
    raw_image = tf.image.convert_image_dtype(raw_image, dtype=tf.float32)

    assertion = tf.assert_equal(tf.shape(raw_image)[2], 3, message="image does not have 3 channels")
    with tf.control_dependencies([assertion]):
        raw_input = tf.identity(raw_image)

    raw_input.set_shape([None, None, 3])

    width = tf.shape(raw_image)[1]
    left = pre_process(raw_image[:, :width // 2, :])
    right = pre_process(raw_image[:, width // 2:, :])

    seed = random.randint(0, 2 ** 31 - 1)
    left = transform(left, seed, scale_size, crop_size)
    right = transform(right, seed, scale_size, crop_size)

    inputs, targets = right, left

    paths, inputs, targets = tf.train.batch([paths, inputs, targets], batch_size=batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / batch_size))

    return paths, inputs, targets, steps_per_epoch


def save_images(results, output_dir):
    """
    Save images to the given output directory
    """
    image_dir = os.path.join(output_dir, "images")
    check_folder(image_dir)

    filesets = []
    for i, in_path in enumerate(results["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name}
        for kind in ["inputs", "outputs", "targets"]:
            filename = f"{name}-{kind}.png"
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = results[kind][i]
            with open(out_path, "wb") as file:
                file.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, output_dir):
    """
    Write the test results to the index
    """
    index_path = os.path.join(output_dir, "index.html")

    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr><th>Name</th><th>Input</th><th>Output</th><th>Target</th></tr>")

    for fileset in filesets:
        index.write(f"<tr><td>{fileset['name']}</td>")

        for kind in ["inputs", "outputs", "targets"]:
            index.write(f"<td><img src='images/{fileset[kind]}'></td>")

        index.write("</tr>")

    return index_path
