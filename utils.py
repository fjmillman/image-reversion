from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import math
import tensorflow as tf


def check_folder(path_dir):
    """
    Checks if directory exists and creates one if not
    """
    if not os.path.isdir(path_dir):
        os.makedirs(path_dir)


def transform(image):
    """
    Resize image to 256 pixels height and width
    """
    return tf.image.resize_images(image, [256, 256], method=tf.image.ResizeMethod.AREA)


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


def load_images(input_dir, batch_size):
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
    raw_image.set_shape([None, None, 3])

    width = tf.shape(raw_image)[1]
    left, right = pre_process(raw_image[:, :width // 2, :]), pre_process(raw_image[:, width // 2:, :])

    inputs, targets = transform(left), transform(right)

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
