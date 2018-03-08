"""
Adapted from github.com/affinelayer/pix2pix-tensorflow
"""
import tensorflow as tf


def discrim_conv(batch_input, out_channels, stride):
    """
    Discriminator Convolution
    """
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid",
                            kernel_initializer=tf.random_normal_initializer(0, 0.02))


def gen_conv(batch_input, out_channels):
    """
    Generator Convolution
    """
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                            kernel_initializer=tf.random_normal_initializer(0, 0.02))


def gen_deconv(batch_input, out_channels):
    """
    Transposed Convolution
    """
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                      kernel_initializer=tf.random_normal_initializer(0, 0.02))


def lrelu(x, a):
    """
    Leaky ReLU
    """
    return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(tf.identity(x))


def batchnorm(inputs):
    """
    Batch Normalisation
    """
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
