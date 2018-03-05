import tensorflow as tf


def conv(batch_input, out_channels, stride):
    """
    Convolution
    """
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        conv_filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32,
                                      initializer=tf.random_normal_initializer(0, 0.02))
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        return tf.nn.conv2d(padded_input, conv_filter, [1, stride, stride, 1], padding="VALID")


def deconv(batch_input, out_channels):
    """
    Transposed Convolution
    """
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        deconv_filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(0, 0.02))
        return tf.nn.conv2d_transpose(batch_input, deconv_filter, [batch, in_height * 2, in_width * 2, out_channels],
                                      [1, 2, 2, 1], padding="SAME")


def lrelu(x, a):
    """
    Leaky ReLU
    """
    return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    """
    Batch Normalisation
    """
    with tf.variable_scope("batchnorm"):
        channels = inputs.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32,  initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2], keep_dims=False)
        normalized = tf.nn.batch_normalization(inputs, mean, variance, offset, scale, variance_epsilon=1e-5)
        return normalized
