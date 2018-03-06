from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import *
from ops import *

import time
import tensorflow as tf

EPS = 1e-12


class GAN(object):
    def __init__(self, input_dir, output_dir, checkpoint, paths, inputs, targets, batch_size, steps_per_epoch,
                 ngf, ndf, lr, beta1, l1_weight, gan_weight):
        """
        Args:
            input_dir
            output_dir
            checkpoint
            paths
            inputs
            targets
            batch_size
            steps_per_epoch
            ngf
            ndf
            lr
            beta1
            l1_weight
            gan_weight
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.checkpoint = checkpoint
        self.paths = paths
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.ngf = ngf
        self.ndf = ndf
        self.lr = lr
        self.beta1 = beta1
        self.l1_weight = l1_weight
        self.gan_weight = gan_weight

        # Build the model
        self.outputs, self.train_op, self.gen_loss_GAN, self.gen_loss_L1, self.discrim_loss = self.build_model(self.inputs, self.targets)

        self.saver = tf.train.Saver(max_to_keep=1)

    def build_model(self, inputs, targets):
        """
        Build the model
        """
        with tf.variable_scope("generator"):
            out_channels = int(targets.get_shape()[-1])
            outputs = self.generator(inputs, out_channels)

        # create two copies of discriminator, one for real pairs and one for fake pairs
        # they share the same underlying variables
        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_real = self.discriminator(inputs, targets)

        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_fake = self.discriminator(inputs, outputs)

        with tf.name_scope("discriminator_loss"):
            # minimizing -tf.log will try to get inputs to 1
            # predict_real => 1
            # predict_fake => 0
            discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

        with tf.name_scope("generator_loss"):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
            gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
            gen_loss = gen_loss_GAN * self.gan_weight + gen_loss_L1 * self.l1_weight

        with tf.name_scope("discriminator_train"):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer(self.lr, self.beta1)
            discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
            discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

        with tf.name_scope("generator_train"):
            with tf.control_dependencies([discrim_train]):
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                gen_optim = tf.train.AdamOptimizer(self.lr, self.beta1)
                gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step + 1)

        gen_loss_GAN = ema.average(gen_loss_GAN)
        gen_loss_L1 = ema.average(gen_loss_L1)
        discrim_loss = ema.average(discrim_loss)
        train_op = tf.group(update_losses, incr_global_step, gen_train)

        return outputs, train_op, gen_loss_GAN, gen_loss_L1, discrim_loss

    def generator(self, generator_inputs, generator_outputs_channels):
        """
        Create generator neural network
        """
        layers = []

        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        with tf.variable_scope("encoder_1"):
            output = gen_conv(generator_inputs, self.ngf)
            layers.append(output)

        layer_specs = [
            self.ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            self.ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            self.ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            self.ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            self.ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            self.ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            self.ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]

        for out_channels in layer_specs:
            with tf.variable_scope("encoder_{}".format(len(layers) + 1)):
                rectified = lrelu(layers[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height / 2, in_width / 2, out_channels]
                convolved = gen_conv(rectified, out_channels)
                output = batchnorm(convolved)
                layers.append(output)

        layer_specs = [
            (self.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (self.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (self.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (self.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (self.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (self.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (self.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]

        num_encoder_layers = len(layers)
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_{}".format(skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = layers[-1]
                else:
                    input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

                rectified = tf.nn.relu(input)
                # [batch, in_height, in_width, in_channels] => [batch, in_height * 2, in_width * 2, out_channels]
                output = gen_deconv(rectified, out_channels)
                output = batchnorm(output)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

                layers.append(output)

        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, 3]
        with tf.variable_scope("decoder_1"):
            input = tf.concat([layers[-1], layers[0]], axis=3)
            rectified = tf.nn.relu(input)
            output = gen_deconv(rectified, generator_outputs_channels)
            output = tf.tanh(output)
            layers.append(output)

        return layers[-1]

    def discriminator(self, discrim_inputs, discrim_targets):
        """
        Create discriminator neural network
        """
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, self.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = self.ndf * min(2 ** (i + 1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = discrim_conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    def train(self, sv, sess, max_epochs, progress_freq, save_freq):
        """
        Train the GAN
        """
        if self.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(self.checkpoint)
            self.saver.restore(sess, checkpoint)

        max_steps = max_epochs * self.steps_per_epoch

        start = time.time()

        for step in range(max_steps):
            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

            fetches = {
                "train": self.train_op,
                "global_step": sv.global_step,
            }

            if should(progress_freq):
                fetches["discrim_loss"] = self.discrim_loss
                fetches["gen_loss_GAN"] = self.gen_loss_GAN
                fetches["gen_loss_L1"] = self.gen_loss_L1

            results = sess.run(fetches)

            if should(progress_freq):
                train_epoch = math.ceil(results["global_step"] / self.steps_per_epoch)
                train_step = (results["global_step"] - 1) % self.steps_per_epoch + 1
                rate = (step + 1) * self.batch_size / (time.time() - start)
                remaining = (max_steps - step) * self.batch_size / rate
                print(f"Progress | Epoch: {train_epoch} - Step: {train_step} - Image/sec: {rate} - Remaining time: "
                      f"{int(remaining / 60)}m")
                print(f"Discriminator loss: {results['discrim_loss']}")
                print(f"Generator loss GAN: {results['gen_loss_GAN']}")
                print(f"Generator loss L1: {results['gen_loss_L1']}")

            if should(save_freq):
                print("Saving model")
                self.saver.save(sess, os.path.join(self.output_dir, "model"), global_step=sv.global_step)

            if sv.should_stop():
                break

    def test(self, sess):
        """
        Test the GAN
        """
        output_images = {
            "paths": self.paths,
            "inputs": tf.map_fn(tf.image.encode_png, convert(de_process(self.inputs)), dtype=tf.string, name="inputs_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, convert(de_process(self.targets)), dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, convert(de_process(self.outputs)), dtype=tf.string, name="output_pngs"),
        }

        print(f"Number of images: {len(output_images)}")

        start = time.time()

        # Restore from checkpoint
        checkpoint = tf.train.latest_checkpoint(self.output_dir)
        self.saver.restore(sess, checkpoint)

        # Save outputs
        for step in range(self.steps_per_epoch):
            results = sess.run(output_images)
            filesets = save_images(results, self.output_dir)
            for fileset in filesets:
                print(f"Evaluated image {fileset['name']}")
            index_path = append_index(filesets, self.output_dir)

        print(f"Wrote index at {index_path}")
        print(f"Rate: {(time.time() - start) / self.steps_per_epoch}")
