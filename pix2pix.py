from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
import ops
from .model import build_darknet_model
from tensorflow.keras.layers import (Input, Conv2D, Concatenate,
                                     MaxPooling2D, BatchNormalization,
                                     LeakyReLU)
from tensorflow.keras.activations import tanh
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.losses import mean_squared_error, sparse_categorical_crossentropy


# Define globals.
EPS = 1e-12
Examples = collections.namedtuple(
    "Examples",
    "paths, inputs, targets, count, steps_per_epoch"
)
Model = collections.namedtuple(
    "Model",
    "outputs, predict_real, predict_fake, detect_real, detect_fake, discrim_loss, discrim_grads_and_vars, task_loss, task_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train"
)


def preprocess(image, add_noise=False):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        # return image * 2 - 1
        if add_noise:
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0,
                                     stddev=0.5, dtype=tf.float32)
            return tf.image.per_image_standardization(image) + noise
        else:
            return tf.image.per_image_standardization(image)


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        # return (image + 1) / 2
        return image


def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack(
            [(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110],
            axis=3
        )


def augment(a, image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(a, lab)
    return rgb


# def discrim_conv(batch_input, out_channels, stride):
#     padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]],
#                           mode="CONSTANT")
#     return tf.layers.conv2d(
#         padded_input,
#         out_channels,
#         kernel_size=4,
#         strides=(stride, stride),
#         padding="valid",
#         kernel_initializer=tf.random_normal_initializer(0, 0.02)
#     )
#
#
# def gen_conv(a, batch_input, out_channels):
#     # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
#     initializer = tf.random_normal_initializer(0, 0.02)
#     if a.separable_conv:
#         return tf.layers.separable_conv2d(batch_input, out_channels,
#                                           kernel_size=4, strides=(2, 2),
#                                           padding="same",
#                                           depthwise_initializer=initializer,
#                                           pointwise_initializer=initializer)
#     else:
#         return tf.layers.conv2d(batch_input, out_channels, kernel_size=4,
#                                 strides=(2, 2), padding="same",
#                                 kernel_initializer=initializer)
#
#
# def gen_deconv(a, batch_input, out_channels):
#     # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
#     initializer = tf.random_normal_initializer(0, 0.02)
#     if a.separable_conv:
#         _b, h, w, _c = batch_input.shape
#         resized_input = tf.image.resize_images(
#             batch_input,
#             [h * 2, w * 2],
#             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
#         )
#         return tf.layers.separable_conv2d(
#             resized_input,
#             out_channels,
#             kernel_size=4,
#             strides=(1, 1),
#             padding="same",
#             depthwise_initializer=initializer,
#             pointwise_initializer=initializer
#         )
#     else:
#         return tf.layers.conv2d_transpose(
#             batch_input,
#             out_channels,
#             kernel_size=4,
#             strides=(2, 2),
#             padding="same",
#             kernel_initializer=initializer
#         )


# def lrelu(x, a):
#     with tf.name_scope("lrelu"):
#         # adding these together creates the leak part and linear part
#         # then cancels them out by subtracting/adding an absolute value term
#         # leak: a*x/2 - a*abs(x)/2
#         # linear: x/2 + abs(x)/2
# 
#         # this block looks like it has 2 inputs on the graph unless we do this
#         x = tf.identity(x)
#         return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


# def batchnorm(inputs):
#     return tf.layers.batch_normalization(
#         inputs,
#         axis=3,
#         epsilon=1e-5,
#         momentum=0.1,
#         training=True,
#         gamma_initializer=tf.random_normal_initializer(1.0, 0.02)
#     )


def check_image(a, image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3,
                                message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension a.n_channels so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = a.n_channels
    image.set_shape(shape)
    return image


# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(a, srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(a, srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + \
                (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)
            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels,
                                                [1/0.950456, 1.0, 1/1.088754])
            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3),
                                  dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3),
                                       dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) \
                * linear_mask + (xyz_normalized_pixels ** (1/3)) \
                * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + \
                tf.constant([-16.0, 0.0, 0.0])
        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(a, lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(a, lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(
                lab_pixels + tf.constant([16.0, 0.0, 0.0]),
                lab_to_fxfyfz
            )
            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon,
                                       dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * \
                linear_mask + (fxfyfz_pixels ** 3) * exponential_mask
            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + \
                ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask
        return tf.reshape(srgb_pixels, tf.shape(lab))


def load_examples(a):
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")
    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png
    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")
    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name
    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)
    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(
            input_paths,
            shuffle=a.mode == "train"
        )
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)
        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3,
                                    message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, a.n_channels])

        if a.lab_colorization:
            # load color and brightness from image, no B image exists here
            lab = rgb_to_lab(a, raw_input)
            L_chan, a_chan, b_chan = preprocess_lab(lab)
            a_images = tf.expand_dims(L_chan, axis=2)
            b_images = tf.stack([a_chan, b_chan], axis=2)
        else:
            # break apart image pair and move to range [-1, 1]
            width = tf.shape(raw_input)[1] # [height, width, channels]
            a_images = preprocess(raw_input[:, :width//2, :], add_noise=True)
            b_images = preprocess(raw_input[:, width//2:, :])
    if a.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif a.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise ValueError(f"Invalid direction given: {a.which_direction}")

    # synchronize seed for image operations so that we do the same operations
    # to both input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)
        # area produces a nice downscaling, but does nearest neighbor for
        # upscaling assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size],
                                   method=tf.image.ResizeMethod.AREA)
        offset = tf.cast(tf.floor(
            tf.random_uniform(
                [2],
                0,
                a.scale_size - a.crop_size + 1,
                seed=seed)
            ),
            dtype=tf.int32
        )
        if a.scale_size > a.crop_size:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1],
                                              a.crop_size, a.crop_size)
        elif a.scale_size < a.crop_size:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)
    with tf.name_scope("target_images"):
        target_images = transform(targets)
    paths_batch, inputs_batch, targets_batch = tf.train.batch(
        [paths, input_images, target_images],
        batch_size=a.batch_size
    )
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))
    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )


def google_attention(x, filters, sn=False, scope='attention'):
    with tf.name_scope(scope):
        batch_size, height, width, num_channels = x.get_shape().as_list()
        f = ops.conv(x, filters // 8, kernel_size=(1, 1), strides=(1, 1),
                     sn=sn, scope='f_conv')  # [bs, h, w, c']
        f = MaxPooling2D()(f)
        g = ops.conv(x, filters // 8, kernel_size=(1, 1), strides=(1, 1),
                     sn=sn, scope='g_conv')  # [bs, h, w, c']
        h = ops.conv(x, filters // 2, kernel_size=(1, 1), strides=(1, 1),
                     sn=sn, scope='h_conv')  # [bs, h, w, c]
        h = MaxPooling2D()(h)

        # N = h * w
        s = tf.matmul(ops.hw_flatten(g), ops.hw_flatten(f),
                      transpose_b=True)  # # [bs, N, N]
        beta = tf.nn.softmax(s)  # attention map
        o = tf.matmul(beta, ops.hw_flatten(h))  # [bs, N, C]
        gamma = tf.compat.v1.get_variable(
            "gamma",
            [1],
            initializer=tf.constant_initializer(0.0)
        )
        o = tf.reshape(o, shape=[batch_size,
                                 height,
                                 width,
                                 num_channels // 2])  # [bs, h, w, C]
        o = ops.conv(o, filters, kernel_size=(1, 1), strides=(1, 1), sn=sn,
                     scope='attn_conv')
        x = gamma * o + x
    return x


def create_generator(a, input_shape, generator_outputs_channels):
    """Creates the generator network.

    Args:
        a: command-line arguments object.
        input_shape: shape of the input image.
        generator_output_channels: number of channels generator should output.

    Returns:
        Generator network model.
    """

    x_in = Input(shape=input_shape)
    num_blocks = 8
    num_filters = a.ngf
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.name_scope("generator"):
        skip_layers = []
        x = ops.up_resblock(x_in, filters=num_filters, sn=a.spec_norm,
                            scope='front_down_resblock_0')
        for i in range(num_blocks // 2):
            x = ops.down_resblock(x, filters=num_filters // 2, sn=a.spec_norm,
                                  scope=f'mid_down_resblock_{i}')
            num_filters = num_filters // 2
            skip_layers.append(x)

        x = google_attention(x, filters=num_filters, scope='self_attention')

        # Build the back end of the generator with skip connections.
        for i in range(num_blocks // 2, num_blocks):
            x = ops.up_resblock(Concatenate()([x, skip_layers.pop()]),
                                filters=num_filters,
                                sn=a.sn,
                                scope=f'back_up_resblock_{i}')
            num_filters = num_filters * 2

        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = ops.deconv(x, filters=generator_outputs_channels, padding='same',
                       scope='g_logit')
        x = tanh(x, name='generator')
        return Model(inputs=x_in, outputs=x)


def create_discriminator(a, input_shape, target_shape):
    """Creates the discriminator network.

    Args:
        a: command-line arguments object.
        input_shape: input images shape (generator priors).
        target_shape: target images shape (real or generator outputs).

    Returns:
        Discriminator network model.
    """
    n_layers = 3

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    x_in = Input(shape=input_shape)
    y_in = Input(shape=target_shape)
    input_concat = Concatenate([x_in, y_in], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    x = ops.down_resblock(input_concat, filters=a.ndf,
                          sn=a.spec_norm, scope='layer_1')

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        out_channels = a.ndf * min(2**(i+1), 8)
        x = ops.down_resblock(x, filters=out_channels, sn=a.spec_norm,
                              scope=f'layer_{i+1}')
        x = BatchNormalization()(x)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    x = ops.down_resblock(x, filters=1, to_down=False, sn=a.spec_norm,
                          scope=f'layer_{n_layers + 1}')
    x = tf.nn.sigmoid(x, name='discriminator')

    return Model(inputs=[x_in, y_in], outputs=x)


def create_task_net(a, input_shape):
    """Creates the task network.

    Args:
        input_shape: shape of input images.

    Returns:
        Task network (detection) model.
    """
    # Feature pyramid network or darknet or something with res blocks.
    model = build_darknet_model(input_shape)
    # Predictor heads for object centroid, width, height.
    pred_xy = Conv2D(
        filters=2,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer='he_normal',
        activation='sigmoid',
        kernel_regularizer=l1_l2(a.l1_reg_kernel,
                                 a.l2_reg_kernel),
        bias_regularizer=l1_l2(a.l1_reg_bias,
                               a.l2_reg_bias)
    )(model.output)
    pred_wh = Conv2D(
        filters=2,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer='he_normal',
        activation='sigmoid',
        kernel_regularizer=l1_l2(a.l1_reg_kernel,
                                 a.l2_reg_kernel),
        bias_regularizer=l1_l2(a.l1_reg_bias,
                               a.l2_reg_bias)
    )(model.output)
    pred_obj = Conv2D(
        filters=2,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer='he_normal',
        activation='sigmoid',
        kernel_regularizer=l1_l2(a.l1_reg_kernel,
                                 a.l2_reg_kernel),
        bias_regularizer=l1_l2(a.l1_reg_bias,
                               a.l2_reg_bias)
    )(model.output)
    pred_class = Conv2D(
        filters=a.num_classes,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer='he_normal',
        activation='sigmoid',
        kernel_regularizer=l1_l2(a.l1_reg_kernel,
                                 a.l2_reg_kernel),
        bias_regularizer=l1_l2(a.l1_reg_bias,
                               a.l2_reg_bias)
    )(model.output)
    return Model(inputs=model.input,
                 outputs=[pred_xy, pred_wh, pred_obj, pred_class])


def create_model(a, inputs, targets, task_targets=None):
    input_shape = inputs.shape.as_list()
    target_shape = targets.shape.as_list()
    with tf.name_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        generator = create_generator(a, input_shape, out_channels)
        fake_img = generator(inputs)

    # Create two copies of the task network, one for real images (targets
    # input to this method) and one for generated images (outputs from
    # generator). The task targets (detected objects) should be the same for
    # both.
    with tf.name_scope('task_net'):
        task_net = create_task_net(a, target_shape)

    # Create two copies of discriminator, one for real pairs and one for fake
    # pairs they share the same underlying variables
    with tf.name_scope("discriminator"):
        # TODO (NLT): figure out discriminator loss, interaction with Keras changes.
        discriminator = create_discriminator(a, input_shape, target_shape)
        predict_real = discriminator(inputs, targets)
        predict_fake = discriminator(inputs, fake_img)
    #     with tf.compat.v1.variable_scope("discriminator"):
    #         # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
    #         predict_real = create_discriminator(a, inputs, targets)

    # with tf.name_scope("fake_discriminator"):
    #     with tf.compat.v1.variable_scope("discriminator", reuse=True):
    #         # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
    #         predict_fake = create_discriminator(a, inputs, generator.outputs[0])

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.math.log(predict_real + EPS) + \
                       tf.math.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.math.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - fake_img))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope('task_loss'):
        # TODO (NLT): implement YOLO loss or similar for detection.
        pred_xy, pred_wh, pred_obj, pred_class = task_net(targets)
        pred_xy_fake, pred_wh_fake, pred_obj_fake, pred_class_fake = \
            task_net(fake_img)
        xy_loss = mean_squared_error(pred_xy, task_targets.xy)
        wh_loss = mean_squared_error(pred_wh, task_targets.wh)
        obj_loss = sparse_categorical_crossentropy(pred_obj, task_targets.obj)
        class_loss = sparse_categorical_crossentropy(pred_class, 0)
        task_loss_real = xy_loss + wh_loss + obj_loss + class_loss
        xy_loss_fake = mean_squared_error(pred_xy_fake, task_targets.xy)
        wh_loss_fake = mean_squared_error(pred_wh_fake, task_targets.wh)
        obj_loss_fake = sparse_categorical_crossentropy(pred_obj_fake,
                                                        task_targets.obj)
        class_loss_fake = sparse_categorical_crossentropy(pred_class_fake, 0)
        task_loss_fake = xy_loss_fake + wh_loss_fake + obj_loss_fake + \
            class_loss_fake
        task_loss = task_loss_real + task_loss_fake
    
    model = Model(inputs=[inputs, targets],
                  outputs=[generator.outputs, discriminator.outputs,
                           task_net.outputs])

    # TODO (NLT): compile the model with appropriate losses, optimizers, callbacks, etc.
    losses = {'generator': gen_loss,
              'discriminator': discrim_loss,
              'task_net': task_loss}
    loss_weights = {'generator': a.gen_weight,
                    'discriminator': a.dsc_weight,
                    'task_net': a.task_weight}}
    optimizers = {'generator': 'adam',
                  'discriminator': 'adam',
                  'task_net': 'adam'}
    metrics = {'generator': 'mse',
               'discriminator': 'sparse_categorical_crossentropy',
               'task_net': 'mse'}
    model.compile(loss=losses
                  loss_weights=loss_weights,
                  optimizer=optimizers,
                  metrics=metrics)


def save_images(a, fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(a, filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")
    for fileset in filesets:
        index.write("<tr>")
        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])
        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])
        index.write("</tr>")
    return index_path


def main(a):
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)
    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)
    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")
        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = a.crop_size
        a.flip = False
    for k, v in a._get_kwargs():
        print(k, "=", v)
    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    if a.mode == "export":
        # export the generator to a meta graph that can be imported later for standalone generation
        if a.lab_colorization:
            raise Exception("export not supported for lab_colorization")
        x_in = tf.placeholder(tf.string, shape=[1])
        input_data = tf.decode_base64(x_in[0])
        input_image = tf.image.decode_png(input_data)
        # remove alpha channel if present
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 4),
                                       lambda: input_image[:,:,:3],
                                       lambda: input_image)
        # convert grayscale to RGB
        if a.n_channels == 3:
            input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 1),
                                  lambda: tf.image.grayscale_to_rgb(input_image),
                                  lambda: input_image)
        input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
        input_image.set_shape([a.crop_size, a.crop_size, a.n_channels])
        batch_input = tf.expand_dims(input_image, axis=0)

        with tf.variable_scope("generator"):
            batch_output = deprocess(
                create_generator(
                    a,
                    preprocess(batch_input, add_noise=True),
                    a.n_channels
                )
            )
        output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.float32)[0]
        if a.output_filetype == "png":
            output_data = tf.image.encode_png(output_image)
        elif a.output_filetype == "jpeg":
            output_data = tf.image.encode_jpeg(output_image, quality=80)
        else:
            raise Exception("invalid filetype")
        output = tf.convert_to_tensor([tf.encode_base64(output_data)])
        key = tf.placeholder(tf.string, shape=[1])
        inputs = {
            "key": key.name,
            "input": x_in.name
        }
        tf.add_to_collection("inputs", json.dumps(inputs))
        outputs = {
            "key":  tf.identity(key).name,
            "output": output.name,
        }
        tf.add_to_collection("outputs", json.dumps(outputs))
        init_op = tf.global_variables_initializer()
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            restore_saver.restore(sess, checkpoint)
            print("exporting model")
            export_saver.export_meta_graph(
                filename=os.path.join(a.output_dir, "export.meta")
            )
            export_saver.save(sess, os.path.join(a.output_dir, "export"),
                              write_meta_graph=False)
        return  # if a.mode == 'export'

    examples = load_examples(a)
    print("examples count = %d" % examples.count)
    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(a, examples.inputs, examples.targets)
    # undo colorization splitting on images that we use for display/output
    if a.lab_colorization:
        if a.which_direction == "AtoB":
            # inputs is brightness, this will be handled fine as a grayscale image
            # need to augment targets and outputs with brightness
            if a.n_channels > 1:
                targets = augment(a, examples.targets, examples.inputs)
                outputs = augment(a, model.outputs, examples.inputs)
            # inputs can be deprocessed normally and handled as if they are single channel
            # grayscale images
            inputs = deprocess(examples.inputs)
        elif a.which_direction == "BtoA":
            # inputs will be color channels only, get brightness from targets
            if a.n_channels > 1:
                inputs = augment(a, examples.inputs, examples.targets)
            targets = deprocess(examples.targets)
            outputs = deprocess(model.outputs)
        else:
            raise Exception("invalid direction")
    else:
        inputs = deprocess(examples.inputs)
        targets = deprocess(examples.targets)
        outputs = deprocess(model.outputs)

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [a.crop_size, int(round(a.crop_size * a.aspect_ratio))]
            image = tf.image.resize_images(
                image,
                size=size,
                method=tf.image.ResizeMethod.BICUBIC
            )
        return tf.image.convert_image_dtype(image, dtype=tf.uint16,
                                            saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)
    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)
    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)
    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs,
                                dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets,
                                 dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs,
                                 dtype=tf.string, name="output_pngs"),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        #summary_inputs = tf.image.convert_image_dtype(converted_inputs,
        #                                              dtype=tf.float32)
        #tf.summary.image("inputs", summary_inputs)
        tf.summary.image("inputs", inputs)
    with tf.name_scope("targets_summary"):
        #summary_targets = tf.image.convert_image_dtype(converted_targets,
        #                                               dtype=tf.float32)
        #tf.summary.image("targets", summary_targets)
        tf.summary.image("targets", targets)
    with tf.name_scope("outputs_summary"):
        #summary_outputs = tf.image.convert_image_dtype(converted_outputs,
        #                                               dtype=tf.float32)
        #tf.summary.image("outputs", summary_outputs)
        tf.summary.image("outputs", outputs)
    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real",
                         tf.image.convert_image_dtype(model.predict_real,
                                                      dtype=tf.float32))
    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake",
                         tf.image.convert_image_dtype(model.predict_fake,
                                                      dtype=tf.float32))
    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(a, results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(a, filesets)
            print("wrote index at", index_path)
            print("rate", (time.time() - start) / max_steps)
        else:
            # training
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(a, results["display"], step=results["global_step"])
                    append_index(a, filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("gen_loss_L1", results["gen_loss_L1"])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="path to folder containing images")
    parser.add_argument("--mode", required=True,
                        choices=["train", "test", "export"])
    parser.add_argument("--output_dir", required=True,
                        help="where to put output files")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", default=None,
                        help="directory with checkpoint to resume training from or use for testing")

    parser.add_argument("--max_steps", type=int,
                        help="number of training steps (0 to disable)")
    parser.add_argument("--max_epochs", type=int, help="number of training epochs")
    parser.add_argument("--summary_freq", type=int, default=100,
                        help="update summaries every summary_freq steps")
    parser.add_argument("--progress_freq", type=int, default=50,
                        help="display progress every progress_freq steps")
    parser.add_argument("--trace_freq", type=int, default=0,
                        help="trace execution every trace_freq steps")
    parser.add_argument("--display_freq", type=int, default=0,
                        help="write current training images every display_freq steps")
    parser.add_argument("--save_freq", type=int, default=5000,
                        help="save model every save_freq steps, 0 to disable")

    parser.add_argument("--separable_conv", action="store_true",
                        help="use separable convolutions in the generator")
    parser.add_argument("--aspect_ratio", type=float, default=1.0,
                        help="aspect ratio of output images (width/height)")
    parser.add_argument("--lab_colorization", action="store_true",
                        help="split input image into brightness (A) and color (B)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="number of images in batch")
    parser.add_argument("--which_direction", type=str, default="AtoB",
                        choices=["AtoB", "BtoA"])
    parser.add_argument("--ngf", type=int, default=64,
                        help="number of generator filters in first conv layer")
    parser.add_argument("--ndf", type=int, default=64,
                        help="number of discriminator filters in first conv layer")
    parser.add_argument("--scale_size", type=int, default=286,
                        help="scale images to this size before cropping to 256x256")
    parser.add_argument("--flip", dest="flip", action="store_true",
                        help="flip images horizontally")
    parser.add_argument("--no_flip", dest="flip", action="store_false",
                        help="don't flip images horizontally")
    parser.set_defaults(flip=True)
    parser.add_argument("--lr_gen", type=float, default=4e-4,
                        help="initial learning rate for generator adam")
    parser.add_argument("--lr_dsc", type=float, default=1e-4,
                        help="initial learning rate for discriminator adam")
    parser.add_argument("--lr_task", type=float, default=1e-4,
                        help="initial learning rate for task adam")
    parser.add_argument("--beta1_gen", type=float, default=0.5,
                        help="momentum term of generator adam")
    parser.add_argument("--beta1_dsc", type=float, default=0.5,
                        help="momentum term of discriminator adam")
    parser.add_argument("--beta1_task", type=float, default=0.5,
                        help="momentum term of task adam")
    parser.add_argument("--l1_weight", type=float, default=100.0,
                        help="weight on L1 term for generator gradient")
    parser.add_argument("--gan_weight", type=float, default=1.0,
                        help="weight on GAN term for generator gradient")
    parser.add_argument("--n_channels", type=int, default=3,
                        help="Number of channels in image.")
    parser.add_argument("--transform", action="store_true", default=False,
                        help="Whether to apply image transformations.")
    parser.add_argument("--crop_size", type=int, default=256,
                        help="Size of cropped image chunks.")
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Number of classes in the data.')
    parser.add_argument('--l1_reg_kernel', type=float, default=0.,
                        help='L1 regularization term for kernels')                   
    parser.add_argument('--l2_reg_kernel', type=float, default=0.,
                        help='L2 regularization term for kernels')                   
    parser.add_argument('--l1_reg_bias', type=float, default=0.,
                        help='L1 regularization term for bias')                   
    parser.add_argument('--l2_reg_bias', type=float, default=0.,
                        help='L2 regularization term for bias')                   

    # export options
    parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
    args = parser.parse_args()
    main(args)
