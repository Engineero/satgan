from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
import argparse
<<<<<<< HEAD
import os
import json
import glob
import random
import collections
import math
import time
from ops import conv, max_pooling, hw_flatten

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True,
                    choices=["train", "test", "export"])
parser.add_argument("--output_dir", required=True,
                    help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument(
    "--checkpoint",
    default=None,
    help="directory with checkpoint to resume training from or use for testing"
)
parser.add_argument("--max_steps", type=int,
                    help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int,
                    help="number of training epochs")
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
parser.add_argument("--flip", dest="flip", action="store_true", default=False,
                    help="flip images horizontally")
parser.add_argument("--lr_gen", type=float, default=4e-4,
                    help="initial learning rate for generator adam")
parser.add_argument("--lr_dsc", type=float, default=1e-4,
                    help="initial learning rate for discriminator adam")
parser.add_argument("--beta1_gen", type=float, default=0.5,
                    help="momentum term of generator adam")
parser.add_argument("--beta1_dsc", type=float, default=0.5,
                    help="momentum term of discriminator adam")
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

# export options
parser.add_argument("--output_filetype", default="png",
                    choices=["png", "jpeg"])
a = parser.parse_args()

EPS = 1e-12

Examples = collections.namedtuple(
    "Examples",
    "paths, inputs, objects, targets, count, steps_per_epoch"
)
Model = collections.namedtuple(
    "Model",
    "outputs, combined_outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train"
)


def preprocess(image, add_noise=False):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        # return image * 2 - 1
        if add_noise:
            noise = tf.random_normal(shape=tf.shape(image), mean=0.0,
                                     stddev=1., dtype=tf.float32)
            return (tf.image.per_image_standardization(image), noise)
        else:
            return tf.image.per_image_standardization(image)
=======
from pathlib import Path
from utils import ops
from utils.darknet import build_darknet_model
from tensorflow.keras.layers import (Input, Conv2D, Concatenate, Dropout,
                                     MaxPooling2D, BatchNormalization,
                                     LeakyReLU, GlobalAveragePooling2D)
from tensorflow.keras.activations import tanh
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.losses import (MSE, mean_absolute_error,
                                     categorical_crossentropy)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import Mean
from yolo_v3 import build_yolo_model, load_yolo_model_weights


def preprocess(image, add_noise=False):
    """Performs image standardization, optinoally adds Gaussian noise.
>>>>>>> release-1.0.0

    Args:
        image: the image to transform

    Keyword Args:
        add_noise: whether to add Gaussian noise to the image. Default is
            False.

    Returns:
        Image shifted to zero mean and unit standard deviation with optional
            Gaussian noise added.
    """
    with tf.name_scope("preprocess"):
        image = tf.cast(image, tf.float32)
        result = tf.image.per_image_standardization(image)
        if add_noise:
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0,
                                     stddev=1.0, dtype=tf.float32)
            return (result, noise)
        return result


def _parse_example(serialized_example, a):
    """Parses a single TFRecord Example for the task network."""
    # Parse serialized example.
    example = tf.io.parse_example(
        serialized_example,
        {
            'a_raw': tf.io.VarLenFeature(dtype=tf.string),
            'b_raw': tf.io.VarLenFeature(dtype=tf.string),
            'a_filename': tf.io.VarLenFeature(dtype=tf.string),
            'a_height': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'a_width': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'a_classes': tf.io.VarLenFeature(dtype=tf.int64),
            'a_ymin': tf.io.VarLenFeature(dtype=tf.float32),
            'a_ymax': tf.io.VarLenFeature(dtype=tf.float32),
            'a_ycenter': tf.io.VarLenFeature(dtype=tf.float32),
            'a_xmin': tf.io.VarLenFeature(dtype=tf.float32),
            'a_xmax': tf.io.VarLenFeature(dtype=tf.float32),
            'a_xcenter': tf.io.VarLenFeature(dtype=tf.float32),
            'a_magnitude': tf.io.VarLenFeature(dtype=tf.float32),
            'b_filename': tf.io.VarLenFeature(dtype=tf.string),
            'b_height': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'b_width': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'b_classes': tf.io.VarLenFeature(dtype=tf.int64),
            'b_ymin': tf.io.VarLenFeature(dtype=tf.float32),
            'b_ymax': tf.io.VarLenFeature(dtype=tf.float32),
            'b_ycenter': tf.io.VarLenFeature(dtype=tf.float32),
            'b_xmin': tf.io.VarLenFeature(dtype=tf.float32),
            'b_xmax': tf.io.VarLenFeature(dtype=tf.float32),
            'b_xcenter': tf.io.VarLenFeature(dtype=tf.float32),
            'b_magnitude': tf.io.VarLenFeature(dtype=tf.float32),
        }
    )

    # Cast parsed objects into usable types.
    a_width = tf.cast(example['a_width'], tf.int32)
    a_height = tf.cast(example['a_height'], tf.int32)
    a_xcenter = tf.cast(tf.sparse.to_dense(example['a_xcenter']), tf.float32)
    # a_xmin = tf.cast(tf.sparse.to_dense(example['a_xmin']), tf.float32)
    # a_xmax = tf.cast(tf.sparse.to_dense(example['a_xmax']), tf.float32)
    a_ycenter = tf.cast(tf.sparse.to_dense(example['a_ycenter']), tf.float32)
    # a_ymin = tf.cast(tf.sparse.to_dense(example['a_ymin']), tf.float32)
    # a_ymax = tf.cast(tf.sparse.to_dense(example['a_ymax']), tf.float32)
    a_classes = tf.cast(tf.sparse.to_dense(example['a_classes']), tf.float32)
    b_width = tf.cast(example['b_width'], tf.int32)
    b_height = tf.cast(example['b_height'], tf.int32)
    # b_xcenter = tf.cast(tf.sparse.to_dense(example['b_xcenter']), tf.float32)
    b_xmin = tf.cast(tf.sparse.to_dense(example['b_xmin']), tf.float32)
    b_xmax = tf.cast(tf.sparse.to_dense(example['b_xmax']), tf.float32)
    # b_ycenter = tf.cast(tf.sparse.to_dense(example['b_ycenter']), tf.float32)
    b_ymin = tf.cast(tf.sparse.to_dense(example['b_ymin']), tf.float32)
    b_ymax = tf.cast(tf.sparse.to_dense(example['b_ymax']), tf.float32)
    b_classes = tf.cast(tf.sparse.to_dense(example['b_classes']), tf.float32)

    # Calculate bounding boxes for A images (SatSim makes really tight boxes).
    a_xmin = a_xcenter - 10. / tf.cast(a_width[0], tf.float32)
    a_xmax = a_xcenter + 10. / tf.cast(a_width[0], tf.float32)
    a_ymin = a_ycenter - 10. / tf.cast(a_height[0], tf.float32)
    a_ymax = a_ycenter + 10. / tf.cast(a_height[0], tf.float32)

    # Parse images and preprocess.
    a_image = tf.sparse.to_dense(example['a_raw'], default_value='')
    a_image = tf.io.decode_raw(a_image, tf.uint16)
    a_image = tf.reshape(a_image, [-1, a_height[0], a_width[0], 1])
    b_image = tf.sparse.to_dense(example['b_raw'], default_value='')
    b_image = tf.io.decode_raw(b_image, tf.uint16)
    b_image = tf.reshape(b_image, [-1, b_height[0], b_width[0], 1])

    # Package things up for output.
    a_objects = tf.stack([a_ymin, a_xmin, a_ymax, a_xmax, a_classes], axis=-1)
    b_objects = tf.stack([b_ymin, b_xmin, b_ymax, b_xmax, b_classes], axis=-1)

    # Need to pad objects to max inferences (not all images will have same
    # number of objects).
    a_paddings = tf.constant([[0, 0], [0, a.max_inferences], [0, 0]])
    a_paddings = a_paddings - (tf.constant([[0, 0], [0, 1], [0, 0]]) * tf.shape(a_objects)[1])
    a_objects = tf.pad(tensor=a_objects, paddings=a_paddings, constant_values=0.)
    a_objects = tf.tile(a_objects, [1, a.num_pred_layers, 1])
    b_paddings = tf.constant([[0, 0], [0, a.max_inferences], [0, 0]])
    b_paddings = b_paddings - (tf.constant([[0, 0], [0, 1], [0, 0]]) * tf.shape(b_objects)[1])
    b_objects = tf.pad(tensor=b_objects, paddings=b_paddings, constant_values=0.)
    b_objects = tf.tile(b_objects, [1, a.num_pred_layers, 1])

    # task_targets = (objects, width, height)
    if a.which_direction == 'AtoB':
        a_image, gen_input = preprocess(a_image, add_noise=True)
        b_image = preprocess(b_image, add_noise=False)
        return ((a_image, gen_input, b_image), (b_image, a_objects, b_objects))
    else:
        b_image, gen_input = preprocess(b_image, add_noise=True)
        a_image = preprocess(a_image, add_noise=False)
        return ((b_image, gen_input, a_image), (a_image, b_objects, a_objects))


def load_examples(a):
    # Create data queue from training dataset.
    if a.train_dir is None or not Path(a.train_dir).resolve().is_dir():
        raise NotADirectoryError(
            f"Training directory {a.train_dir} does not exist!"
        )
    train_paths = list(Path(a.train_dir).resolve().glob('**/*.tfrecords'))
    if len(train_paths) == 0:
        raise ValueError(
            f"Training directory {a.input_dir} contains no TFRecords files!"
        )
    train_data = tf.data.TFRecordDataset(
        filenames=[p.as_posix() for p in train_paths]
    )

    # Create data queue from validation dataset.
    if a.valid_dir is None or not Path(a.valid_dir).resolve().is_dir():
        raise NotADirectoryError(
            f"Validation directory {a.valid_dir} does not exist!"
        )
    valid_paths = list(Path(a.valid_dir).resolve().glob('**/*.tfrecords'))
    if len(valid_paths) == 0:
        raise ValueError(
            f"Validation directory {a.valid_dir} contains no TFRecords files!"
        )
    valid_data = tf.data.TFRecordDataset(
        filenames=[p.as_posix() for p in valid_paths]
    )

    # Create data queue from testing dataset, if given.
    if a.test_dir is not None:
        if not Path(a.test_dir).resolve().is_dir():
            raise NotADirectoryError(
                f"Testing directory {a.test_dir} does not exist!"
            )
        test_paths = list(Path(a.test_dir).resolve().glob('**/*.tfrecords'))
        if len(test_paths) == 0:
            raise ValueError(
                f"Testing directory {a.test_dir} contains no TFRecords files!"
            )
        test_data = tf.data.TFRecordDataset(
            filenames=[p.as_posix() for p in test_paths]
        )
<<<<<<< HEAD
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
            lab = rgb_to_lab(raw_input)
            L_chan, a_chan, b_chan = preprocess_lab(lab)
            a_images = tf.expand_dims(L_chan, axis=2)
            b_images = tf.stack([a_chan, b_chan], axis=2)
        else:
            # break apart image pair and move to range [-1, 1]
            width = tf.shape(raw_input)[1] # [height, width, channels]
            a_images, gen_inputs = preprocess(raw_input[:, :width//2, :], add_noise=True)
            b_images = preprocess(raw_input[:, width//2:, :])
    if a.which_direction == "AtoB":
        inputs, objects, targets = [gen_inputs, a_images, b_images]
    elif a.which_direction == "BtoA":
        inputs, objects, targets = [gen_inputs, b_images, a_images]
    else:
        raise Exception("invalid direction")

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
    with tf.name_scope('object_images'):
        object_images = transform(objects)
    with tf.name_scope("target_images"):
        target_images = transform(targets)
    paths_batch, inputs_batch, objects_batch, targets_batch = tf.train.batch(
        [paths, input_images, object_images, target_images],
        batch_size=a.batch_size
    )
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))
    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        objects=objects_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
=======
    else:
        test_data = None

    # Specify transformations on datasets.
    train_data = train_data.map(
        lambda x: _parse_example(x, a)
>>>>>>> release-1.0.0
    )
    train_data = train_data.shuffle(a.buffer_size)
    train_data = train_data.batch(a.batch_size, drop_remainder=True)

    valid_data = valid_data.map(
        lambda x: _parse_example(x, a)
    )
    valid_data = valid_data.batch(a.batch_size, drop_remainder=True)

    if a.test_dir is not None:
        test_data = test_data.map(
            lambda x: _parse_example(x, a)
        )
        test_data = test_data.batch(a.batch_size, drop_remainder=True)
    return train_data, valid_data, test_data


def google_attention(x, filters, sn=False, scope='attention'):
    with tf.name_scope(scope):
        _, height, width, num_channels = x.get_shape().as_list()
        f = ops.conv(x, filters // 8, kernel_size=(1, 1), strides=(1, 1),
                     sn=sn, scope='f_conv')  # [bs, h, w, c']
        f = MaxPooling2D()(f)
        g = ops.conv(x, filters // 8, kernel_size=(1, 1), strides=(1, 1),
                     sn=sn, scope='g_conv')  # [bs, h, w, c']
        h = ops.conv(x, filters // 2, kernel_size=(1, 1), strides=(1, 1),
                     sn=sn, scope='h_conv')  # [bs, h, w, c]
        h = MaxPooling2D()(h)

        # N = h * w
        flat_g = tf.reshape(g, [-1, width*height, filters // 8])
        flat_f = tf.reshape(f, [-1, width*height // 4, filters // 8])
        flat_h = tf.reshape(h, [-1, width*height // 4, filters // 2])
        s = tf.matmul(flat_g, flat_f, transpose_b=True)  # # [bs, N, N]
        beta = tf.nn.softmax(s)  # attention map
        o = tf.matmul(beta, flat_h)  # [bs, N, C]
        gamma = tf.compat.v1.get_variable(
            "gamma",
            [1],
            initializer=tf.constant_initializer(0.0)
        )
        o = tf.reshape(o, shape=[-1,
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

<<<<<<< HEAD
    return layers[-1]


def create_model(inputs, objects, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
=======
    x_in = Input(shape=input_shape)
    num_filters = a.ngf
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.name_scope("generator"):
        if a.use_sagan:
            skip_layers = []
            x = ops.down_resblock(x_in, filters=num_filters, sn=a.spec_norm,
                                  scope='front_down_resblock_0')
            for i in range(a.n_blocks_gen // 2):
                x = ops.down_resblock(x, filters=num_filters // 2, sn=a.spec_norm,
                                      scope=f'mid_down_resblock_{i}')
                num_filters = num_filters // 2
                skip_layers.append(x)

            x = google_attention(x, filters=num_filters, scope='self_attention')

            # Build the back end of the generator with skip connections.
            for i in range(a.n_blocks_gen // 2, a.n_blocks_gen):
                x = ops.up_resblock(Concatenate()([x, skip_layers.pop()]),
                                    filters=num_filters,
                                    sn=a.spec_norm,
                                    scope=f'back_up_resblock_{i}')
                num_filters = num_filters * 2

            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = ops.deconv(x, filters=generator_outputs_channels, padding='same',
                           scope='g_logit')
            x = tanh(x)
        else:
            layers = []
            gen_conv = lambda x, n: ops.conv(x, n, kernel_size=(4, 4),
                                             strides=(2, 2), padding='same')
            gen_deconv = lambda x, n: ops.deconv(x, n, kernel_size=(4, 4),
                                                 strides=(2, 2),
                                                 padding='same')
            x = gen_conv(x_in, a.ngf)
            layers.append(x)

            layer_specs = [
                a.ngf * 2,
                a.ngf * 4,
                a.ngf * 8,
                a.ngf * 8,
                a.ngf * 8,
                a.ngf * 8,
                a.ngf * 8,
                a.ngf * 8,
            ]

            for layer_num, out_channels in enumerate(layer_specs):
                if layer_num == 3:
                    x = google_attention(layers[-1], out_channels, sn=True,
                                         scope='gen_self_attention')
                else:
                    with tf.name_scope(f'encoder_{len(layers) + 1}'):
                        x = BatchNormalization()(layers[-1])
                        x = LeakyReLU()(x)
                        x = gen_conv(x, out_channels)
                layers.append(x)

            layer_specs = [
                (a.ngf * 8, 0.5),
                (a.ngf * 8, 0.5),
                (a.ngf * 8, 0.5),
                (a.ngf * 8, 0.),
                (a.ngf * 4, 0.),
                (a.ngf * 2, 0.),
                (a.ngf, 0.),
            ]

            num_encoder_layers = len(layers) - 1  # -1 for attention layer
            for decoder_layer, (out_channels, rate) in enumerate(layer_specs):
                skip_layer = num_encoder_layers - decoder_layer - 1
                if decoder_layer <= 3:
                    skip_layer += 1  # offset for attention layer
                with tf.name_scope(f'decoder_{skip_layer + 1}'):
                    if decoder_layer == 0:
                        # First decoder layer doesn't have skip connections.
                        x = layers[-1]
                    else:
                        x = Concatenate(axis=3)([layers[-1], layers[skip_layer]])
                    x = BatchNormalization()(x)
                    x = LeakyReLU()(x)
                    x = gen_deconv(x, out_channels)
                    if rate > 0.0:
                        x = Dropout(rate)(x)
                    layers.append(x)
            
            with tf.name_scope('decoder_1'):
                x = Concatenate(axis=3)([layers[-1], layers[0]])
                x = BatchNormalization()(x)
                x = LeakyReLU()(x)
                x = gen_deconv(x, generator_outputs_channels)
                x = tanh(x)

        return Model(inputs=x_in, outputs=x, name='generator')


def create_discriminator(a, target_shape):
    """Creates the discriminator network.

    Args:
        a: command-line arguments object.
        target_shape: target images shape (real or generator outputs).

    Returns:
        Discriminator network model.
    """

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    x_in = Input(shape=target_shape)

    if a.use_sagan:
        # layer_1: [batch, h, w, in_channels * 2] => [batch, h//2, w//2, ndf]
        x = ops.down_resblock(x_in, filters=a.ndf,
                              sn=a.spec_norm, scope='layer_1')

        # layer_2: [batch, h//2, w//2, ndf] => [batch, h//4, w//4, ndf * 2]
        # layer_3: [batch, h//4, w//4, ndf * 2] => [batch, h//8, w//8, ndf * 4]
        # layer_4: [batch, h//8, w//8, ndf * 4] => [batch, h//16, w//16, ndf * 8]
        for i in range(a.n_layer_dsc):
            out_channels = a.ndf * min(2**(i+1), 8)
            x = ops.down_resblock(x, filters=out_channels, sn=a.spec_norm,
                                  scope=f'layer_{i+1}')

        # Add self attention layer before final down resblock.
        x = google_attention(x, out_channels, sn=a.spec_norm,
                             scope='discrim_attention')

        # layer_5: [batch, h//16, w//16, ndf * 8] => [batch, h//16-1, w//16-1, 2]
        x = ops.down_resblock(x, filters=2, to_down=False, sn=a.spec_norm,
                              scope=f'layer_{a.n_layer_dsc + 1}')
        x = tf.nn.softmax(x, name='discriminator')
    else:
>>>>>>> release-1.0.0
        n_layers = 3
        discrim_conv = lambda x, n, s: ops.conv(x, n, kernel_size=(4, 4),
                                                strides=(s, s), padding='same')
        with tf.name_scope('layer_1'):
            x = BatchNormalization()(x_in)
            x = LeakyReLU()(x)
            x = discrim_conv(x, a.ndf, 2)
        for i in range(n_layers):
<<<<<<< HEAD
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2**(i+1), 8)
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

    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(objects, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(objects, outputs + objects)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(outputs + objects - targets))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr_dsc, a.beta1_dsc)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr_gen, a.beta1_gen)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        combined_outputs=outputs+objects,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "objects", "outputs", "targets"]:
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


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
=======
            out_channels = a.ndf * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer stride = 2
            with tf.name_scope(f'layer_{i+2}'):
                x = BatchNormalization()(x)
                x = LeakyReLU()(x)
                x = discrim_conv(x, out_channels, stride)
        with tf.name_scope('output_layer'):
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = discrim_conv(x, 2, 1)
            x = tf.nn.softmax(x, name='discriminator')

    return Model(inputs=x_in, outputs=x, name='discriminator')


def create_task_net(a, input_shape):
    """Creates the task network.

    Args:
        input_shape: shape of input images.

    Returns:
        Task network (detection) model.
    """

    pred_list = []
    # Feature pyramid network or darknet or something with res blocks.
    model = build_darknet_model(input_shape)
    # Predictor heads for object centroid, width, height.
    for _, output in zip(range(a.num_pred_layers), model.outputs):
        # Predict the object centroid.
        pred_xy = Conv2D(
            filters=2*a.max_inferences,
            kernel_size=1,
            strides=1,
            padding='same',
            kernel_initializer='he_normal',
            activation='sigmoid',
            kernel_regularizer=l1_l2(a.l1_reg_kernel,
                                     a.l2_reg_kernel),
            bias_regularizer=l1_l2(a.l1_reg_bias,
                                   a.l2_reg_bias)
        )(output)
        pred_xy = GlobalAveragePooling2D()(pred_xy)
        pred_xy = tf.reshape(pred_xy, (-1, a.max_inferences, 2))
        pred_xy = tf.sigmoid(pred_xy)

        # Predict bounding box width and height.
        pred_wh = Conv2D(
            filters=2*a.max_inferences,
            kernel_size=1,
            strides=1,
            padding='same',
            kernel_initializer='he_normal',
            activation='sigmoid',
            kernel_regularizer=l1_l2(a.l1_reg_kernel,
                                     a.l2_reg_kernel),
            bias_regularizer=l1_l2(a.l1_reg_bias,
                                   a.l2_reg_bias)
        )(output)
        pred_wh = GlobalAveragePooling2D()(pred_wh)
        pred_wh = tf.reshape(pred_wh, (-1, a.max_inferences, 2))
        pred_wh = tf.sigmoid(pred_wh)

        # Predict object confidence.
        pred_conf = Conv2D(
            filters=a.max_inferences,
            kernel_size=1,
            strides=1,
            padding='same',
            kernel_initializer='he_normal',
            activation='sigmoid',
            kernel_regularizer=l1_l2(a.l1_reg_kernel,
                                     a.l2_reg_kernel),
            bias_regularizer=l1_l2(a.l1_reg_bias,
                                   a.l2_reg_bias)
        )(output)
        pred_conf = GlobalAveragePooling2D()(pred_conf)
        pred_conf = tf.nn.sigmoid(pred_conf)
        pred_conf = tf.expand_dims(pred_conf, -1)

        # Predict the class of the object. 0 is no object.
        pred_class = Conv2D(
            filters=a.num_classes*a.max_inferences,
            kernel_size=1,
            strides=1,
            padding='same',
            kernel_initializer='he_normal',
            activation='sigmoid',
            kernel_regularizer=l1_l2(a.l1_reg_kernel,
                                     a.l2_reg_kernel),
            bias_regularizer=l1_l2(a.l1_reg_bias,
                                   a.l2_reg_bias)
        )(output)
        pred_class = GlobalAveragePooling2D()(pred_class)
        pred_class = tf.reshape(pred_class, (-1, a.max_inferences,
                                             a.num_classes))
        pred_class = tf.nn.softmax(pred_class)

        # Shift predicted xy by half width and height.
        pred_xy_min = pred_xy - pred_wh / 2.
        pred_xy_max = pred_xy + pred_wh / 2.

        # Build prediction.
        prediction = tf.concat(
            [pred_xy_min, pred_xy_max, pred_conf, pred_class],
            axis=-1
        )
        pred_list.append(prediction)

    # Combine outputs together.
    if a.num_pred_layers > 1 and len(model.outputs) > 1:
        # pred_xy = tf.stack(xy_list, axis=-1, name='stack_xy')
        predict = tf.concat(pred_list, axis=1, name='concat_xy')
>>>>>>> release-1.0.0
    else:
        predict = pred_list[0]
    return Model(inputs=model.input, outputs=predict, name='task_net')


def create_model(a, train_data):
    (inputs, noise, targets), (_, a_task_targets, b_task_targets) = \
        next(iter(train_data))
    input_shape = inputs.shape.as_list()[1:]  # don't give Input the batch dim
    noise_shape = noise.shape.as_list()[1:]
    target_shape = targets.shape.as_list()[1:]
    a_task_targets_shape = a_task_targets.shape.as_list()[1:]
    b_task_targets_shape = b_task_targets.shape.as_list()[1:]
    inputs = Input(input_shape)
    noise = Input(noise_shape)
    targets = Input(target_shape)
    a_task_targets = Input(a_task_targets_shape)
    b_task_targets = Input(b_task_targets_shape)

    with tf.device(f'/device:GPU:{a.devices[0]}'):
        # Create the generator.
        with tf.name_scope("generator"):
            out_channels = target_shape[-1]
            generator = create_generator(a, input_shape, out_channels)
            print(f'Generator model summary:\n{generator.summary()}')
            gen_noise = generator(noise)
            fake_img = gen_noise + inputs
            gen_outputs = tf.stack([fake_img, gen_noise], axis=0,
                                   name='generator')

        # Create two copies of discriminator, one for real pairs and one for
        # fake pairs they share the same underlying variables.
        with tf.name_scope("discriminator"):
            # TODO (NLT): figure out discriminator loss, interaction with Keras changes.
            discriminator = create_discriminator(a, target_shape)
            print(f'Discriminator model summary\n:{discriminator.summary()}')
            predict_real = discriminator(targets)  # should -> [0, 1]
            predict_fake = discriminator(fake_img)  # should -> [1, 0]
            discrim_outputs = tf.stack([predict_real, predict_fake], axis=0,
                                       name='discriminator')

    # Create two copies of the task network, one for real images (targets
    # input to this method) and one for generated images (outputs from
    # generator). The task targets (detected objects) should be the same for
    # both.
    with tf.device(f'/device:GPU:{a.devices[-1]}'):
        with tf.name_scope('task_net'):
            if a.use_yolo:
                task_net, _, _ = build_yolo_model(
                    base_model_name=a.base_model_name,
                    is_recurrent=a.is_recurrent,
                    num_predictor_heads=a.num_pred_layers,
                    max_inferences_per_image=a.max_inferences,
                    max_bbox_overlap=a.max_bbox_overlap,
                    confidence_threshold=a.confidence_threshold,
                )
<<<<<<< HEAD
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
            "input": input.name
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

    examples = load_examples()
    print("examples count = %d" % examples.count)
    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.objects, examples.targets)
    # undo colorization splitting on images that we use for display/output
    if a.lab_colorization:
        if a.which_direction == "AtoB":
            # inputs is brightness, this will be handled fine as a grayscale image
            # need to augment targets and outputs with brightness
            if a.n_channels > 1:
                targets = augment(examples.targets, examples.inputs)
                outputs = augment(model.outputs, examples.inputs)
            # inputs can be deprocessed normally and handled as if they are single channel
            # grayscale images
            inputs = deprocess(examples.inputs)
        elif a.which_direction == "BtoA":
            # inputs will be color channels only, get brightness from targets
            if a.n_channels > 1:
                inputs = augment(examples.inputs, examples.targets)
            targets = deprocess(examples.targets)
            outputs = deprocess(model.outputs)
        else:
            raise Exception("invalid direction")
    else:
        inputs = deprocess(examples.inputs)
        objects = deprocess(examples.objects)
        targets = deprocess(examples.targets)
        outputs = deprocess(model.outputs)
        combined_outputs = deprocess(model.combined_outputs)

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
    with tf.name_scope("convert_objects"):
        converted_objects = convert(objects)
    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)
    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)
    with tf.name_scope("convert_combined_outputs"):
        converted_combined_outputs = convert(combined_outputs)
    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs,
                                dtype=tf.string, name="input_pngs"),
            "objects": tf.map_fn(tf.image.encode_png, converted_objects,
                                 dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets,
                                 dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs,
                                 dtype=tf.string, name="output_pngs"),
            "combined_outputs": tf.map_fn(tf.image.encode_png,
                                          converted_combined_outputs,
                                          dtype=tf.string, name="output_pngs"),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        #summary_inputs = tf.image.convert_image_dtype(converted_inputs,
        #                                              dtype=tf.float32)
        #tf.summary.image("inputs", summary_inputs)
        tf.summary.image("inputs", inputs)
    with tf.name_scope("objects_summary"):
        #summary_inputs = tf.image.convert_image_dtype(converted_inputs,
        #                                              dtype=tf.float32)
        #tf.summary.image("inputs", summary_inputs)
        tf.summary.image("objects", objects)
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
    with tf.name_scope("combined_outputs_summary"):
        #summary_outputs = tf.image.convert_image_dtype(converted_outputs,
        #                                               dtype=tf.float32)
        #tf.summary.image("outputs", summary_outputs)
        tf.summary.image("combined_outputs", combined_outputs)
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
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)
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
                    filesets = save_images(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

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


main()
=======
                task_net = load_yolo_model_weights(task_net,
                                                   a.checkpoint_load_path)
            else:
                task_net = create_task_net(a, input_shape)
            pred_task = task_net(targets)
            pred_task_fake = task_net(fake_img)
            pred_task_noise = task_net(gen_noise)
        print(f'Task Net model summary:\n{task_net.summary()}')
        task_outputs = tf.stack([pred_task, pred_task_fake, pred_task_noise],
                                axis=0,
                                name='task_net')

    model = Model(inputs=[inputs, noise, targets],
                  outputs=[gen_outputs, discrim_outputs, task_outputs])

    # Plot the sub-models and overall model.
    if a.plot_models:
        plot_model(generator, to_file='plots/generator.svg')
        plot_model(task_net, to_file='plots/task_net.svg')
        plot_model(discriminator, to_file='plots/discriminator.svg')
        plot_model(model, to_file='plots/full_model.svg')

    # Return the model. We'll define losses and a training loop back in the
    # main function.
    return model


def main(a):
    # Set the visible devices to those specified:
    physical_devices = tf.config.list_physical_devices('GPU')
    used_devices = [physical_devices[i] for i in a.devices]
    try:
        # Specify enabled GPUs.
        tf.config.set_visible_devices(used_devices, 'GPU')
        logical_devices = tf.config.list_logical_devices('GPU')
        print(f'{len(physical_devices)} physical GPUs,',
              f'{len(logical_devices)} logical GPUs.')
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        print(f'{len(physical_devices)} physical GPUs.',
              'Could not set visible devices!')

    # Set up the summary writer.
    output_path = Path(a.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    writer = tf.summary.create_file_writer(output_path.as_posix())

    # Build data generators.
    train_data, val_data, test_data = load_examples(a)

    # Build the model.
    model = create_model(a, train_data)
    print(f'Overall model summary:\n{model.summary()}')

    # Define model losses and helpers for computing and applying gradients.
    with tf.name_scope("compute_total_loss"):
        def compute_total_loss(model_inputs, model_outputs, step,
                               return_all=False):
            discrim_loss = calc_discriminator_loss(model_inputs,
                                                   model_outputs,
                                                   step)
            gen_loss = calc_generator_loss(model_inputs, model_outputs, step)
            task_loss = calc_task_loss(model_inputs, model_outputs, step)
            total_loss = discrim_loss + gen_loss + task_loss
            tf.summary.scalar(name='total_loss', data=total_loss,
                              step=step)
            if return_all:
                return total_loss, discrim_loss, gen_loss, task_loss
            else:
                return total_loss

    with tf.name_scope('apply_gradients'):
        def compute_apply_gradients(model, data, optimizer_list,
                                    loss_function_list, step):
            """Computes and applies gradients with optional lists of
            optimizers and corresponding loss functions.

            Args:
                model: the TF model to optimize.
                data: data on which to train the model.
                optimizer_list: list of optimizers or single optimizer for
                    full model.
                loss_function_list: list of loss functions or single loss
                    function for full model.
                step: training step.

            """

            if not isinstance(optimizer_list, list):
                optimizer_list = [optimizer_list]
            if not isinstance(loss_function_list, list):
                loss_function_list = [loss_function_list]
            # Parse out the batch data.
            (inputs, noise, targets), (_, a_task_targets, b_task_targets) = \
                data
            # Compute and apply gradients.
            for optimizer, loss_function in zip(optimizer_list,
                                                loss_function_list):
                with tf.GradientTape() as tape:
                    # Run the model.
                    gen_outputs, discrim_outputs, task_outputs = \
                        model([inputs, noise, targets])
                    model_inputs = (inputs, targets, a_task_targets,
                                    b_task_targets, noise)
                    model_outputs = (gen_outputs,
                                     discrim_outputs,
                                     task_outputs)
                    # Compute the loss.
                    loss = loss_function(model_inputs,
                                         model_outputs,
                                         step)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients,
                                              model.trainable_variables))
                # watched_vars = tape.watched_variables()
                # gradients = tape.gradient(loss, watched_vars)
                # optimizer.apply_gradients(zip(gradients, watched_vars))


    with tf.device(f'/device:GPU:{a.devices[0]}'):
        with tf.name_scope("discriminator_loss"):
            def calc_discriminator_loss(model_inputs, model_outputs, step,
                                        **kwargs):
                # minimizing -tf.log will try to get inputs to 1
                # discrim_outputs[0] = predict_real => [0, 1]
                # discrim_outputs[1] = predict_fake => [1, 0]
                discrim_outputs = model_outputs[1]
                predict_real = discrim_outputs[0]
                predict_fake = discrim_outputs[1]
                predict_real = tf.reshape(predict_real, [predict_real.shape[0], -1, 2])
                predict_fake = tf.reshape(predict_fake, [predict_fake.shape[0], -1, 2])
                targets_one = tf.ones(shape=predict_real.shape[:-1])
                targets_zero = tf.zeros(shape=predict_fake.shape[:-1])
                real_loss = tf.math.reduce_mean(
                    categorical_crossentropy(
                        tf.stack([targets_zero, targets_one], axis=-1),
                        predict_real,
                        label_smoothing=0.1,
                    )
                )
                fake_loss = tf.math.reduce_mean(
                    categorical_crossentropy(
                        tf.stack([targets_one, targets_zero], axis=-1),
                        predict_fake,
                        label_smoothing=0.1,
                    )
                )
                discrim_loss = real_loss + fake_loss

                # Write summaries.
                tf.summary.scalar(name='discrim_real_loss', data=real_loss,
                                  step=step)
                tf.summary.scalar(name='discrim_fake_loss', data=fake_loss,
                                  step=step)
                tf.summary.scalar(name='discrim_total_loss',
                                  data=discrim_loss,
                                  step=step)
                return a.dsc_weight * discrim_loss

        with tf.name_scope("generator_loss"):
            def calc_generator_loss(model_inputs, model_outputs, step, **kwargs):
                # predict_fake => [0, 1]
                # abs(targets - outputs) => 0
                fake_img = model_outputs[0][0]
                discrim_fake = model_outputs[1][1]
                discrim_fake = tf.reshape(discrim_fake,
                                          [discrim_fake.shape[0], -1, 2])
                targets_ones = tf.ones(shape=discrim_fake.shape[:-1])
                targets_zeros = tf.zeros(shape=discrim_fake.shape[:-1])
                targets = model_inputs[1]
                gen_loss_GAN = tf.reduce_mean(
                    categorical_crossentropy(
                        tf.stack([targets_zeros, targets_ones], axis=-1),
                        discrim_fake,
                        label_smoothing=0.1,
                    )
                )
                gen_loss_L1 = tf.reduce_mean(mean_absolute_error(targets,
                                                                 fake_img))
                gen_loss = a.gan_weight * gen_loss_GAN + a.l1_weight * gen_loss_L1

                # Write summaries.
                tf.summary.scalar(name='gen_L1_loss', data=gen_loss_L1,
                                  step=step)
                tf.summary.scalar(name='gen_GAN_loss', data=gen_loss_GAN,
                                  step=step)
                tf.summary.scalar(name='gen_total_loss', data=gen_loss,
                                  step=step)
                return a.gen_weight * gen_loss

    with tf.device(f'/device:GPU:{a.devices[-1]}'):
        with tf.name_scope('task_loss'):
            def calc_iou(targets, outputs):
                y_a = tf.maximum(targets[..., 0], outputs[..., 0])
                x_a = tf.maximum(targets[..., 1], outputs[..., 1])
                y_b = tf.minimum(targets[..., 2], outputs[..., 2])
                x_b = tf.minimum(targets[..., 3], outputs[..., 3])
                intersection = tf.maximum(x_b - x_a + 1., 0.) * \
                               tf.maximum(y_b - y_a + 1., 0.)
                target_area = (targets[..., 2] - targets[..., 0] + 1.) * \
                              (targets[..., 3] - targets[..., 1] + 1.)
                output_area = (outputs[..., 2] - outputs[..., 0] + 1.) * \
                              (outputs[..., 3] - outputs[..., 1] + 1.)
                union = target_area + output_area - intersection
                return 1. - intersection / union

            @tf.function
            def calc_task_loss(model_inputs, model_outputs, step):
                # task_targets are [ymin, xmin, ymax, xmax, class]
                # task_outputs are [ymin, xmin, ymax, xmax, *class] where *class
                # is a one-hot encoded score for each class in the dataset for
                # custom detector. For YOLO model, *class is just a scalar class
                # score.
                a_task_targets = model_inputs[2]  # input's objects
                b_task_targets = model_inputs[3]  # target's objects
                task_outputs = model_outputs[2]
                a_target_classes = tf.one_hot(tf.cast(a_task_targets[..., -1],
                                                      tf.int32),
                                              a.num_classes)
                b_target_classes = tf.one_hot(tf.cast(b_task_targets[..., -1],
                                                      tf.int32),
                                              a.num_classes)
                # Create noise target classes (should be no objects).
                targets_ones = tf.ones_like(a_task_targets[..., -1])
                targets_zeros = tf.zeros_like(a_task_targets[..., -1])
                n_target_classes = tf.stack([targets_ones, targets_zeros],
                                            axis=-1)

                # Handle YOLO's class output only being a scalar.
                if a.use_yolo:
                    b_output_class = tf.stack([1. - task_outputs[0][..., -1],
                                               task_outputs[0][..., -1]],
                                              axis=-1)
                    a_output_class = tf.stack([1. - task_outputs[1][..., -1],
                                               task_outputs[1][..., -1]],
                                              axis=-1)
                    n_output_class = tf.stack([1. - task_outputs[2][..., -1],
                                               task_outputs[2][..., -1]],
                                              axis=-1)
                else:
                    b_output_class = task_outputs[0][..., 5:]
                    a_output_class = task_outputs[1][..., 5:]
                    n_output_class = task_outputs[2][..., 5:]
                a_bool_mask = (a_task_targets[..., -1] != 0)
                b_bool_mask = (b_task_targets[..., -1] != 0)
                a_object_target = tf.cast(tf.stack([a_bool_mask,
                                                    tf.logical_not(a_bool_mask)],
                                                   axis=-1),
                                          dtype=tf.int32)
                b_object_target = tf.cast(tf.stack([b_bool_mask,
                                                    tf.logical_not(b_bool_mask)],
                                                   axis=-1),
                                          dtype=tf.int32)

                # Grab/calculate yolo/custom network outputs.
                a_task_wh = a_task_targets[..., 2:4] - a_task_targets[..., :2]
                a_task_xy = a_task_targets[..., :2] + a_task_wh / 2.
                b_task_wh = b_task_targets[..., 2:4] - b_task_targets[..., :2]
                b_task_xy = b_task_targets[..., :2] + b_task_wh / 2.
                a_real_wh = task_outputs[1][..., 2:4] - task_outputs[1][..., :2]
                a_real_xy = task_outputs[1][..., :2] + a_real_wh / 2.
                b_real_wh = task_outputs[0][..., 2:4] - task_outputs[0][..., :2]
                b_real_xy = task_outputs[0][..., :2] + a_real_wh / 2.
                a_iou_outputs = task_outputs[1]
                b_iou_outputs = task_outputs[0]

                # Calculate loss on real images.
                b_xy_loss = tf.reduce_sum(tf.where(
                    b_bool_mask,
                    MSE(b_task_xy, b_real_xy),
                    tf.zeros_like(b_bool_mask, dtype=tf.float32)
                ))
                b_wh_loss = tf.reduce_sum(tf.where(
                    b_bool_mask,
                    MSE(b_task_wh, b_real_wh),
                    tf.zeros_like(b_bool_mask, dtype=tf.float32)
                ))
                b_iou_loss = tf.math.reduce_mean(
                    calc_iou(b_task_targets, b_iou_outputs)
                )
                b_obj_loss = tf.math.reduce_mean(
                    categorical_crossentropy(
                        b_object_target,
                        tf.stack([1. - task_outputs[0][..., 4],
                                  task_outputs[0][..., 4]],
                                 axis=-1),
                        label_smoothing=0.1
                    )
                )
                b_class_loss = tf.math.reduce_mean(
                    categorical_crossentropy(b_target_classes,
                                             b_output_class,
                                             label_smoothing=0.1)
                )
                b_loss = a.xy_weight * b_xy_loss + a.wh_weight * b_wh_loss + \
                         a.iou_weight * b_iou_loss + \
                         a.class_weight * b_class_loss + \
                         a.obj_weight * b_obj_loss

                # Calculate loss on fake images.
                a_xy_loss = tf.reduce_sum(tf.where(
                    a_bool_mask,
                    MSE(a_task_xy, a_real_xy),
                    tf.zeros_like(a_bool_mask, dtype=tf.float32)
                ))
                a_wh_loss = tf.reduce_sum(tf.where(
                    a_bool_mask,
                    MSE(a_task_wh, a_real_wh),
                    tf.zeros_like(a_bool_mask, dtype=tf.float32)
                ))
                a_iou_loss = tf.math.reduce_mean(
                    calc_iou(a_task_targets, a_iou_outputs)
                )
                a_obj_loss = tf.math.reduce_mean(
                    categorical_crossentropy(
                        a_object_target,
                        tf.stack([1. - task_outputs[1][..., 4],
                                  task_outputs[1][..., 4]],
                                 axis=-1),
                        label_smoothing=0.1
                    )
                )
                a_class_loss = tf.math.reduce_mean(
                    categorical_crossentropy(a_target_classes,
                                             a_output_class,
                                             label_smoothing=0.1)
                )
                a_loss = a.xy_weight * a_xy_loss + a.wh_weight * a_wh_loss + \
                         a.iou_weight * a_iou_loss + \
                         a.class_weight * a_class_loss + \
                         a.obj_weight * a_obj_loss

                # Calculate loss on generated noise.
                n_class_loss = tf.math.reduce_sum(
                    categorical_crossentropy(n_target_classes,
                                             n_output_class,
                                             label_smoothing=0.1)
                )
                n_loss = a.class_weight * n_class_loss

                task_loss = a_loss + b_loss + n_loss

                # Write summaries.
                tf.summary.scalar(name='Task B XY Loss', data=b_xy_loss,
                                  step=step)
                tf.summary.scalar(name='Task A XY Loss', data=a_xy_loss,
                                  step=step)
                tf.summary.scalar(name='Task B wh Loss', data=b_wh_loss,
                                  step=step)
                tf.summary.scalar(name='Task A wh Loss', data=a_wh_loss,
                                  step=step)
                tf.summary.scalar(name='Task B IoU Loss', data=b_iou_loss,
                                  step=step)
                tf.summary.scalar(name='Task A IoU Loss', data=a_iou_loss,
                                  step=step)
                tf.summary.scalar(name='Task B Objectness Loss', data=b_obj_loss,
                                  step=step)
                tf.summary.scalar(name='Task A Objectness Loss', data=a_obj_loss,
                                  step=step)
                tf.summary.scalar(name='Task B Class Loss', data=b_class_loss,
                                  step=step)
                tf.summary.scalar(name='Task A Class Loss', data=a_class_loss,
                                  step=step)
                tf.summary.scalar(name='Noise Class Loss', data=n_class_loss,
                                  step=step)
                tf.summary.scalar(name='Total Task B Loss',
                                  data=b_loss,
                                  step=step)
                tf.summary.scalar(name='Total Task A Loss',
                                  data=a_loss,
                                  step=step)
                tf.summary.scalar(name='Total Task Loss', data=task_loss,
                                  step=step)
                return a.task_weight * task_loss

    # Define the optimizer, losses, and weights.
    if a.multi_optim:
        with tf.device(f'/device:GPU:{a.devices[0]}'):
            optimizer_discrim = Adam(learning_rate=a.lr_dsc, amsgrad=a.ams_grad)
            optimizer_gen = Adam(learning_rate=a.lr_gen, amsgrad=a.ams_grad)
        with tf.device(f'/device:GPU:{a.devices[-1]}'):
            optimizer_task = Adam(learning_rate=a.lr_task, amsgrad=a.ams_grad)
        optimizer_list = [optimizer_discrim, optimizer_gen, optimizer_task]
        loss_list = [calc_discriminator_loss, calc_generator_loss, calc_task_loss]
    else:
        optimizer_list = [Adam(learning_rate=a.lr_single, amsgrad=a.ams_grad)]
        loss_list = [compute_total_loss]

    # Train the model.
    batches_seen = tf.Variable(0, dtype=tf.int64)
    with writer.as_default():
        # Create metrics for accumulating validation, test losses.
        mean_total = Mean()
        mean_discrim = Mean()
        mean_gen = Mean()
        mean_task = Mean()
        mean_list = [mean_total, mean_discrim, mean_gen, mean_task]
        tf.config.experimental_functions_run_eagerly()
        for epoch in range(a.max_epochs):
            print(f'Training epoch {epoch+1} of {a.max_epochs}...')
            epoch_start = time.time()

            for batch_num, batch in enumerate(train_data):
                # Save summary images, statistics.
                if batch_num % a.summary_freq == 0:
                    print(f'Writing outputs for epoch {epoch+1}, batch {batch_num}.')
                    (inputs, noise, targets), (_, a_task_targets, b_task_targets) = batch
                    gen_outputs, discrim_outputs, task_outputs = model(
                        [inputs, noise, targets]
                    )
                    model_inputs = (inputs, targets, a_task_targets,
                                    b_task_targets, noise)
                    model_outputs = (gen_outputs, discrim_outputs,
                                     task_outputs)

                    tf.summary.image(
                        name='Fake image',
                        data=gen_outputs[0],
                        step=batches_seen,
                    )
                    tf.summary.image(
                        name='Generated noise',
                        data=gen_outputs[1],
                        step=batches_seen,
                    )
                    tf.summary.image(
                        name='A image',
                        data=inputs,
                        step=batches_seen,
                    )
                    tf.summary.image(
                        name='Input noise',
                        data=noise,
                        step=batches_seen,
                    )
                    tf.summary.image(
                        name='B image',
                        data=targets,
                        step=batches_seen,
                    )
                    tf.summary.image(
                        name='Predict real map',
                        data=tf.expand_dims(discrim_outputs[0][..., 1],
                                            axis=-1),
                        step=batches_seen,
                    )
                    tf.summary.image(
                        name='Predict fake map',
                        data=tf.expand_dims(discrim_outputs[1][..., 0],
                                            axis=-1),
                        step=batches_seen,
                    )

                    # Create object bboxes and summarize task outputs, targets.
                    b_detects = task_outputs[0]
                    a_detects = task_outputs[1]
                    n_detects = task_outputs[2]
                    b_mask = tf.tile(
                        tf.expand_dims(b_detects[..., -1] > a.obj_threshold,
                                       axis=-1),
                        [1, 1, b_detects.shape[-1]]
                    )
                    a_mask = tf.tile(
                        tf.expand_dims(a_detects[..., -1] > a.obj_threshold,
                                       axis=-1),
                        [1, 1, a_detects.shape[-1]]
                    )
                    n_mask = tf.tile(
                        tf.expand_dims(n_detects[..., -1] > a.obj_threshold,
                                       axis=-1),
                        [1, 1, n_detects.shape[-1]]
                    )
                    b_detects = tf.where(b_mask,
                                         b_detects,
                                         tf.zeros_like(b_detects))
                    a_detects = tf.where(a_mask,
                                         a_detects,
                                         tf.zeros_like(a_detects))
                    n_detects = tf.where(n_mask,
                                         n_detects,
                                         tf.zeros_like(n_detects))

                    # Bounding boxes are [ymin, xmin, ymax, xmax]. Need to
                    # calculate that from YOLO.
                    a_true_bboxes = a_task_targets[..., :4]
                    b_true_bboxes = b_task_targets[..., :4]
                    a_fake_bboxes = a_detects[..., :4]
                    b_fake_bboxes = b_detects[..., :4]
                    n_fake_bboxes = n_detects[..., :4]

                    # Add bounding boxes to sample images.
                    target_bboxes = tf.image.draw_bounding_boxes(
                        images=tf.image.grayscale_to_rgb(targets),
                        boxes=b_true_bboxes,
                        colors=np.array([[1., 0., 0.]])
                    )
                    target_bboxes = tf.image.draw_bounding_boxes(
                        images=target_bboxes,
                        boxes=b_fake_bboxes,
                        colors=np.array([[0., 1., 0.]])
                    )
                    generated_bboxes = tf.image.draw_bounding_boxes(
                        images=tf.image.grayscale_to_rgb(gen_outputs[0]),
                        boxes=a_true_bboxes,
                        colors=np.array([[1., 0., 0.]])
                    )
                    generated_bboxes = tf.image.draw_bounding_boxes(
                        images=generated_bboxes,
                        boxes=a_fake_bboxes,
                        colors=np.array([[0., 1., 0.]])
                    )
                    noise_bboxes = tf.image.draw_bounding_boxes(
                        images=tf.image.grayscale_to_rgb(gen_outputs[1]),
                        boxes=n_fake_bboxes,
                        colors=np.array([[0., 1., 0.]])
                    )

                    # Save task outputs.
                    tf.summary.image(
                        name='Task output on A domain',
                        data=generated_bboxes,
                        step=batches_seen,
                    )
                    tf.summary.image(
                        name='Task output on B domain',
                        data=target_bboxes,
                        step=batches_seen,
                    )
                    tf.summary.image(
                        name='Task output on gen noise',
                        data=noise_bboxes,
                        step=batches_seen,
                    )

                    # Compute batch losses.
                    total_loss, discrim_loss, gen_loss, task_loss = \
                        compute_total_loss(model_inputs,
                                           model_outputs,
                                           batches_seen,
                                           return_all=True)
                    print(f'Batch {batch_num} performance\n',
                          f'total loss: {total_loss:.4f}\t',
                          f'discriminator loss: {discrim_loss:.4f}\t',
                          f'generator loss: {gen_loss:.4f}\t',
                          f'task loss: {task_loss:.4f}\t')

                # Update the model.
                compute_apply_gradients(model,
                                        batch,
                                        optimizer_list,
                                        loss_list,
                                        batches_seen)
                batches_seen.assign_add(1)
                writer.flush()

            epoch_time = time.time() - epoch_start
            print(f'Epoch {epoch+1} completed in {epoch_time}.\n')

            # Eval on validation data, save best model, early stopping...
            for m in mean_list:
                m.reset_states()
            for batch in val_data:
                (inputs, noise, targets), (_, a_task_targets, b_task_targets) = batch
                gen_outputs, discrim_outputs, task_outputs = model([inputs,
                                                                    targets])
                model_inputs = (inputs, targets, a_task_targets, b_task_targets)
                model_outputs = (gen_outputs, discrim_outputs, task_outputs)

                total_loss, discrim_loss, gen_loss, task_loss = \
                    compute_total_loss(model_inputs,
                                       model_outputs,
                                       batches_seen,
                                       return_all=True)
                if epoch == 0:
                    min_loss = total_loss
                    epochs_without_improvement = 0
                for m, loss in zip(mean_list, [total_loss,
                                               discrim_loss,
                                               gen_loss,
                                               task_loss]):
                    m.update_states([loss])
            if mean_list[0].result().numpy() <= min_loss \
                and a.output_dir is not None:
                min_loss = mean_list[0].result().numpy()
                print(f'Saving model with total loss {min_loss:.4f} ',
                      f'to {a.output_dir}.')
                model.save(a.output_dir)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            print(f'Epoch {epoch+1} performance\n',
                  f'total loss: {mean_list[0].result().numpy():.4f}\t',
                  f'discriminator loss: {mean_list[1].result().numpy():.4f}\t',
                  f'generator loss: {mean_list[2].result().numpy():.4f}\t',
                  f'task loss: {mean_list[3].result().numpy():.4f}\t')

            # Check for early stopping.
            if epochs_without_improvement >= a.early_stop_patience:
                print(f'{a.early_stop_patience} epochs passed without ',
                      'improvement. Stopping training.')
                break

        # Test the best saved model or current model if not saving.
        if a.output_dir is not None:
            model = load_model(a.output_dir)
        for m in mean_list:
            m.reset_states()
        for batch in test_data:
            (inputs, noise, targets), (_, a_task_targets, b_task_targets) = batch
            gen_outputs, discrim_outputs, task_outputs = model([inputs,
                                                                targets])
            model_inputs = (inputs, targets, a_task_targets, b_task_targets)
            model_outputs = (gen_outputs, discrim_outputs, task_outputs)

            total_loss, discrim_loss, gen_loss, task_loss = \
                compute_total_loss(model_inputs,
                                   model_outputs,
                                   batches_seen,
                                   return_all=True)
            for m, loss in zip(mean_list, [total_loss,
                                           discrim_loss,
                                           gen_loss,
                                           task_loss]):
                m.update_states([loss])
        print(f'Test performance\n',
              f'total loss: {mean_list[0].result().numpy():.4f}\t',
              f'discriminator loss: {mean_list[1].result().numpy():.4f}\t',
              f'generator loss: {mean_list[2].result().numpy():.4f}\t',
              f'task loss: {mean_list[3].result().numpy():.4f}\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dir",
        default=None,
        help="Path to folder containing TFRecords training files."
    )
    parser.add_argument(
        "--valid_dir",
        default=None,
        help="Path to folder containing TFRecords validation files."
    )
    parser.add_argument(
        "--test_dir",
        default=None,
        help="Path to folder containing TFRecords testing files."
    )
    parser.add_argument("--mode", required=True,
                        choices=["train", "test", "export"])
    parser.add_argument("--output_dir", required=True,
                        help="where to put output files")
    parser.add_argument("--tensorboard_dir", default=None,
                        help="Directory where tensorboard files are written.")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="directory with checkpoint to resume training from or use for testing"
    )
    parser.add_argument("--max_steps", type=int, default=0,
                        help="number of training steps (0 to disable)")
    parser.add_argument("--max_epochs", type=int, help="number of training epochs")
    parser.add_argument("--summary_freq", type=int, default=100,
                        help="Update summaries every summary_freq steps")
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
    parser.add_argument("--n_blocks_gen", type=int, default=8,
                        help="Number of ResNet blocks in generator. Must be even.")
    parser.add_argument("--n_layer_dsc", type=int, default=3,
                        help="Number of layers in discriminator.")
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
    parser.add_argument("--lr_single", type=float, default=1e-4,
                        help="initial learning rate for single adam optimizer")
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
    parser.add_argument('--buffer_size', type=int, default=512,
                        help='Buffer size for shuffling input data.')
    parser.add_argument('--spec_norm', default=False, action='store_true',
                        help='Whether to perform spectral normalization.')
    parser.add_argument('--plot_models', default=False, action='store_true',
                        help='Whether to plot model architectures.')
    parser.add_argument('--max_inferences', default=20, type=int,
                        help='Max inferences per image. Default 100.')
    parser.add_argument('--num_pred_layers', default=1, type=int,
                        help='Number of predictor layers to use in network.')
    parser.add_argument('--gen_weight', default=1., type=float,
                        help='Relative weight of generator loss.')
    parser.add_argument('--dsc_weight', default=1., type=float,
                        help='Relative weight of discriminator loss.')
    parser.add_argument('--task_weight', default=1., type=float,
                        help='Relative weight of task loss.')
    parser.add_argument('--xy_weight', default=1., type=float,
                        help='Relative weight of xy task loss component.')
    parser.add_argument('--wh_weight', default=1., type=float,
                        help='Relative weight of wh task loss component.')
    parser.add_argument('--iou_weight', default=1., type=float,
                        help='Relative weight of IoU task loss component.')
    parser.add_argument('--class_weight', default=1., type=float,
                        help='Relative weight of class task loss component.')
    parser.add_argument('--obj_weight', default=1., type=float,
                        help='Relative weight of objectness task loss component.')
    parser.add_argument('--early_stop_patience', default=10, type=int,
                        help='Early stopping patience, epochs. Default 10.')
    parser.add_argument('--multi_optim', default=False, action='store_true',
                        help='Whether to use separate optimizers for each loss.')
    parser.add_argument('--ams_grad', default=False, action='store_true',
                        help='Whether to use AMS Grad variant of Adam optimizer.')
    parser.add_argument('--obj_threshold', type=float, default=0.5,
                        help='Objectness threshold, under which a detection is ignored.')
    parser.add_argument('--use_yolo', default=False, action='store_true',
                        help='Whether to use existing YOLO SatNet for task network.')
    parser.add_argument('--use_sagan', default=False, action='store_true',
                        help='Whether to use self-attending GAN architecture with ResNet blocks.')
    parser.add_argument('--checkpoint_load_path', type=str,
                        default=None,
                        help='Path to the model checkpoint to load.')
    parser.add_argument('--base_model_name', type=str,
                        default="DarkNet",
                        help='The name of the base network to be used.')
    parser.add_argument('--max_bbox_overlap', type=float,
                        default=1.0,
                        help='Maximum amount two inferred boxes can overlap.')
    parser.add_argument('--confidence_threshold', type=float,
                        default=0.0,
                        help='Minimum confidence required to infer a box.')
    parser.add_argument('--is_recurrent', action='store_true',
                        default=False,
                        help='Should we use a recurrent (Convolutional LSTM) '
                             'variant of the model')
    parser.add_argument('--devices', nargs='+', type=int,   
                        help='List of physical devices for TensorFlow to use.')

    # export options
    parser.add_argument("--output_filetype", default="png",
                        choices=["png", "jpeg"])
    args = parser.parse_args()
    main(args)
>>>>>>> release-1.0.0
