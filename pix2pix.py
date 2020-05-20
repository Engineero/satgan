from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
import argparse
import json
import glob
import random
import collections
import math
import time
from pathlib import Path
from utils import ops
from utils.darknet import build_darknet_model
from utils.SaveImagesCallback import SaveImagesCallback
from tensorflow.keras.layers import (Input, Conv2D, Concatenate,
                                     MaxPooling2D, BatchNormalization,
                                     LeakyReLU, GlobalAveragePooling2D)
from tensorflow.keras.activations import tanh
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.losses import (mean_squared_error, mean_absolute_error,
                                     binary_crossentropy)
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, SGD


# Define globals.
EPS = 1e-12


def preprocess(image, add_noise=False):
    """Performs image standardization, optinoally adds Gaussian noise.

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
                                     stddev=0.5, dtype=tf.float32)
            result += noise
        return result


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


def _parse_example(serialized_example, a):
    """Parses a single TFRecord Example for the task network."""
    # Parse serialized example.
    example = tf.io.parse_example(
        serialized_example,
        {
            'a_raw': tf.io.VarLenFeature(dtype=tf.string),
            'b_raw': tf.io.VarLenFeature(dtype=tf.string),
            'filename': tf.io.VarLenFeature(dtype=tf.string),
            'height': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'width': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'classes': tf.io.VarLenFeature(dtype=tf.int64),
            'ymin': tf.io.VarLenFeature(dtype=tf.float32),
            'ymax': tf.io.VarLenFeature(dtype=tf.float32),
            'ycenter': tf.io.VarLenFeature(dtype=tf.float32),
            'xmin': tf.io.VarLenFeature(dtype=tf.float32),
            'xmax': tf.io.VarLenFeature(dtype=tf.float32),
            'xcenter': tf.io.VarLenFeature(dtype=tf.float32),
        }
    )

    # Cast parsed objects into usable types.
    width = tf.cast(example['width'], tf.int32)
    height = tf.cast(example['height'], tf.int32)
    xcenter = tf.cast(tf.sparse.to_dense(example['xcenter']), tf.float32)
    # xmin = tf.cast(tf.sparse.to_dense(example['xmin']), tf.float32)
    # xmax = tf.cast(tf.sparse.to_dense(example['xmax']), tf.float32)
    ycenter = tf.cast(tf.sparse.to_dense(example['ycenter']), tf.float32)
    # ymin = tf.cast(tf.sparse.to_dense(example['ymin']), tf.float32)
    # ymax = tf.cast(tf.sparse.to_dense(example['ymax']), tf.float32)
    # classes = tf.cast(tf.sparse.to_dense(example['classes']), tf.float32)

    # Parse images and preprocess.
    a_image = tf.sparse.to_dense(example['a_raw'], default_value='')
    a_image = tf.io.decode_raw(a_image, tf.uint16)
    a_image = tf.reshape(a_image, [-1, height[0], width[0], 1])
    a_image = preprocess(a_image, add_noise=True)
    b_image = tf.sparse.to_dense(example['b_raw'], default_value='')
    b_image = tf.io.decode_raw(b_image, tf.uint16)
    b_image = tf.reshape(b_image, [-1, height[0], width[0], 1])
    b_image = preprocess(b_image, add_noise=False)

    # Package things up for output.
    objects = tf.stack([xcenter, ycenter], axis=1)
    # Need to pad objects to max inferences (not all images will have same
    # number of objects).
    paddings = tf.constant([[0, 0], [0, 0], [0, a.max_inferences]])
    paddings = paddings - (tf.constant([[0, 0], [0, 0], [0, 1]]) * tf.shape(objects)[-1])
    objects = tf.pad(tensor=objects, paddings=paddings, constant_values=0.0)
    objects = tf.tile(objects, [1, 1, a.num_pred_layers])

    # TODO (NLT): either mask these bboxes to 64x64 images or figure out how to
    # get 100 bboxes per task net output...

    # task_targets = (objects, width, height)
    if a.which_direction == 'AtoB':
        return ((a_image, b_image), (b_image, 0, objects))
    else:
        return ((b_image, a_image), (a_image, 0, objects))


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
    else:
        test_data = None

    # Specify transformations on datasets.
    train_data = train_data.shuffle(a.buffer_size)
    train_data = train_data.batch(a.batch_size, drop_remainder=True)
    train_data = train_data.map(
        lambda x: _parse_example(x, a)
    )
    train_data = train_data.repeat(a.max_epochs)

    valid_data = valid_data.shuffle(a.buffer_size)
    valid_data = valid_data.batch(a.batch_size, drop_remainder=True)
    valid_data = valid_data.map(
        lambda x: _parse_example(x, a)
    )
    valid_data = valid_data.repeat(a.max_epochs)

    if a.test_dir is not None:
        test_data = test_data.shuffle(a.buffer_size)
        test_data = test_data.batch(a.batch_size, drop_remainder=True)
        test_data = test_data.map(
            lambda x: _parse_example(x, a)
        )
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

    x_in = Input(shape=input_shape)
    num_blocks = 8
    num_filters = a.ngf
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.name_scope("generator"):
        skip_layers = []
        x = ops.down_resblock(x_in, filters=num_filters, sn=a.spec_norm,
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
                                sn=a.spec_norm,
                                scope=f'back_up_resblock_{i}')
            num_filters = num_filters * 2

        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = ops.deconv(x, filters=generator_outputs_channels, padding='same',
                       scope='g_logit')
        x = tanh(x)
        return Model(inputs=x_in, outputs=x, name='generator')


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
    input_concat = Concatenate(axis=-1)([x_in, y_in])

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

    return Model(inputs=[x_in, y_in], outputs=x, name='discriminator')


def create_task_net(a, input_shape):
    """Creates the task network.

    Args:
        input_shape: shape of input images.

    Returns:
        Task network (detection) model.
    """

    x_list = []
    y_list = []
    # Feature pyramid network or darknet or something with res blocks.
    model = build_darknet_model(input_shape)
    # Predictor heads for object centroid, width, height.
    for _, output in zip(range(a.num_pred_layers), model.outputs):
        pred_x = Conv2D(
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
        pred_x = GlobalAveragePooling2D()(pred_x)
        x_list.append(pred_x)
        pred_y = Conv2D(
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
        pred_y = GlobalAveragePooling2D()(pred_y)
        y_list.append(pred_y)

    # Combine outputs together.
    if a.num_pred_layers > 1 and len(model.outputs) > 1:
        pred_x = tf.stack(x_list, axis=-1, name='stack_x')
        pred_y = tf.stack(y_list, axis=-1, name='stack_y')
    else:
        pred_x = tf.expand_dims(x_list[0], axis=-1)
        pred_y = tf.expand_dims(y_list[0], axis=-1)
    return Model(inputs=model.input, outputs=[pred_x, pred_y], name='task_net')


def create_model(a, train_data):
    (inputs, targets), (_, _, task_targets) = next(iter(train_data))
    input_shape = inputs.shape.as_list()[1:]  # don't give Input the batch dim
    target_shape = targets.shape.as_list()[1:]
    task_targets_shape = task_targets.shape.as_list()[1:]
    inputs = Input(input_shape)
    targets = Input(target_shape)
    task_targets = Input(task_targets_shape)
    with tf.name_scope("generator"):
        out_channels = target_shape[-1]
        generator = create_generator(a, input_shape, out_channels)
        print(f'Generator model summary:\n{generator.summary()}')
        fake_img = generator(inputs)

    # Create two copies of the task network, one for real images (targets
    # input to this method) and one for generated images (outputs from
    # generator). The task targets (detected objects) should be the same for
    # both.
    with tf.name_scope('task_net'):
        task_net = create_task_net(a, input_shape)
        print(f'Task Net model summary:\n{task_net.summary()}')
        pred_x, pred_y = task_net(targets)
        pred_xy = tf.stack([pred_x, pred_y], axis=1)
        pred_xy = tf.reshape(pred_xy, [a.batch_size, pred_xy.shape[1], -1],
                             name='pred_xy')
        pred_x_fake, pred_y_fake = task_net(fake_img)
        pred_xy_fake = tf.stack([pred_x_fake, pred_y_fake], axis=1)
        pred_xy_fake = tf.reshape(pred_xy_fake, [a.batch_size,
                                                 pred_xy_fake.shape[1],
                                                 -1],
                                  name='pred_xy_fake')
        task_outputs = tf.stack([pred_xy, pred_xy_fake], axis=0,
                                name='task_net')

    # Create two copies of discriminator, one for real pairs and one for fake
    # pairs they share the same underlying variables
    with tf.name_scope("discriminator"):
        # TODO (NLT): figure out discriminator loss, interaction with Keras changes.
        discriminator = create_discriminator(a, input_shape, target_shape)
        print(f'Discriminator model summary\n:{discriminator.summary()}')
        predict_real = discriminator([inputs, targets])
        predict_fake = discriminator([inputs, fake_img])
        discrim_outputs = tf.stack([predict_real, predict_fake], axis=0,
                                   name='discriminator')

    model = Model(inputs=[inputs, targets],
                  outputs=[fake_img, discrim_outputs, task_outputs])

    # Plot the sub-models and overall model.
    if a.plot_models:
        plot_model(generator, to_file='plots/generator.svg')
        plot_model(task_net, to_file='plots/task_net.svg')
        plot_model(discriminator, to_file='plots/discriminator.svg')
        plot_model(model, to_file='plots/full_model.svg')

    # Return the model. We'll define losses and a training loop back in the
    # main function.
    return model


def save_images(a, fetches, step=None):
    image_dir = Path(a.output_dir).resolve() / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name = Path(in_path.decode("utf8")).stem
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = image_dir / filename
            contents = fetches[kind][i]
            with open(out_path.as_posix(), "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(a, filesets, step=False):
    index_path = Path(a.output_dir).resolve() / 'index.html'
    first_line = False
    if not index_path.is_dir():
        first_line = True
    with open(index_path.as_posix(), 'a+') as index:
        if first_line:
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
    # Set up the summary writer.
    output_path = Path(a.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    writer = tf.summary.create_file_writer(output_path.as_posix())

    # Build data generators.
    train_data, val_data, test_data = load_examples(a)

    # Build the model.
    model = create_model(a, train_data)
    print(f'Overall model summary:\n{model.summary()}')

    # Define the optimizer.
    optimizer = Adam()

    # Define model losses and helpers for computing and applying gradients.
    with tf.name_scope("compute_loss"):
        @tf.function
        def compute_loss(model, data, step):
            (inputs, targets), (_, _, task_targets) = data
            fake_img, discrim_outputs, task_outputs = model([inputs, targets])
            discrim_loss = calc_discriminator_loss(discrim_outputs)
            gen_loss = calc_generator_loss(targets, fake_img, discrim_outputs)
            task_loss = calc_task_loss(task_targets, task_outputs)
            total_loss = a.dsc_weight * discrim_loss + \
                a.gen_weight * gen_loss + a.task_weight * task_loss
            tf.summary.scalar(name='discriminator_loss', data=discrim_loss,
                              step=step)
            tf.summary.scalar(name='generator_loss', data=gen_loss,
                              step=step)
            tf.summary.scalar(name='task_loss', data=task_loss,
                              step=step)
            tf.summary.scalar(name='total_loss', data=total_loss,
                              step=step)
            return total_loss
        
    with tf.name_scope('apply_gradients'):
        @tf.function
        def compute_apply_gradients(model, data, optimizer, step):
            with tf.GradientTape() as tape:
                loss = compute_loss(model, data, step)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,
                                          model.trainable_variables))

    with tf.name_scope("discriminator_loss"):
        @tf.function
        def calc_discriminator_loss(discrim_outputs):
            # minimizing -tf.log will try to get inputs to 1
            # discrim_outputs[0] = predict_real => 1
            # discrim_outputs[1] = predict_fake => 0
            real_loss = -tf.math.log(discrim_outputs[0] + EPS)
            fake_loss = -tf.math.log(1 - discrim_outputs[1] + EPS)
            return tf.reduce_mean(real_loss + fake_loss)

    with tf.name_scope("generator_loss"):
        @tf.function
        def calc_generator_loss(targets, fake_img, discrim_outputs):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            gen_loss_GAN = tf.reduce_mean(
                -tf.math.log(discrim_outputs[1] + EPS)
            )
            gen_loss_L1 = tf.reduce_mean(mean_absolute_error(targets,
                                                             fake_img))
            return a.gan_weight * gen_loss_GAN + a.l1_weight * gen_loss_L1

    with tf.name_scope('task_loss'):
        @tf.function
        def calc_task_loss(task_targets, task_outputs):
            # task_targets are [xcenter, ycenter]
            target_sum = tf.math.reduce_sum(tf.math.abs(task_targets), axis=1)
            bool_mask = (target_sum[:, 0] != 0)
            # bool_mask = tf.expand_dims(bool_mask, axis=1)
            print(f'bool mask shape: {bool_mask.shape}')
            print(f'task targets shape: {task_targets.shape}')
            print(f'task outputs shape: {task_outputs.shape}')
            task_targets = tf.boolean_mask(task_targets, bool_mask)
            real_outputs = tf.boolean_mask(task_outputs[0], bool_mask)
            fake_outputs = tf.boolean_mask(task_outputs[1], bool_mask)
            xy_loss = mean_squared_error(real_outputs, task_targets)
            xy_loss_fake = mean_squared_error(fake_outputs, task_targets)
            return tf.reduce_mean(xy_loss + xy_loss_fake)

    # Train the model.
    batches_seen = tf.Variable(0, dtype=tf.int64)
    with writer.as_default():
        for epoch in range(a.max_epochs):
            print(f'Training epoch {epoch+1} of {a.max_epochs}...')
            epoch_start = time.time()
            for batch_num, batch in enumerate(train_data):
                batches_seen.assign_add(1)
                compute_apply_gradients(model, batch, optimizer, batches_seen)
                # Save summary images, statistics.
                if batch_num % a.summary_freq == 0:
                    print(f'Writing outputs for batch {batch_num}.')
                    (inputs, targets), (_, _, task_targets) = batch
                    fake_img, discrim_outputs, task_outputs = model([inputs, targets])
                    tf.summary.image(
                        name='fake_image',
                        data=fake_img,
                        step=batches_seen,
                    )
                    tf.summary.image(
                        name='blank_image',
                        data=batch[0][0],
                        step=batches_seen,
                    )
                    tf.summary.image(
                        name='target_image',
                        data=batch[0][1],
                        step=batches_seen,
                    )
                    tf.summary.image(
                        name='predict_real',
                        data=discrim_outputs[0],
                        step=batches_seen,
                    )
                    tf.summary.image(
                        name='predict_fake',
                        data=discrim_outputs[1],
                        step=batches_seen,
                    )

                    # Create object bboxes and summarize task outputs, targets
                    task_real = task_outputs[0]
                    task_fake = task_outputs[1]
                    true_bboxes = tf.stack([task_targets[:, 1] - 0.02,
                                            task_targets[:, 0] - 0.02,
                                            task_targets[:, 1] + 0.02,
                                            task_targets[:, 0] + 0.02], axis=-1)
                    bboxes_real = tf.stack([task_real[:, 1] - 0.02,
                                            task_real[:, 0] - 0.02,
                                            task_real[:, 1] + 0.02,
                                            task_real[:, 0] + 0.02], axis=-1)
                    bboxes_fake = tf.stack([task_fake[:, 1] - 0.02,
                                            task_fake[:, 0] - 0.02,
                                            task_fake[:, 1] + 0.02,
                                            task_fake[:, 0] + 0.02], axis=-1)
                    target_bboxes = tf.image.draw_bounding_boxes(
                        images=tf.image.grayscale_to_rgb(targets),
                        boxes=bboxes_real,
                        colors=np.array([[0., 1., 0.]])
                    )
                    target_bboxes = tf.image.draw_bounding_boxes(
                        images=target_bboxes,
                        boxes=true_bboxes,
                        colors=np.array([[1., 0., 0.]])
                    )
                    generated_bboxes = tf.image.draw_bounding_boxes(
                        images=tf.image.grayscale_to_rgb(fake_img),
                        boxes=bboxes_fake,
                        colors=np.array([[0., 1., 0.]])
                    )
                    generated_bboxes = tf.image.draw_bounding_boxes(
                        images=generated_bboxes,
                        boxes=true_bboxes,
                        colors=np.array([[1., 0., 0.]])
                    )

                    # Save task outputs.
                    tf.summary.image(
                        name='task_real',
                        data=target_bboxes,
                        step=batches_seen,
                    )
                    tf.summary.image(
                        name='task_fake',
                        data=generated_bboxes,
                        step=batches_seen,
                    )
                writer.flush()

        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch+1} completed in {epoch_time}.')
        # TODO (NLT): test on validation data, save best model, early stopping...

    # TODO (NLT): Test the model.


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
    parser.add_argument('--buffer_size', type=int, default=512,
                        help='Buffer size for shuffling input data.')
    parser.add_argument('--spec_norm', default=False, action='store_true',
                        help='Whether to perform spectral normalization.')
    parser.add_argument('--plot_models', default=False, action='store_true',
                        help='Whether to plot model architectures.')
    parser.add_argument('--max_inferences', default=100, type=int,
                        help='Max inferences per image. Default 100.')
    parser.add_argument('--num_pred_layers', default=1, type=int,
                        help='Number of predictor layers to use in network.')
    parser.add_argument('--gen_weight', default=1., type=float,
                        help='Relative weight of generator loss.')
    parser.add_argument('--dsc_weight', default=1., type=float,
                        help='Relative weight of discriminator loss.')
    parser.add_argument('--task_weight', default=1., type=float,
                        help='Relative weight of task loss.')

    # export options
    parser.add_argument("--output_filetype", default="png",
                        choices=["png", "jpeg"])
    args = parser.parse_args()
    main(args)
