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
from tensorflow.keras.losses import (MSE, mean_absolute_error,
                                     categorical_crossentropy)
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import Mean
from yolo_v3 import build_yolo_model, load_yolo_model_weights


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
    # xcenter = tf.cast(tf.sparse.to_dense(example['xcenter']), tf.float32)
    xmin = tf.cast(tf.sparse.to_dense(example['xmin']), tf.float32)
    xmax = tf.cast(tf.sparse.to_dense(example['xmax']), tf.float32)
    # ycenter = tf.cast(tf.sparse.to_dense(example['ycenter']), tf.float32)
    ymin = tf.cast(tf.sparse.to_dense(example['ymin']), tf.float32)
    ymax = tf.cast(tf.sparse.to_dense(example['ymax']), tf.float32)
    classes = tf.cast(tf.sparse.to_dense(example['classes']), tf.float32)

    # Parse images and preprocess.
    a_image = tf.sparse.to_dense(example['a_raw'], default_value='')
    a_image = tf.io.decode_raw(a_image, tf.uint16)
    a_image = tf.reshape(a_image, [-1, height[0], width[0], 1])
    b_image = tf.sparse.to_dense(example['b_raw'], default_value='')
    b_image = tf.io.decode_raw(b_image, tf.uint16)
    b_image = tf.reshape(b_image, [-1, height[0], width[0], 1])

    # Package things up for output.
    objects = tf.stack([ymin, xmin, ymax, xmax, classes], axis=-1)
    # Need to pad objects to max inferences (not all images will have same
    # number of objects).
    paddings = tf.constant([[0, 0], [0, a.max_inferences], [0, 0]])
    paddings = paddings - (tf.constant([[0, 0], [0, 1], [0, 0]]) * tf.shape(objects)[1])
    objects = tf.pad(tensor=objects, paddings=paddings, constant_values=0.)
    objects = tf.tile(objects, [1, a.num_pred_layers, 1])

    # task_targets = (objects, width, height)
    if a.which_direction == 'AtoB':
        a_image, gen_input = preprocess(a_image, add_noise=True)
        b_image = preprocess(b_image, add_noise=False)
        return ((a_image, gen_input, b_image), (b_image, 0, objects))
    else:
        b_image, gen_input = preprocess(b_image, add_noise=True)
        a_image = preprocess(a_image, add_noise=False)
        return ((b_image, gen_input, a_image), (a_image, 0, objects))


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
    num_filters = a.ngf
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.name_scope("generator"):
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

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    x_in = Input(shape=input_shape)
    y_in = Input(shape=target_shape)
    input_concat = Concatenate(axis=-1)([x_in, y_in])

    # layer_1: [batch, h, w, in_channels * 2] => [batch, h//2, w//2, ndf]
    x = ops.down_resblock(input_concat, filters=a.ndf,
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

    return Model(inputs=[x_in, y_in], outputs=x, name='discriminator')


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

        # # Predict object confidence.
        # pred_conf = Conv2D(
        #     filters=2*a.max_inferences,
        #     kernel_size=1,
        #     strides=1,
        #     padding='same',
        #     kernel_initializer='he_normal',
        #     activation='sigmoid',
        #     kernel_regularizer=l1_l2(a.l1_reg_kernel,
        #                              a.l2_reg_kernel),
        #     bias_regularizer=l1_l2(a.l1_reg_bias,
        #                            a.l2_reg_bias)
        # )(output)
        # pred_conf = GlobalAveragePooling2D()(pred_conf)
        # pred_conf = tf.reshape(pred_conf, (-1, a.max_inferences, 2))
        # pred_conf = tf.nn.softmax(pred_conf)

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
        prediction = tf.concat([pred_xy_min, pred_xy_max, pred_class],
                               axis=-1)
        pred_list.append(prediction)

    # Combine outputs together.
    if a.num_pred_layers > 1 and len(model.outputs) > 1:
        # pred_xy = tf.stack(xy_list, axis=-1, name='stack_xy')
        predict = tf.concat(pred_list, axis=1, name='concat_xy')
    else:
        predict = pred_list[0]
    return Model(inputs=model.input, outputs=predict, name='task_net')


def create_model(a, train_data):
    (inputs, noise, targets), (_, _, task_targets) = next(iter(train_data))
    input_shape = inputs.shape.as_list()[1:]  # don't give Input the batch dim
    noise_shape = noise.shape.as_list()[1:]
    target_shape = targets.shape.as_list()[1:]
    task_targets_shape = task_targets.shape.as_list()[1:]
    inputs = Input(input_shape)
    noise = Input(noise_shape)
    targets = Input(target_shape)
    task_targets = Input(task_targets_shape)
    with tf.name_scope("generator"):
        out_channels = target_shape[-1]
        generator = create_generator(a, input_shape, out_channels)
        print(f'Generator model summary:\n{generator.summary()}')
        gen_noise = generator(noise)
        fake_img = gen_noise + inputs
        gen_outputs = tf.stack([fake_img, gen_noise], axis=0,
                               name='generator')

    # Create two copies of the task network, one for real images (targets
    # input to this method) and one for generated images (outputs from
    # generator). The task targets (detected objects) should be the same for
    # both.
    with tf.name_scope('task_net'):
        if a.use_yolo:
            task_net, task_loss, encoder = build_yolo_model(
                base_model_name=a.base_model_name,
                is_recurrent=a.is_recurrent,
                num_predictor_heads=a.num_pred_layers,
                max_inferences_per_image=a.max_inferences,
                max_bbox_overlap=a.max_bbox_overlap,
                confidence_threshold=a.confidence_threshold,
            )
            task_net = load_yolo_model_weights(task_net,
                                               a.checkpoint_load_path)
            # TODO (NLT): batch task_net inputs using miss data generator, encoder?
            pred_task = task_net(targets)
            pred_task_fake = task_net(fake_img)
        else:
            task_net = create_task_net(a, input_shape)
            pred_task = task_net(targets)
            pred_task_fake = task_net(fake_img)
        print(f'Task Net model summary:\n{task_net.summary()}')
        task_outputs = tf.stack([pred_task, pred_task_fake], axis=0,
                                name='task_net')

    # Create two copies of discriminator, one for real pairs and one for fake
    # pairs they share the same underlying variables
    with tf.name_scope("discriminator"):
        # TODO (NLT): figure out discriminator loss, interaction with Keras changes.
        discriminator = create_discriminator(a, input_shape, target_shape)
        print(f'Discriminator model summary\n:{discriminator.summary()}')
        predict_real = discriminator([inputs, targets])  # should -> [0, 1]
        predict_fake = discriminator([inputs, fake_img])  # should -> [1, 0]
        discrim_outputs = tf.stack([predict_real, predict_fake], axis=0,
                                   name='discriminator')

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
    if a.use_yolo:
        return model, task_loss, encoder
    else:
        return model


def main(a):
    # Set up the summary writer.
    output_path = Path(a.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    writer = tf.summary.create_file_writer(output_path.as_posix())

    # Build data generators.
    train_data, val_data, test_data = load_examples(a)

    # Build the model.
    if a.use_yolo:
        model, task_loss_obj, encoder = create_model(a, train_data)
    else:
        model = create_model(a, train_data)
    print(f'Overall model summary:\n{model.summary()}')

    # Define model losses and helpers for computing and applying gradients.
    with tf.name_scope("compute_total_loss"):
        @tf.function
        def compute_total_loss(model_inputs, model_outputs, step,
                               return_all=False):
            discrim_loss = calc_discriminator_loss(model_inputs,
                                                   model_outputs,
                                                   step)
            gen_loss = calc_generator_loss(model_inputs, model_outputs, step)
            task_loss = calc_task_loss(model_inputs, model_outputs, step)
            total_loss = a.dsc_weight * discrim_loss + \
                a.gen_weight * gen_loss + a.task_weight * task_loss
            tf.summary.scalar(name='total_loss', data=total_loss,
                              step=step)
            if return_all:
                return total_loss, discrim_loss, gen_loss, task_loss
            else:
                return total_loss

    with tf.name_scope('apply_gradients'):
        @tf.function
        def compute_apply_gradients(model, data, optimizer_list,
                                    loss_function_list, step,
                                    loss_weight_list=None):
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

            Keyword Args:
                loss_weight_list: weights associated with loss function.
                    Default is None which applies even weight to all losses.
            """

            if not isinstance(optimizer_list, list):
                optimizer_list = [optimizer_list]
            if not isinstance(loss_function_list, list):
                loss_function_list = [loss_function_list]
            if loss_weight_list is None:
                loss_weight_list = [1.] * len(optimizer_list)
            if not isinstance(loss_weight_list, list):
                loss_weight_list = [loss_weight_list]
            # Parse out the batch data.
            (inputs, noise, targets), (_, _, task_targets) = data
            # Compute and apply gradients.
            for optimizer, loss_function, weight in zip(optimizer_list,
                                                        loss_function_list,
                                                        loss_weight_list):
                with tf.GradientTape() as tape:
                    gen_outputs, discrim_outputs, task_outputs = \
                        model([inputs, noise, targets])
                    model_inputs = (inputs, targets, task_targets, noise)
                    model_outputs = (gen_outputs,
                                     discrim_outputs,
                                     task_outputs)
                    loss = weight * loss_function(model_inputs,
                                                  model_outputs,
                                                  step, encoder=encoder)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    with tf.name_scope("discriminator_loss"):
        @tf.function
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
            return discrim_loss

    with tf.name_scope("generator_loss"):
        @tf.function
        def calc_generator_loss(model_inputs, model_outputs, step, **kwargs):
            # predict_fake => 1
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
            return gen_loss

    with tf.name_scope('task_loss'):
        @tf.function
        def calc_task_loss(model_inputs, model_outputs, step):
            # task_targets are [ymin, xmin, ymax, xmax, class]
            # task_outputs are [ymin, xmin, ymax, xmax, *class] where *class
            # is a one-hot encoded score for each class in the dataset for
            # custom detector. For YOLO model, *class is just a scalar class
            # score.
            task_targets = model_inputs[2]
            task_outputs = model_outputs[2]
            print(f'task targets: {task_targets}')
            print(f'task targets shape: {task_targets.shape}')
            print(f'task outputs: {task_outputs}')
            print(f'task outputs shape: {task_outputs.shape}')

            if a.use_yolo:
                _, task_targets_enc = encoder.encode_for_yolo(
                    None,
                    tf.reshape(task_targets,
                               [-1, task_targets.shape[-1]]),
                    None
                )
                _, real_task_outputs_enc = encoder.encode_for_yolo(
                    None,
                    tf.reshape(task_outputs[0],
                               [-1, task_outputs[0].shape[-1]]),
                    None
                )
                _, fake_task_outputs_enc = encoder.encode_for_yolo(
                    None,
                    tf.reshape(task_outputs[1],
                               [-1, task_outputs[1].shape[-1]]),
                    None
                )
                real_loss = task_loss_obj.compute_loss(task_targets_enc,
                                                       real_task_outputs_enc)
                fake_loss = task_loss_obj.compute_loss(task_targets,
                                                       fake_task_outputs_enc)
            else:
                target_classes = tf.one_hot(tf.cast(task_targets[..., -1],
                                                    tf.int32),
                                            a.num_classes)
                # target_objects = tf.one_hot(tf.cast(task_targets[..., -2],
                #                                     tf.int32),
                #                             2)
                bool_mask = (task_targets[..., -1] != 0)

                # TODO (NLT): Calculate intersection and union.
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

                # Calculate loss on real images.
                task_wh = task_targets[..., 2:4] - task_targets[..., :2]
                task_xy = task_targets[..., :2] + task_wh / 2.
                xy_loss = tf.reduce_sum(tf.where(
                    bool_mask,
                    MSE(task_xy, task_outputs[0][..., :2]),
                    tf.zeros_like(bool_mask, dtype=tf.float32)
                ))
                iou_loss = tf.math.reduce_mean(
                    calc_iou(task_targets, task_outputs[0])
                )
                # obj_loss = tf.math.reduce_mean(
                #     categorical_crossentropy(target_objects,
                #                              task_outputs[0][..., 4:6],
                #                              label_smoothing=0.1)
                # )
                class_loss = tf.math.reduce_mean(
                    categorical_crossentropy(target_classes,
                                             task_outputs[0][..., 4:],
                                             label_smoothing=0.1)
                )
                real_loss = xy_loss + a.iou_weight * iou_loss + \
                            a.class_weight * class_loss

                # Calculate loss on fake images.
                xy_loss_fake = tf.reduce_sum(tf.where(
                    bool_mask,
                    MSE(task_xy, task_outputs[1][..., :2]),
                    tf.zeros_like(bool_mask, dtype=tf.float32)
                ))
                iou_loss_fake = tf.math.reduce_mean(
                    calc_iou(task_targets, task_outputs[1])
                )
                # obj_loss_fake = tf.math.reduce_mean(
                #     categorical_crossentropy(target_objects,
                #                              task_outputs[1][..., 4:6],
                #                              label_smoothing=0.1)
                # )
                class_loss_fake = tf.math.reduce_mean(
                    categorical_crossentropy(target_classes,
                                             task_outputs[1][..., 4:],
                                             label_smoothing=0.1)
                )
                fake_loss = xy_loss_fake + a.iou_weight * iou_loss_fake + \
                            a.class_weight * class_loss_fake

                # Write summaries.
                tf.summary.scalar(name='task_real_xy_loss', data=xy_loss,
                                  step=step)
                tf.summary.scalar(name='task_fake_xy_loss', data=xy_loss_fake,
                                  step=step)
                tf.summary.scalar(name='task_real_iou_loss', data=iou_loss,
                                  step=step)
                tf.summary.scalar(name='task_fake_iou_loss', data=iou_loss_fake,
                                  step=step)
                # tf.summary.scalar(name='task_real_obj_loss', data=obj_loss,
                #                   step=step)
                # tf.summary.scalar(name='task_fake_obj_loss', data=obj_loss_fake,
                #                   step=step)
                tf.summary.scalar(name='task_real_class_loss', data=class_loss,
                                  step=step)
                tf.summary.scalar(name='task_fake_class_loss',
                                  data=class_loss_fake,
                                  step=step)
            
            task_loss = real_loss + fake_loss
            tf.summary.scalar(name='task_real_loss',
                              data=real_loss,
                              step=step)
            tf.summary.scalar(name='task_fake_loss',
                              data=fake_loss,
                              step=step)
            tf.summary.scalar(name='task_loss', data=task_loss,
                              step=step)
            return task_loss


    # Define the optimizer, losses, and weights.
    if a.multi_optim:
        optimizer_gen = Adam(learning_rate=a.lr_gen, amsgrad=a.ams_grad)
        optimizer_discrim = Adam(learning_rate=a.lr_dsc, amsgrad=a.ams_grad)
        optimizer_task = Adam(learning_rate=a.lr_task, amsgrad=a.ams_grad)
        optimizer_list = [optimizer_gen, optimizer_discrim, optimizer_task]
        loss_list = [calc_generator_loss, calc_discriminator_loss, calc_task_loss]
        loss_weights = [a.gen_weight, a.dsc_weight, a.task_weight]
    else:
        optimizer_list = [Adam(learning_rate=1e-4, amsgrad=a.ams_grad)]
        loss_list = [compute_total_loss]
        loss_weights = None

    # Train the model.
    batches_seen = tf.Variable(0, dtype=tf.int64)
    with writer.as_default():
        # Create metrics for accumulating validation, test losses.
        mean_total = Mean()
        mean_discrim = Mean()
        mean_gen = Mean()
        mean_task = Mean()
        mean_list = [mean_total, mean_discrim, mean_gen, mean_task]
        for epoch in range(a.max_epochs):
            print(f'Training epoch {epoch+1} of {a.max_epochs}...')
            epoch_start = time.time()

            for batch_num, batch in enumerate(train_data):
                # Save summary images, statistics.
                if batch_num % a.summary_freq == 0:
                    print(f'Writing outputs for epoch {epoch+1}, batch {batch_num}.')
                    (inputs, noise, targets), (_, _, task_targets) = batch
                    if a.use_yolo:
                        targets_enc, task_targets_enc = encoder.encode_for_yolo(
                            inputs,
                            tf.reshape(task_targets,
                                       [-1, task_targets.shape[-1]]),
                            None
                        )
                        gen_outputs, discrim_outputs, task_outputs = model(
                            [inputs, noise, targets_enc]
                        )
                        # gen_outputs, task_outputs = encoder.encode_for_yolo(
                        #     gen_outputs,
                        #     tf.reshape(task_outputs, [-1,
                        #                               task_outputs.shape[-1]]),
                        #     None
                        # )
                        model_inputs = (inputs, targets_enc, task_targets_enc,
                                        noise)
                    else:
                        gen_outputs, discrim_outputs, task_outputs = model(
                            [inputs, noise, targets]
                        )
                        model_inputs = (inputs, targets, task_targets, noise)
                    model_outputs = (gen_outputs, discrim_outputs,
                                     task_outputs)
                    tf.summary.image(
                        name='fake_image',
                        data=gen_outputs[0],
                        step=batches_seen,
                    )
                    tf.summary.image(
                        name='generated_noise',
                        data=gen_outputs[1],
                        step=batches_seen,
                    )
                    tf.summary.image(
                        name='blank_image',
                        data=inputs,
                        step=batches_seen,
                    )
                    tf.summary.image(
                        name='input_noise',
                        data=noise,
                        step=batches_seen,
                    )
                    tf.summary.image(
                        name='target_image',
                        data=targets,
                        step=batches_seen,
                    )
                    tf.summary.image(
                        name='predict_real',
                        data=tf.expand_dims(discrim_outputs[0][..., 1],
                                            axis=-1),
                        step=batches_seen,
                    )
                    tf.summary.image(
                        name='predict_fake',
                        data=tf.expand_dims(discrim_outputs[1][..., 0],
                                            axis=-1),
                        step=batches_seen,
                    )

                    # Create object bboxes and summarize task outputs, targets.
                    real_detects = task_outputs[0]
                    fake_detects = task_outputs[1]
                    # true_detects = task_targets
                    real_mask = tf.tile(
                        tf.expand_dims(real_detects[..., -1] > a.obj_threshold,
                                       axis=-1),
                        [1, 1, real_detects.shape[-1]]
                    )
                    fake_mask = tf.tile(
                        tf.expand_dims(fake_detects[..., -1] > a.obj_threshold,
                                       axis=-1),
                        [1, 1, real_detects.shape[-1]]
                    )
                    real_detects = tf.where(real_mask,
                                            real_detects,
                                            tf.zeros_like(real_detects))
                    fake_detects = tf.where(fake_mask,
                                            fake_detects,
                                            tf.zeros_like(fake_detects))

                    # Bounding boxes are [ymin, xmin, ymax, xmax].
                    true_bboxes = task_targets[..., :4]
                    bboxes_real = real_detects[..., :4]
                    bboxes_fake = fake_detects[..., :4]

                    # Add bounding boxes to sample images.
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
                        images=tf.image.grayscale_to_rgb(gen_outputs[0]),
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
                                        batches_seen,
                                        loss_weight_list=loss_weights)
                batches_seen.assign_add(1)
                writer.flush()

            epoch_time = time.time() - epoch_start
            print(f'Epoch {epoch+1} completed in {epoch_time}.\n')

            # Eval on validation data, save best model, early stopping...
            for m in mean_list:
                m.reset_states()
            for batch in val_data:
                (inputs, noise, targets), (_, _, task_targets) = batch
                gen_outputs, discrim_outputs, task_outputs = model([inputs,
                                                                    targets])
                model_inputs = (inputs, targets, task_targets)
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
            (inputs, noise, targets), (_, _, task_targets) = batch
            gen_outputs, discrim_outputs, task_outputs = model([inputs,
                                                                targets])
            model_inputs = (inputs, targets, task_targets)
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
    parser.add_argument('--iou_weight', default=1., type=float,
                        help='Relative weight of IoU task loss component.')
    parser.add_argument('--class_weight', default=1., type=float,
                        help='Relative weight of class task loss component.')
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

    # export options
    parser.add_argument("--output_filetype", default="png",
                        choices=["png", "jpeg"])
    args = parser.parse_args()
    main(args)
