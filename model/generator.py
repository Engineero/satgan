from .utils import ops
from .attention import google_attention

import tensorflow as tf
from tensorflow.keras.layers import (Input, Concatenate, BatchNormalization,
                                     Dropout, LeakyReLU)
from tensorflow.keras.activations import tanh
from tensorflow.keras.models import Model
from tensorflow_addons.activations import mish


def create_generator(a, input_shape, generator_outputs_channels):
    """Creates the generator network.

    Args:
        a: command-line arguments object.
        input_shape: shape of the input image.
        generator_output_channels: number of channels generator should output.

    Returns:
        Generator network model.
    """

    # Define the activation function to be used.
    if a.activation == 'lrelu':
        activation_fcn = lambda x: LeakyReLU()(x)
    elif a.activation == 'mish':
        activation_fcn = lambda x: mish(x)
    else:
        raise ValueError("activation must be 'lrelu' or 'mish'")

    x_in = Input(shape=input_shape)
    num_filters = a.ngf
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.name_scope("generator"):
        if a.use_sagan:
            skip_layers = []
            x = ops.down_resblock(x_in, filters=num_filters, sn=a.spec_norm,
                                  scope='front_down_resblock_0',
                                  activation=a.activation)
            skip_layers.append(x)
            for i in range(a.n_blocks_gen // 2):
                num_filters = num_filters * 2
                x = ops.down_resblock(x, filters=num_filters, sn=a.spec_norm,
                                      scope=f'mid_down_resblock_{i}',
                                      activation=a.activation)
                skip_layers.append(x)

            x = google_attention(skip_layers.pop(), filters=num_filters,
                                 scope='self_attention')

            # Build the back end of the generator with skip connections.
            for i in range(a.n_blocks_gen // 2, a.n_blocks_gen):
                if i > a.n_blocks_gen // 2:
                    # No skip connection for first up resblock.
                    x = Concatenate(axis=3)([x, skip_layers.pop()])
                x = ops.up_resblock(x,
                                    filters=num_filters,
                                    sn=a.spec_norm,
                                    scope=f'back_up_resblock_{i}',
                                    activation=a.activation)
                num_filters = num_filters // 2

            x = Concatenate(axis=3)([x, skip_layers.pop()])
            x = BatchNormalization()(x)
            x = activation_fcn(x)
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
                        x = activation_fcn(x)
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
                    x = activation_fcn(x)
                    x = gen_deconv(x, out_channels)
                    if rate > 0.0:
                        x = Dropout(rate)(x)
                    layers.append(x)

            with tf.name_scope('decoder_1'):
                x = Concatenate(axis=3)([layers[-1], layers[0]])
                x = BatchNormalization()(x)
                x = activation_fcn(x)
                x = gen_deconv(x, generator_outputs_channels)
                x = tanh(x)

        return Model(inputs=x_in, outputs=x, name='generator')
