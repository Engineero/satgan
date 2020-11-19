"""Defines the discriminator for the GAN."""


from .utils import ops
from .attention import google_attention

import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model


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
                              sn=a.spec_norm, scope='layer_1',
                              activation=a.activation)

        # layer_2: [batch, h//2, w//2, ndf] => [batch, h//4, w//4, ndf * 2]
        # layer_3: [batch, h//4, w//4, ndf * 2] => [batch, h//8, w//8, ndf * 4]
        # layer_4: [batch, h//8, w//8, ndf * 4] => [batch, h//16, w//16, ndf * 8]
        for i in range(a.n_layer_dsc):
            out_channels = a.ndf * min(2**(i+1), 8)
            x = ops.down_resblock(x, filters=out_channels, sn=a.spec_norm,
                                  scope=f'layer_{i+1}',
                                  activation=a.activation)

        # Add self attention layer before final down resblock.
        x = google_attention(x, out_channels, sn=a.spec_norm,
                             scope='discrim_attention')

        # layer_5: [batch, h//16, w//16, ndf * 8] => [batch, h//16-1, w//16-1, 2]
        x = ops.down_resblock(x, filters=2, to_down=False, sn=a.spec_norm,
                              scope=f'layer_{a.n_layer_dsc + 1}',
                              activation=a.activation)
        x = tf.nn.softmax(x, name='discriminator')
    else:
        discrim_conv = lambda x, n, s: ops.conv(x, n, kernel_size=(4, 4),
                                                strides=(s, s), padding='same')

        # Define the activation function to be used.
        if a.activation == 'lrelu':
            activation_fcn = lambda x: LeakyReLU()(x)
        elif a.activation == 'mish':
            from tensorflow_addons.activations import mish
            activation_fcn = lambda x: mish(x)
        else:
            raise ValueError("activation must be 'lrelu' or 'mish'")

        with tf.name_scope('layer_1'):
            x = BatchNormalization()(x_in)
            x = activation_fcn(x)
            x = discrim_conv(x, a.ndf, 2)
        for i in range(a.n_layer_dsc):
            out_channels = a.ndf * min(2**(i+1), 8)
            stride = 1 if i == a.n_layer_dsc - 1 else 2  # last layer stride = 1
            with tf.name_scope(f'layer_{i+2}'):
                x = BatchNormalization()(x)
                x = activation_fcn(x)
                x = discrim_conv(x, out_channels, stride)
        with tf.name_scope('output_layer'):
            x = BatchNormalization()(x)
            x = activation_fcn(x)
            x = discrim_conv(x, 2, 1)
            x = tf.nn.softmax(x, name='discriminator')

    return Model(inputs=x_in, outputs=x, name='discriminator')
