import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Flatten, Conv2DTranspose, Dense,
                                     Reshape, Input, BatchNormalization,
                                     UpSampling2D, LeakyReLU, ReLU,
                                     AveragePooling2D, MaxPooling2D)
from .model.SpectralNormalization import SpectralNormalization


##################################################################################
# Layers
##################################################################################
def conv(x, filters, kernel_size=(1, 1), strides=(1, 1), padding=None,
         use_bias=False, sn=False, scope='conv_0'):
    """Defines a convolutional block with optional spectral norm.

    Args:
        x: input to the block.
        filters: number of filters in the block.
    
    Keyword Args:
        kernel_size: kernel size. Default is (1, 1).
        strides: strides. Default is (1, 1).
        padding: mode for padding. Default is None.
        use_bias: whether to use bias. Default is False.
        sn: whether to apply spectral normalization. Default is False.
        scope: scope name for the block. Default is 'conv_0'.

    Returns:
        Convolutional block output.
    """

    with tf.name_scope(scope):
        if sn:
            x = SpectralNormalization(
                Conv2D(filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       use_bias=use_bias,
                       padding=padding,
                       sn=sn)(x)
            )
        else:
            x = Conv2D(filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       use_bias=use_bias,
                       padding=padding,
                       sn=sn)(x)
        return x


def deconv(x, filters, kernel_size=(4, 4), strides=(2, 2), padding='same',
           use_bias=True, sn=False, scope='deconv_0'):
    """Defines a deconvolutional block with optional spectral norm.

    Args:
        x: input to the block.
        filters: number of filters in the block.
    
    Keyword Args:
        kernel_size: kernel size. Default is (4, 4).
        strides: strides. Default is (2, 2).
        padding: mode for padding. Default is 'same'.
        use_bias: whether to use bias. Default is True.
        sn: whether to apply spectral normalization. Default is False.
        scope: scope name for the block. Default is 'deconv_0'.

    Returns:
        Deconvolutional block output.
    """

    with tf.name_scope(scope):
        if sn:
            x = SpectralNormalization(
                Conv2DTranspose(filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                use_bias=use_bias)(x)
            )
        else:
            x = Conv2DTranspose(filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                use_bias=use_bias)(x)
        return x


def fully_connected(x, units, activation=None, use_bias=True, sn=False,
                    scope='linear'):
    """Fully connected layer with optional spectral norm.

    Args:
        units: number of units in FC output.
    
    Keyword Args:
        activation: activation to use. Default is None which applies no
            activation (linear layer).
        use_bias: whether to use bias. Default is True.
        sn: whether to use spectral normalization. Default is False.
        scope: name scope for layer. Default is 'linear'.

    Returns:
        Dense layer output.
    """

    with tf.name_scope(scope):
        if sn:
            x = SpectralNormalization(
                Dense(units, activation=activation, use_bias=use_bias)(x)
            )
        else:
            x = Dense(units, activation=activation, use_bias=use_bias)(x)
        return x


def hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])


##################################################################################
# Residual-blocks
##################################################################################
def up_resblock(x_init, filters, use_bias=True, sn=False, scope='resblock'):
    """Residual block with upsampling.

    Args:
        x_init: input image.
        filters: number of filters in the block.

    Keyword Args:
        use_bias: whether to use bias in res blocks. Default is True.
        sn: whether to use spectral normalization. Default is False.
        scope: name scope in which to create blocks. Default is 'resblock'.

    Returns:
        Residual block output layer.
    """

    with tf.name_scope(scope):
        with tf.name_scope('res1'):
            x = BatchNormalization()(x_init)
            x = LeakyReLU()(x)
            x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
            x = conv(x,
                     filters,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     use_bias=False,
                     sn=sn)

        with tf.name_scope('res2'):
            x = BatchNormalization()(x_init)
            x = LeakyReLU()(x)
            x = conv(x,
                     filters,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     use_bias=use_bias,
                     sn=sn)

        with tf.name_scope('shortcut'):
            x_init = UpSampling2D(size=(2, 2), interpolation='nearest')(x_init)
            x_init = conv(x_init,
                          filters,
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          padding='same',
                          use_bias=False,
                          sn=sn)

        return x + x_init

def down_resblock(x_init, filters, to_down=True, use_bias=True, sn=False, scope='resblock'):
    """Residual block without average pooling.

    Args:
        x_init: input image.
        filters: number of filters in the block.

    Keyword Args:
        use_bias: whether to use bias in res blocks. Default is True.
        sn: whether to use spectral normalization. Default is False.
        scope: name scope in which to create blocks. Default is 'resblock'.

    Returns:
        Residual block output layer.
    """
    with tf.name_scope(scope):
        init_channel = x_init.shape.as_list()[-1]
        with tf.name_scope('res1'):
            x = LeakyReLU()(x_init)
            x = conv(x,
                     filters,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     use_bias=use_bias,
                     sn=sn)

        with tf.name_scope('res2'):
            x = LeakyReLU()(x)
            x = conv(x,
                     filters,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     use_bias=use_bias,
                     sn=sn)

            if to_down :
                x = AveragePooling2D(pool_size=(2, 2), padding='same')(x)

        if to_down or init_channel != filters :
            with tf.name_scope('shortcut'):
                x_init = conv(x_init,
                              filters,
                              kernel_size=(1, 1),
                              strides=(1, 1),
                              use_bias=use_bias,
                              sn=sn)
                if to_down :
                    x_init = AveragePooling2D(pool_size=(2, 2),
                                              padding='same')(x_init)

        return x + x_init

def init_down_resblock(x_init, filters, use_bias=True, sn=False, scope='resblock'):
    """Initial residual block with average pooling.

    Args:
        x_init: input image.
        filters: number of filters in the block.

    Keyword Args:
        use_bias: whether to use bias in res blocks. Default is True.
        sn: whether to use spectral normalization. Default is False.
        scope: name scope in which to create blocks. Default is 'resblock'.

    Returns:
        Residual block output layer.
    """

    with tf.name_scope(scope):
        with tf.name_scope('res1'):
            x = conv(x_init,
                     filters,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     use_bias=use_bias,
                     sn=sn)
            x = LeakyReLU()(x)

        with tf.name_scope('res2'):
            x = conv(x_init,
                     filters,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     use_bias=use_bias,
                     sn=sn)
            x = AveragePooling2D(pool_size=(2, 2), padding='same')(x)

        with tf.name_scope('shortcut'):
            x_init = AveragePooling2D(pool_size=(2, 2), padding='same')(x_init)
            x_init = conv(x_init,
                          filters,
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          use_bias=use_bias,
                          sn=sn)

        return x + x_init


##################################################################################
# Loss functions
##################################################################################