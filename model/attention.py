from .utils import ops

import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D


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