"""Defines the task network."""


from ..utils.darknet import build_darknet_model

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2


# Enable eager execution.
tf.compat.v1.enable_eager_execution()


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

        # Build prediction.
        prediction = tf.concat(
            [pred_xy, pred_wh, pred_conf, pred_class],
            axis=-1
        )
        pred_list.append(prediction)

    # Combine outputs together.
    if a.num_pred_layers > 1 and len(model.outputs) > 1:
        # pred_xy = tf.stack(xy_list, axis=-1, name='stack_xy')
        predict = tf.concat(pred_list, axis=1, name='concat_xy')
    else:
        predict = pred_list[0]
    return Model(inputs=model.input, outputs=predict, name='task_net')

