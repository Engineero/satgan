"""Defines loss functions and helpers for calculating SCATGAN gradients."""


import tensorflow as tf
from tensorflow.keras.losses import (mean_absolute_error,
                                     categorical_crossentropy)


# Define model losses and helpers for computing and applying gradients.
def compute_total_loss(a, model_inputs, model_outputs, step,
                       return_all=False, val=False):
    """Computes all component losses and returns total loss.

    Used to compute losses without applying the gradients. Called to apply the
    gradients, but also to record losses when outputting status.

    Args:
        a: argparse argument object from training script.
        model_inputs: tuple of (inputs, targets, a_task_targets,
            b_task_targets, noise).
        model_outputs: tuple of (gen_outputs, discrim_outputs, task_outputs).
        step: a monotonically-increasing training step value.
    
    Keyword Args:
        return_all: if true, return all individual losses, else only the total
            loss. Default is False.
        val: whether this is the validation dataset. Default is False.

    Returns:
        total loss and optionally each component loss for the generator,
            discriminator, and task network.
    """

    with tf.name_scope("compute_total_loss"):
        discrim_loss = calc_discriminator_loss(a, model_inputs,
                                               model_outputs,
                                               step, val=val)
        gen_loss = calc_generator_loss(a, model_inputs, model_outputs, step,
                                       val=val)
        task_loss = calc_task_loss(a, model_inputs, model_outputs, step,
                                   val=val)
        total_loss = discrim_loss + gen_loss + task_loss
        if val:
            tf.summary.scalar(name='total_loss_val', data=total_loss,
                              step=step)
        else:
            tf.summary.scalar(name='total_loss', data=total_loss,
                              step=step)
        if return_all:
            return total_loss, discrim_loss, gen_loss, task_loss
        else:
            return total_loss


def compute_apply_gradients(a, model, a_batch, b_batch, noise, optimizer_list,
                            loss_function_list, step):
    """Computes and applies gradients with optional lists of
    optimizers and corresponding loss functions.

    Args:
        a: arguments passed from training script call.
        model: the TF model to optimize.
        a_batch: source domain data on which to train the model.
        b_batch: target domain data on which to train the model.
        noise: generator input noise for the batch.
        optimizer_list: list of optimizers or single optimizer for
            full model.
        loss_function_list: list of loss functions or single loss
            function for full model.
        step: training step.

    """

    with tf.name_scope('apply_gradients'):
        if not isinstance(optimizer_list, list):
            optimizer_list = [optimizer_list]
        if not isinstance(loss_function_list, list):
            loss_function_list = [loss_function_list]
        # Parse out the batch data.
        inputs, a_task_targets = a_batch
        targets, b_task_targets = b_batch
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
                loss = loss_function(a,
                                     model_inputs,
                                     model_outputs,
                                     step)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,
                                          model.trainable_variables))


def calc_discriminator_loss(a, model_inputs, model_outputs, step,
                            val=False, **kwargs):
    """Calculates the discriminator loss for the GAN.

    Args:
        a: argparse argument object from training script.
        model_inputs: tuple of (inputs, targets, a_task_targets,
            b_task_targets, noise).
        model_outputs: tuple of (gen_outputs, discrim_outputs, task_outputs).
        step: a monotonically-increasing training step value.

    Keyword Args:
        val: whether this is the validation dataset. Default is False.
    
    Returns:
        Total weighted discriminator loss.
    """

    with tf.device(f'/device:GPU:{a.devices[0]}'):
        with tf.name_scope("discriminator_loss"):
            # minimizing -tf.log will try to get inputs to 1
            # discrim_outputs[0] = predict_real => [0, 1]
            # discrim_outputs[1] = predict_fake => [1, 0]
            discrim_outputs = model_outputs[1]
            predict_real = discrim_outputs[0]
            predict_fake = discrim_outputs[1]
            predict_real = tf.reshape(predict_real,
                                      [predict_real.shape[0], -1, 2])
            predict_fake = tf.reshape(predict_fake,
                                      [predict_fake.shape[0], -1, 2])
            targets_one = tf.ones(shape=predict_real.shape[:-1],
                                  dtype=tf.int32)
            targets_zero = tf.zeros(shape=predict_fake.shape[:-1],
                                    dtype=tf.int32)
            real_loss = tf.math.reduce_mean(
                categorical_crossentropy(
                    tf.one_hot(targets_one, a.num_classes),
                    predict_real,
                    label_smoothing=0.1,
                )
            )
            fake_loss = tf.math.reduce_mean(
                categorical_crossentropy(
                    tf.one_hot(targets_zero, a.num_classes),
                    predict_fake,
                    label_smoothing=0.1,
                )
            )
            discrim_loss = real_loss + fake_loss

            # Write summaries.
            if val:
                tf.summary.scalar(name='discrim_real_loss_val',
                                  data=real_loss,
                                  step=step)
                tf.summary.scalar(name='discrim_fake_loss_val',
                                  data=fake_loss,
                                  step=step)
                tf.summary.scalar(name='discrim_total_loss_val',
                                  data=discrim_loss,
                                  step=step)
            else:
                tf.summary.scalar(name='discrim_real_loss', data=real_loss,
                                  step=step)
                tf.summary.scalar(name='discrim_fake_loss', data=fake_loss,
                                  step=step)
                tf.summary.scalar(name='discrim_total_loss',
                                  data=discrim_loss,
                                  step=step)
            return a.dsc_weight * discrim_loss


def calc_generator_loss(a, model_inputs, model_outputs, step,
                        val=False, **kwargs):
    """Calculates the generator loss for the GAN.

    Args:
        a: argparse argument object from training script.
        model_inputs: tuple of (inputs, targets, a_task_targets,
            b_task_targets, noise).
        model_outputs: tuple of (gen_outputs, discrim_outputs, task_outputs).
        step: a monotonically-increasing training step value.

    Keyword Args:
        val: whether this is the validation dataset. Default is False.
    
    Returns:
        Total weighted generator loss.
    """

    with tf.device(f'/device:GPU:{a.devices[0]}'):
        with tf.name_scope("generator_loss"):
            # predict_fake => [0, 1]
            # abs(targets - outputs) => 0
            fake_img = model_outputs[0][0]
            discrim_fake = model_outputs[1][1]
            discrim_fake = tf.reshape(discrim_fake,
                                      [discrim_fake.shape[0], -1, 2])
            targets_ones = tf.ones(shape=discrim_fake.shape[:-1],
                                   dtype=tf.int32)
            targets = model_inputs[1]
            gen_loss_GAN = tf.reduce_mean(
                categorical_crossentropy(
                    tf.one_hot(targets_ones, a.num_classes),
                    discrim_fake,
                    label_smoothing=0.1,
                )
            )
            gen_loss_L1 = tf.reduce_mean(mean_absolute_error(targets,
                                                             fake_img))
            gen_loss = a.gan_weight * gen_loss_GAN + a.l1_weight * gen_loss_L1

            # Write summaries.
            if val:
                tf.summary.scalar(name='gen_L1_loss_val', data=gen_loss_L1,
                                  step=step)
                tf.summary.scalar(name='gen_GAN_loss_val',
                                  data=gen_loss_GAN,
                                  step=step)
                tf.summary.scalar(name='gen_total_loss_val', data=gen_loss,
                                  step=step)
            else:
                tf.summary.scalar(name='gen_L1_loss', data=gen_loss_L1,
                                  step=step)
                tf.summary.scalar(name='gen_GAN_loss', data=gen_loss_GAN,
                                  step=step)
                tf.summary.scalar(name='gen_total_loss', data=gen_loss,
                                  step=step)
            return a.gen_weight * gen_loss


def calc_iou(targets, outputs):
    """Calculates intersection over union (IoU) of bounding boxes.

    Args:
        targets: bounding box targets (truth data).
        outputs: bounding box predictions (network output).
    
    Returns:
        IoU between targets and outputs.
    """

    with tf.name_scope('task_loss'):
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


# @tf.function
def calc_task_loss(a, model_inputs, model_outputs, step, val=False,
                   **kwargs):
    """Calculates the task network loss for the GAN.

    Args:
        a: argparse argument object from training script.
        model_inputs: tuple of (inputs, targets, a_task_targets,
            b_task_targets, noise).
        model_outputs: tuple of (gen_outputs, discrim_outputs, task_outputs).
        step: a monotonically-increasing training step value.

    Keyword Args:
        val: whether this is the validation dataset. Default is False.
    
    Returns:
        Total weighted task network loss.
    """

    with tf.device(f'/device:GPU:{a.devices[-1]}'):
        with tf.name_scope('task_loss'):
            # task_targets are [ymin, xmin, ymax, xmax, class]
            # task_outputs are [ymin, xmin, ymax, xmax, *class] where *class
            # is a one-hot encoded score for each class in the dataset for
            # custom detector.
            a_task_targets = model_inputs[2]  # input's objects
            b_task_targets = model_inputs[3]  # target's objects
            task_outputs = model_outputs[2]
            a_target_classes = tf.one_hot(tf.cast(1 - a_task_targets[..., -1],
                                                  tf.int32),
                                          a.num_classes)
            b_target_classes = tf.one_hot(tf.cast(1 - b_task_targets[..., -1],
                                                  tf.int32),
                                          a.num_classes)
            # Create noise target classes (should be no objects).
            targets_ones = tf.ones_like(a_task_targets[..., -1],
                                        dtype=tf.int32)
            n_target_classes = tf.one_hot(targets_ones, a.num_classes)

            # Grab class outputs.
            b_pred_class = task_outputs[0][..., -a.num_classes:]
            a_pred_class = task_outputs[1][..., -a.num_classes:]
            n_pred_class = task_outputs[2][..., -a.num_classes:]
            a_bool_mask = (a_task_targets[..., -1] > 0)  # true objects
            b_bool_mask = (b_task_targets[..., -1] > 0)

            # Grab/calculate yolo/custom network outputs.
            a_target_wh = a_task_targets[..., 2:4] - a_task_targets[..., :2]
            a_target_xy = a_task_targets[..., :2] + a_target_wh/2.
            b_target_wh = b_task_targets[..., 2:4] - b_task_targets[..., :2]
            b_target_xy = b_task_targets[..., :2] + b_target_wh/2.
            a_pred_wh = task_outputs[1][..., 2:4] - task_outputs[1][..., :2]
            a_pred_xy = task_outputs[1][..., :2] + a_pred_wh/2.
            b_pred_wh = task_outputs[0][..., 2:4] - task_outputs[0][..., :2]
            b_pred_xy = task_outputs[0][..., :2] + b_pred_wh/2.
            a_iou_outputs = task_outputs[1]
            b_iou_outputs = task_outputs[0]

            # Calculate loss on real images.
            b_xy_loss = tf.reduce_sum(tf.where(
                b_bool_mask,
                mean_absolute_error(b_target_xy, b_pred_xy),
                tf.zeros_like(b_bool_mask, dtype=tf.float32)
            ))
            b_wh_loss = tf.reduce_sum(tf.where(
                b_bool_mask,
                mean_absolute_error(b_target_wh, b_pred_wh),
                tf.zeros_like(b_bool_mask, dtype=tf.float32)
            ))
            b_iou_loss = tf.math.reduce_mean(
                calc_iou(b_task_targets, b_iou_outputs)
            )
            b_class_loss = tf.math.reduce_mean(
                categorical_crossentropy(b_target_classes,
                                         b_pred_class,
                                         label_smoothing=0.1)
            )
            b_loss = a.xy_weight * b_xy_loss + a.wh_weight * b_wh_loss + \
                     a.iou_weight * b_iou_loss + \
                     a.class_weight * b_class_loss

            # Calculate loss on fake images.
            a_xy_loss = tf.reduce_sum(tf.where(
                a_bool_mask,
                mean_absolute_error(a_target_xy, a_pred_xy),
                tf.zeros_like(a_bool_mask, dtype=tf.float32)
            ))
            a_wh_loss = tf.reduce_sum(tf.where(
                a_bool_mask,
                mean_absolute_error(a_target_wh, a_pred_wh),
                tf.zeros_like(a_bool_mask, dtype=tf.float32)
            ))
            a_iou_loss = tf.math.reduce_mean(
                calc_iou(a_task_targets, a_iou_outputs)
            )
            a_class_loss = tf.math.reduce_mean(
                categorical_crossentropy(a_target_classes,
                                         a_pred_class,
                                         label_smoothing=0.1)
            )
            a_loss = a.xy_weight * a_xy_loss + a.wh_weight * a_wh_loss + \
                     a.iou_weight * a_iou_loss + \
                     a.class_weight * a_class_loss

            # Calculate loss on generated noise.
            n_class_loss = tf.math.reduce_sum(
                categorical_crossentropy(n_target_classes,
                                         n_pred_class,
                                         label_smoothing=0.1)
            )
            n_loss = a.class_weight * n_class_loss

            task_loss = a_loss + b_loss + n_loss

            # Write summaries.
            if val:
                tf.summary.scalar(name='Task B XY Loss Val',
                                  data=b_xy_loss,
                                  step=step)
                tf.summary.scalar(name='Task A XY Loss Val',
                                  data=a_xy_loss,
                                  step=step)
                tf.summary.scalar(name='Task B wh Loss Val',
                                  data=b_wh_loss,
                                  step=step)
                tf.summary.scalar(name='Task A wh Loss Val',
                                  data=a_wh_loss,
                                  step=step)
                tf.summary.scalar(name='Task B IoU Loss Val',
                                  data=b_iou_loss,
                                  step=step)
                tf.summary.scalar(name='Task A IoU Loss Val',
                                  data=a_iou_loss,
                                  step=step)
                tf.summary.scalar(name='Task B Class Loss Val',
                                  data=b_class_loss,
                                  step=step)
                tf.summary.scalar(name='Task A Class Loss Val',
                                  data=a_class_loss,
                                  step=step)
                tf.summary.scalar(name='Noise Class Loss Val',
                                  data=n_class_loss,
                                  step=step)
                tf.summary.scalar(name='Total Task B Loss Val',
                                  data=b_loss,
                                  step=step)
                tf.summary.scalar(name='Total Task A Loss Val',
                                  data=a_loss,
                                  step=step)
                tf.summary.scalar(name='Total Task Loss Val',
                                  data=task_loss,
                                  step=step)
            else:
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
                tf.summary.scalar(name='Task B Class Loss',
                                  data=b_class_loss,
                                  step=step)
                tf.summary.scalar(name='Task A Class Loss',
                                  data=a_class_loss,
                                  step=step)
                tf.summary.scalar(name='Noise Class Loss',
                                  data=n_class_loss,
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
