"""Utilities for plotting summary images to Tensorboard."""


import tensorflow as tf
import numpy as np


def image_int_to_float(image):
    """Converts an image to uint16 and then float32 in range [0, 1]."""
    image = tf.cast(image, tf.uint16)
    return tf.image.convert_image_dtype(image, tf.float32)


def plot_summaries(a, model_inputs, model_outputs, batches_seen):
    """Plots summary images to Tensorboard.
    
    Args:
        a: arguments from training/testing script.
        model_inputs: tuple of inputs, targets, a_task_targets,
            b_task_targets, and noise.
        model_outputs: tuple of gen_outputs, discrim_outputs,
            and task_outputs.
        batches_seen: number of batches seen so far.
    """

    inputs, targets, a_task_targets, b_task_targets, noise = model_inputs
    gen_outputs, discrim_outputs, task_outputs = model_outputs

    tf.summary.image(
        name='Fake image',
        data=image_int_to_float(gen_outputs[0],
        step=batches_seen,
    )
    tf.summary.image(
        name='Generated noise',
        data=image_int_to_float(gen_outputs[1]),
        step=batches_seen,
    )
    tf.summary.image(
        name='A image',
        data=image_int_to_float(inputs),
        step=batches_seen,
    )
    tf.summary.image(
        name='Input noise',
        data=image_int_to_float(noise),
        step=batches_seen,
    )
    tf.summary.image(
        name='B image',
        data=image_int_to_float(targets),
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
        tf.expand_dims(b_detects[..., -a.num_classes] > a.obj_threshold,
                       axis=-1),
        [1, 1, b_detects.shape[-1]]
    )
    a_mask = tf.tile(
        tf.expand_dims(a_detects[..., -a.num_classes] > a.obj_threshold,
                       axis=-1),
        [1, 1, a_detects.shape[-1]]
    )
    n_mask = tf.tile(
        tf.expand_dims(n_detects[..., -a.num_classes] > a.obj_threshold,
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
        images=tf.image.grayscale_to_rgb(image_int_to_float(targets)),
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