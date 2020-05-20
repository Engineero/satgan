"""
Custom callback to save output images to TensorBoard.

Author: Nathan L. Toner
Date Created: 2020-05-14
"""


import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from PIL import Image
from io import BytesIO


class SaveImagesCallback(Callback):
    """Saves model output images to TensorBoard directory.
    
    Args:
        log_dir: directory in which images will be stored.
        wrtier: tf.summary file writer object.

    Keyword Args:
        update_freq: frequency with which to update. Default is every batch.
    """

    def __init__(self, log_dir, writer, update_freq=1):
        super().__init__()
        self.log_dir = log_dir
        self.writer = writer
        self.update_freq = update_freq
        self.seen = 0

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.seen += 1
        if self.seen % self.update_freq == 0:
            fake_image = self.model.outputs[0]
            blank_image, target_image = self.model.inputs
            predict_real, predict_fake = self.model.outputs[1]
            print(f'fake_image shape: {fake_image.shape}')
            print(f'blank_image shape: {blank_image.shape}')
            print(f'target_image shape: {target_image.shape}')
            print(f'predict_real shape: {predict_real.shape}')
            print(f'predict_fake shape: {predict_fake.shape}')

            # Create image summaries.
            with self.writer.as_default():
                tf.summary.image(
                    name='fake_image',
                    data=tf.cast(fake_image * 255, tf.int32),
                    step=self.seen,
                )
                tf.summary.image(
                    name='blank_image',
                    data=tf.cast(blank_image * 255, tf.int32),
                    step=self.seen,
                )
                tf.summary.image(
                    name='target_image',
                    data=tf.cast(target_image * 255, tf.int32),
                    step=self.seen,
                )
                tf.summary.image(
                    name='predict_real',
                    data=tf.cast(predict_real * 255, tf.int32),
                    step=self.seen,
                )
                tf.summary.image(
                    name='predict_fake',
                    data=tf.cast(predict_fake * 255, tf.int32),
                    step=self.seen,
                )
            self.writer.flush()
