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

            # Create image summaries.
            with self.writer.as_default():
                tf.summary.image(
                    name=f'fake_image_{self.seen}',
                    data=fake_image,
                    step=self.seen,
                )
                tf.summary.image(
                    name=f'blank_image_{self.seen}',
                    data=blank_image,
                    step=self.seen,
                )
                tf.summary.image(
                    name=f'target_image_{self.seen}',
                    data=target_image,
                    step=self.seen,
                )
                tf.summary.image(
                    name=f'predict_real_{self.seen}',
                    data=predict_real,
                    step=self.seen,
                )
                tf.summary.image(
                    name=f'predict_fake_{self.seen}',
                    data=predict_fake,
                    step=self.seen,
                )
            self.writer.flush()
