"""
Custom callback to save output images to TensorBoard.

Author: Nathan L. Toner
Date Created: 2020-05-14
"""


import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import tensorflow.compat.v1.keras.backend as K
from PIL import Image
from io import BytesIO


class SaveImagesCallback(Callback):
    """Saves model output images to TensorBoard directory.
    
    Args:
        log_dir: directory in which images will be stored.
        wrtier: tf.summary file writer object.
        sess: TensorFlow session for this callback.

    Keyword Args:
        update_freq: frequency with which to update. Default is every batch.
    """

    def __init__(self, log_dir, writer, sess, update_freq=1):
        super().__init__()
        self.log_dir = log_dir
        self.writer = writer
        self.update_freq = update_freq
        self.sess = sess
        self.seen = 0

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.seen += 1
        if self.seen % self.update_freq == 0:
            fake_image = self.model.outputs[0]
            blank_image, target_image = self.model.inputs
            predict_real, predict_fake = self.model.outputs[1]

            # Create image summaries.
            with self.sess.as_default():
                with self.writer.as_default():
                    tf.summary.image(
                        name=f'images/fake_image/{self.seen}',
                        data=fake_image.eval(),
                    )
                    tf.summary.image(
                        name=f'images/blank_image/{self.seen}',
                        data=blank_image.eval(),
                    )
                    tf.summary.image(
                        name=f'images/target_image/{self.seen}',
                        data=target_image.eval(),
                    )
                    tf.summary.image(
                        name=f'images/predict_real/{self.seen}',
                        data=predict_real.eval(),
                    )
                    tf.summary.image(
                        name=f'images/predict_fake/{self.seen}',
                        data=predict_fake.eval(),
                    )
                self.writer.flush()
