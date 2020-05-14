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
    """Saves model output images to TensorBoard directory."""

    def __init__(self, log_dir, update_freq):
        super().__init__()
        self.log_dir = log_dir
        self.update_freq = update_freq
        self.seen = 0
    
    def _encode_image(self, numpy_image):
        """Encode images for saving to TensorBoard."""
        height, width, channels = numpy_image.shape
        image = Image.fromarray(numpy_image)
        output = BytesIO
        image.save(output, format='png')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height, width=width,
                                colorspace=channels,
                                encoded_image_string=image_string)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.seen += 1
        if self.seen % self.update_freq == 0:
            summary_str = []
            fake_image = self.model.outputs['generator']
            blank_image, target_image = self.model.inputs
            predict_real, predict_fake = self.model.outputs['tf_op_layer_discriminator_3']
            summary_str.append(
                tf.Summary.Value(tag=f'images/fake_image/{batch}'),
                                 image=self._encode_image(fake_image)
            )
            summary_str.append(
                tf.Summary.Value(tag=f'images/blank_image/{batch}'),
                                 image=self._encode_image(blank_image)
            )
            summary_str.append(
                tf.Summary.Value(tag=f'images/target_image/{batch}'),
                                 image=self._encode_image(target_image)
            )
            summary_str.append(
                tf.Summary.Value(tag=f'images/predict_real/{batch}'),
                                 image=self._encode_image(predict_real)
            )
            summary_str.append(
                tf.Summary.Value(tag=f'images/predict_fake/{batch}'),
                                 image=self._encode_image(predict_fake)
            )
            self.writer.add_summary(tf.Summary(value=summary_str),
                                    global_step=self.seen)
