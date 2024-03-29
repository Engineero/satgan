"""Utilities for loading and parsing tfrecords files."""


from pathlib import Path
import tensorflow as tf
from miss_data_generator import DatasetGenerator
from yolo_v3.model.yolo_encoder import cast_image_to_float

class GanDataset:
    """Dataset class for GAN.

    Basically just mimics the interface for the MISS DatasetGenerator for
    downstream code consistency.

    Args:
        a: argparse argument structure from training script.
        data_dir: directory for the dataset.

    Keyword Args:
        shuffle (bool): whether to shuffle the dataset. Default is False.
        pad_bboxes (int): number of pixels with which to pad bboxes.
    """

    def __init__(self, a, data_dir, shuffle=False, pad_bboxes=None):
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.pad_bboxes = pad_bboxes
        self.a = a  # args structure
        self.dataset = self._create_dataset()

    @staticmethod
    def _preprocess(image):
        """Performs image standardization, optinoally adds Gaussian noise.

        Args:
            image: the image to transform

        Returns:
            Image shifted to zero mean and unit standard deviation as tf.float32.
        """

        with tf.name_scope("preprocess"):
            image = tf.cast(image, tf.float32)
            image = tf.image.per_image_standardization(image)
            return image

    def _create_dataset(self):
        # Crawl the data directory.
        data_paths = list(Path(self.data_dir).resolve().glob('**/*.tfrecords'))
        if len(data_paths) == 0:
            raise ValueError(
                f"Data directory {self.data_dir} contains no TFRecords files!"
            )
        # Create the dataset.
        data = tf.data.TFRecordDataset(
            filenames=[p.as_posix() for p in data_paths]
        )
        # Specify transformations on datasets.
        if self.shuffle:
            data = data.shuffle(self.a.buffer_size)
        data = data.map(
            lambda x: self._parse_example(x),
            num_parallel_calls=self.a.num_parallel_calls
        )
        data = data.batch(self.a.batch_size, drop_remainder=True)
        data = data.prefetch(buffer_size=self.a.buffer_size)
        return data

    def _parse_example(self, serialized_example):
        """Parses a single TFRecord Example for one domain of the task network.

        Args:
            serialized_example: TFRecord example to load/interpret.
            a: argparse object from training script.

        Keyword Args:
            pad_bboxes (int): Number of pixels with which to pad bboxes.

        Returns:
            Images, optional noise or None, and true bounding boxes with
            classifications (objects) as:
                (image, noise), objects
        """

        # Parse serialized example.
        example = tf.io.parse_single_example(
            serialized_example,
            {
                'images_raw': tf.io.VarLenFeature(dtype=tf.string),
                'filename': tf.io.VarLenFeature(dtype=tf.string),
                'height': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
                'width': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
                'classes': tf.io.VarLenFeature(dtype=tf.int64),
                'ymin': tf.io.VarLenFeature(dtype=tf.float32),
                'ymax': tf.io.VarLenFeature(dtype=tf.float32),
                'ycenter': tf.io.VarLenFeature(dtype=tf.float32),
                'xmin': tf.io.VarLenFeature(dtype=tf.float32),
                'xmax': tf.io.VarLenFeature(dtype=tf.float32),
                'xcenter': tf.io.VarLenFeature(dtype=tf.float32),
                'magnitude': tf.io.VarLenFeature(dtype=tf.float32),
            }
        )

        # Cast parsed objects into usable types.
        width = tf.cast(example['width'], tf.int32)
        height = tf.cast(example['height'], tf.int32)
        classes = tf.cast(tf.sparse.to_dense(example['classes']), tf.float32)

        # Grab bboxes directly from data.
        xmin = tf.cast(tf.sparse.to_dense(example['xmin']), tf.float32)
        xmax = tf.cast(tf.sparse.to_dense(example['xmax']), tf.float32)
        ymin = tf.cast(tf.sparse.to_dense(example['ymin']), tf.float32)
        ymax = tf.cast(tf.sparse.to_dense(example['ymax']), tf.float32)
        if self.pad_bboxes is not None:
            # Pad bounding boxes. SatSim makes really tight bboxes...
            padding = tf.cast(self.pad_bboxes, tf.float32)
            xmin -= padding / tf.cast(width, tf.float32)
            xmax += padding / tf.cast(width, tf.float32)
            ymin -= padding / tf.cast(height, tf.float32)
            ymax += padding / tf.cast(height, tf.float32)

        # Clip bounding boxes to be within the image.
        xmin = tf.clip_by_value(xmin, 0., 1.)
        xmax = tf.clip_by_value(xmax, 0., 1.)
        ymin = tf.clip_by_value(ymin, 0., 1.)
        ymax = tf.clip_by_value(ymax, 0., 1.)

        # Parse images and preprocess.
        image = tf.sparse.to_dense(example['images_raw'], default_value='')
        image = tf.io.decode_raw(image, tf.uint16)
        image = tf.reshape(image, [height, width, self.a.n_channels])

        # Package things up for output.
        objects = tf.stack([ymin, xmin, ymax, xmax, classes], axis=-1)

        # Need to pad objects to max inferences (not all images will have same
        # number of objects).
        paddings = tf.constant([[0, self.a.max_inferences], [0, 0]])
        paddings = paddings - (tf.constant([[0, 1], [0, 0]]) * tf.shape(objects)[0])
        objects = tf.pad(tensor=objects, paddings=paddings, constant_values=0.)
        objects = tf.tile(objects, [self.a.num_pred_layers, 1])

        # Normalize and convert the image, then return (inputs, targets) tuple.
        image = self._preprocess(image)
        return image, objects


def _convert_batches(batch):
    """Converts dtype of batches from MISS DatasetGenerator object.

    Args:
        batch: a batch from the DatasetGenerator.

    Returns:
        (image, bboxes) with image converted to tf.float32.
    """

    image, bboxes, _ = batch
    return tf.cast(image, tf.float32), bboxes


def load_examples(a, data_dir, shuffle=False, pad_bboxes=None, encoder=None):
    """Create dataset pipeline.

    Args:
        a: argparse object from training script.
        data_dir: path to data TFRecords.

    Keyword Args:
        shuffle (bool): whether to shuffle the data.
        pad_bboxes (int): number of pixels with which to pad bboxes.
        encoder: YOLO encoder object defining parse_data method.

    Returns:
        TFRecordDataset of data.
    """

    # Create data queue from training dataset.
    if data_dir is None or not Path(data_dir).resolve().is_dir():
        raise NotADirectoryError(
            f"Data directory {data_dir} does not exist!"
        )

    if encoder is not None:
        # Use MISS DatasetGenerator instead.
        data = DatasetGenerator(
            data_dir,
            parse_function=encoder.parse_data,
            augment=False,
            shuffle=shuffle,
            batch_size=a.batch_size,
            num_threads=a.num_parallel_calls,
            buffer=a.buffer_size,
            encoding_function=cast_image_to_float,
        )
        data.dataset = data.dataset.map(
            lambda im, box, fname:_convert_batches((im, box, fname)),
            num_parallel_calls=a.num_parallel_calls
        )
    else:
        data = GanDataset(a, data_dir, shuffle, pad_bboxes)

    return data
