"""Utilities for loading and parsing tfrecords files."""


from pathlib import Path
import tensorflow as tf


def _preprocess(image, add_noise=False):
    """Performs image standardization, optinoally adds Gaussian noise.

    Args:
        image: the image to transform

    Keyword Args:
        add_noise: whether to add Gaussian noise to the image. Default is
            False.

    Returns:
        Image shifted to zero mean and unit standard deviation with optional
            Gaussian noise added.
    """

    with tf.name_scope("preprocess"):
        result = tf.cast(image, tf.float32)
        result = tf.image.per_image_standardization(result)
        noise = None
        if add_noise:
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0,
                                     stddev=1.0, dtype=tf.float32)
        return result, noise


def _parse_single_domain_example(serialized_example, a, pad_bboxes=False,
                                 add_noise=False):
    """Parses a single TFRecord Example for one domain of the task network."""

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

    if pad_bboxes:
        # Calculate bounding boxes for A images (SatSim makes really tight
        # boxes).
        xcenter = tf.cast(tf.sparse.to_dense(example['xcenter']), tf.float32)
        ycenter = tf.cast(tf.sparse.to_dense(example['ycenter']), tf.float32)
        xmin = xcenter - (10. / tf.cast(width, tf.float32))
        xmax = xcenter + (10. / tf.cast(width, tf.float32))
        ymin = ycenter - (10. / tf.cast(height, tf.float32))
        ymax = ycenter + (10. / tf.cast(height, tf.float32))
    else:
        xmin = tf.cast(tf.sparse.to_dense(example['xmin']), tf.float32)
        xmax = tf.cast(tf.sparse.to_dense(example['xmax']), tf.float32)
        ymin = tf.cast(tf.sparse.to_dense(example['ymin']), tf.float32)
        ymax = tf.cast(tf.sparse.to_dense(example['ymax']), tf.float32)

    # Parse images and preprocess.
    image = tf.sparse.to_dense(example['images_raw'], default_value='')
    image = tf.io.decode_raw(image, tf.uint16)
    image = tf.reshape(image, [height, width, a.n_channels])

    # Package things up for output.
    objects = tf.stack([ymin, xmin, ymax, xmax, classes], axis=-1)

    # Need to pad objects to max inferences (not all images will have same
    # number of objects).
    paddings = tf.constant([[0, a.max_inferences], [0, 0]])
    paddings = paddings - (tf.constant([[0, 1], [0, 0]]) * tf.shape(objects)[0])
    objects = tf.pad(tensor=objects, paddings=paddings, constant_values=0.)
    objects = tf.tile(objects, [a.num_pred_layers, 1])

    image, gen_input = _preprocess(image, add_noise=add_noise)
    return (image, gen_input), objects


def load_examples(a, train_dir, valid_dir, test_dir=None, single_domain=False,
                  pad_bboxes=False, add_noise=False):
    """Create dataset pipelines."""

    # Create data queue from training dataset.
    if train_dir is None or not Path(train_dir).resolve().is_dir():
        raise NotADirectoryError(
            f"Training directory {train_dir} does not exist!"
        )
    train_paths = list(Path(train_dir).resolve().glob('**/*.tfrecords'))
    if len(train_paths) == 0:
        raise ValueError(
            f"Training directory {train_dir} contains no TFRecords files!"
        )
    train_data = tf.data.TFRecordDataset(
        filenames=[p.as_posix() for p in train_paths]
    )

    # Create data queue from validation dataset.
    if valid_dir is None or not Path(valid_dir).resolve().is_dir():
        raise NotADirectoryError(
            f"Validation directory {valid_dir} does not exist!"
        )
    valid_paths = list(Path(valid_dir).resolve().glob('**/*.tfrecords'))
    if len(valid_paths) == 0:
        raise ValueError(
            f"Validation directory {valid_dir} contains no TFRecords files!"
        )
    valid_data = tf.data.TFRecordDataset(
        filenames=[p.as_posix() for p in valid_paths]
    )

    # Create data queue from testing dataset, if given.
    if test_dir is not None:
        if not Path(test_dir).resolve().is_dir():
            raise NotADirectoryError(
                f"Testing directory {test_dir} does not exist!"
            )
        test_paths = list(Path(test_dir).resolve().glob('**/*.tfrecords'))
        if len(test_paths) == 0:
            raise ValueError(
                f"Testing directory {test_dir} contains no TFRecords files!"
            )
        test_data = tf.data.TFRecordDataset(
            filenames=[p.as_posix() for p in test_paths]
        )
    else:
        test_data = None

    # Specify transformations on datasets.
    train_data = train_data.shuffle(a.buffer_size)
    train_data = train_data.map(
        lambda x: _parse_single_domain_example(x, a,
                                               pad_bboxes=pad_bboxes,
                                               add_noise=add_noise)
    )
    train_data = train_data.batch(a.batch_size, drop_remainder=True)

    valid_data = valid_data.map(
        lambda x: _parse_single_domain_example(x, a,
                                               pad_bboxes=pad_bboxes,
                                               add_noise=add_noise)
    )
    valid_data = valid_data.batch(a.batch_size, drop_remainder=True)

    if test_dir is not None:
        test_data = test_data.map(
            lambda x: _parse_single_domain_example(x, a,
                                                   pad_bboxes=pad_bboxes,
                                                   add_noise=add_noise)
        )
        test_data = test_data.batch(a.batch_size, drop_remainder=True)

    return train_data, valid_data, test_data
