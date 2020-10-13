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
        image = tf.cast(image, tf.float32)
        result = tf.image.per_image_standardization(image)
        if add_noise:
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0,
                                     stddev=1.0, dtype=tf.float32)
            return (result, noise)
        return result


def _parse_example(serialized_example, a):
    """Parses a single TFRecord Example for the task network."""
    # Parse serialized example.
    example = tf.io.parse_example(
        serialized_example,
        {
            'a_raw': tf.io.VarLenFeature(dtype=tf.string),
            'b_raw': tf.io.VarLenFeature(dtype=tf.string),
            'a_filename': tf.io.VarLenFeature(dtype=tf.string),
            'a_height': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'a_width': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'a_classes': tf.io.VarLenFeature(dtype=tf.int64),
            'a_ymin': tf.io.VarLenFeature(dtype=tf.float32),
            'a_ymax': tf.io.VarLenFeature(dtype=tf.float32),
            'a_ycenter': tf.io.VarLenFeature(dtype=tf.float32),
            'a_xmin': tf.io.VarLenFeature(dtype=tf.float32),
            'a_xmax': tf.io.VarLenFeature(dtype=tf.float32),
            'a_xcenter': tf.io.VarLenFeature(dtype=tf.float32),
            'a_magnitude': tf.io.VarLenFeature(dtype=tf.float32),
            'b_filename': tf.io.VarLenFeature(dtype=tf.string),
            'b_height': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'b_width': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'b_classes': tf.io.VarLenFeature(dtype=tf.int64),
            'b_ymin': tf.io.VarLenFeature(dtype=tf.float32),
            'b_ymax': tf.io.VarLenFeature(dtype=tf.float32),
            'b_ycenter': tf.io.VarLenFeature(dtype=tf.float32),
            'b_xmin': tf.io.VarLenFeature(dtype=tf.float32),
            'b_xmax': tf.io.VarLenFeature(dtype=tf.float32),
            'b_xcenter': tf.io.VarLenFeature(dtype=tf.float32),
            'b_magnitude': tf.io.VarLenFeature(dtype=tf.float32),
        }
    )

    # Cast parsed objects into usable types.
    a_width = tf.cast(example['a_width'], tf.int32)
    a_height = tf.cast(example['a_height'], tf.int32)
    a_xcenter = tf.cast(tf.sparse.to_dense(example['a_xcenter']), tf.float32)
    # a_xmin = tf.cast(tf.sparse.to_dense(example['a_xmin']), tf.float32)
    # a_xmax = tf.cast(tf.sparse.to_dense(example['a_xmax']), tf.float32)
    a_ycenter = tf.cast(tf.sparse.to_dense(example['a_ycenter']), tf.float32)
    # a_ymin = tf.cast(tf.sparse.to_dense(example['a_ymin']), tf.float32)
    # a_ymax = tf.cast(tf.sparse.to_dense(example['a_ymax']), tf.float32)
    a_classes = tf.cast(tf.sparse.to_dense(example['a_classes']), tf.float32)
    b_width = tf.cast(example['b_width'], tf.int32)
    b_height = tf.cast(example['b_height'], tf.int32)
    # b_xcenter = tf.cast(tf.sparse.to_dense(example['b_xcenter']), tf.float32)
    b_xmin = tf.cast(tf.sparse.to_dense(example['b_xmin']), tf.float32)
    b_xmax = tf.cast(tf.sparse.to_dense(example['b_xmax']), tf.float32)
    # b_ycenter = tf.cast(tf.sparse.to_dense(example['b_ycenter']), tf.float32)
    b_ymin = tf.cast(tf.sparse.to_dense(example['b_ymin']), tf.float32)
    b_ymax = tf.cast(tf.sparse.to_dense(example['b_ymax']), tf.float32)
    b_classes = tf.cast(tf.sparse.to_dense(example['b_classes']), tf.float32)

    # Calculate bounding boxes for A images (SatSim makes really tight boxes).
    a_xmin = a_xcenter - 10. / tf.cast(a_width[0], tf.float32)
    a_xmax = a_xcenter + 10. / tf.cast(a_width[0], tf.float32)
    a_ymin = a_ycenter - 10. / tf.cast(a_height[0], tf.float32)
    a_ymax = a_ycenter + 10. / tf.cast(a_height[0], tf.float32)

    # Parse images and preprocess.
    a_image = tf.sparse.to_dense(example['a_raw'], default_value='')
    a_image = tf.io.decode_raw(a_image, tf.uint16)
    a_image = tf.reshape(a_image, [-1, a_height[0], a_width[0], a.n_channels])
    b_image = tf.sparse.to_dense(example['b_raw'], default_value='')
    b_image = tf.io.decode_raw(b_image, tf.uint16)
    b_image = tf.reshape(b_image, [-1, b_height[0], b_width[0], a.n_channels])

    # Package things up for output.
    a_objects = tf.stack([a_ymin, a_xmin, a_ymax, a_xmax, a_classes], axis=-1)
    b_objects = tf.stack([b_ymin, b_xmin, b_ymax, b_xmax, b_classes], axis=-1)

    # Need to pad objects to max inferences (not all images will have same
    # number of objects).
    a_paddings = tf.constant([[0, 0], [0, a.max_inferences], [0, 0]])
    a_paddings = a_paddings - (tf.constant([[0, 0], [0, 1], [0, 0]]) * tf.shape(a_objects)[1])
    a_objects = tf.pad(tensor=a_objects, paddings=a_paddings, constant_values=0.)
    a_objects = tf.tile(a_objects, [1, a.num_pred_layers, 1])
    b_paddings = tf.constant([[0, 0], [0, a.max_inferences], [0, 0]])
    b_paddings = b_paddings - (tf.constant([[0, 0], [0, 1], [0, 0]]) * tf.shape(b_objects)[1])
    b_objects = tf.pad(tensor=b_objects, paddings=b_paddings, constant_values=0.)
    b_objects = tf.tile(b_objects, [1, a.num_pred_layers, 1])

    # task_targets = (objects, width, height)
    if a.which_direction == 'AtoB':
        a_image, gen_input = _preprocess(a_image, add_noise=True)
        b_image = _preprocess(b_image, add_noise=False)
        return ((a_image, gen_input, b_image), (b_image, a_objects, b_objects))
    else:
        b_image, gen_input = _preprocess(b_image, add_noise=True)
        a_image = _preprocess(a_image, add_noise=False)
        return ((b_image, gen_input, a_image), (a_image, b_objects, a_objects))


def load_examples(a):
    # Create data queue from training dataset.
    if a.train_dir is None or not Path(a.train_dir).resolve().is_dir():
        raise NotADirectoryError(
            f"Training directory {a.train_dir} does not exist!"
        )
    train_paths = list(Path(a.train_dir).resolve().glob('**/*.tfrecords'))
    if len(train_paths) == 0:
        raise ValueError(
            f"Training directory {a.input_dir} contains no TFRecords files!"
        )
    train_data = tf.data.TFRecordDataset(
        filenames=[p.as_posix() for p in train_paths]
    )

    # Create data queue from validation dataset.
    if a.valid_dir is None or not Path(a.valid_dir).resolve().is_dir():
        raise NotADirectoryError(
            f"Validation directory {a.valid_dir} does not exist!"
        )
    valid_paths = list(Path(a.valid_dir).resolve().glob('**/*.tfrecords'))
    if len(valid_paths) == 0:
        raise ValueError(
            f"Validation directory {a.valid_dir} contains no TFRecords files!"
        )
    valid_data = tf.data.TFRecordDataset(
        filenames=[p.as_posix() for p in valid_paths]
    )

    # Create data queue from testing dataset, if given.
    if a.test_dir is not None:
        if not Path(a.test_dir).resolve().is_dir():
            raise NotADirectoryError(
                f"Testing directory {a.test_dir} does not exist!"
            )
        test_paths = list(Path(a.test_dir).resolve().glob('**/*.tfrecords'))
        if len(test_paths) == 0:
            raise ValueError(
                f"Testing directory {a.test_dir} contains no TFRecords files!"
            )
        test_data = tf.data.TFRecordDataset(
            filenames=[p.as_posix() for p in test_paths]
        )
    else:
        test_data = None

    # Specify transformations on datasets.
    train_data = train_data.shuffle(a.buffer_size)
    train_data = train_data.batch(a.batch_size, drop_remainder=True)
    train_data = train_data.map(
        lambda x: _parse_example(x, a)
    )

    valid_data = valid_data.batch(a.batch_size, drop_remainder=True)
    valid_data = valid_data.map(
        lambda x: _parse_example(x, a)
    )

    if a.test_dir is not None:
        test_data = test_data.batch(a.batch_size, drop_remainder=True)
        test_data = test_data.map(
            lambda x: _parse_example(x, a)
        )
    return train_data, valid_data, test_data
