"""Creates TFRecords files for pix2pix task network.

Created: 2020-04-21
Author: Nathan L. Toner
"""


import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow_addons.activations import mish
import numpy as np
import argparse
import json
from pathlib import Path
from astropy.io import fits
from itertools import zip_longest
from tqdm import tqdm


def _check_args(args):
    """Checks whether arguments are valid.

    Creates the output directory if it does not exist.

    Raises:
        NotADirectoryError if any input directory is None or invalid.
        ValueError if output_dir is *not* empty.
        ValueError if any other directory *is* empty.
    """
    a_dir = Path(args.a_dir).resolve()
    if args.b_dir is not None:
        b_dir = Path(args.b_dir).resolve()
    elif args.generator_path is None:
        raise ValueError(
            'Either B domain data or generator must be provided!'
        )
    else:
        b_dir = None
    if args.generator_path is not None:
        generator_path = Path(args.generator_path).resolve()
    else:
        generator_path = None
    if args.a_annotation_dir is not None:
        a_annotation_dir = Path(args.a_annotation_dir).resolve()
    else:
        a_annotation_dir = None
    if args.b_annotation_dir is not None:
        b_annotation_dir = Path(args.b_annotation_dir).resolve()
    else:
        b_annotation_dir = None
    output_dir = Path(args.output_dir).resolve()
    if not a_dir.is_dir():
        raise NotADirectoryError(f'Path A {args.a_dir} is not a directory!')
    if b_dir is not None and not b_dir.is_dir():
        raise NotADirectoryError(f'Path B {args.b_dir} is not a directory!')
    if a_annotation_dir is None and b_annotation_dir is None:
        raise ValueError(
            'Either A or B annotation directory (or both) must be given!'
        )
    if a_annotation_dir is not None and not a_annotation_dir.is_dir():
        raise NotADirectoryError(
            f'Annotation path A {args.a_annotation_dir} is not a directory!'
        )
    if b_annotation_dir is not None and not b_annotation_dir.is_dir():
        raise NotADirectoryError(
            f'Annotation path A {args.b_annotation_dir} is not a directory!'
        )
    if generator_path is not None and not generator_path.is_dir():
        raise NotADirectoryError(
            f'Generator path {args.generator_path} is not a directory!'
        )
    if output_dir.is_dir():
        try:
            output_dir.rmdir()
        except:
            raise ValueError('Output directory already exists!')
    else:
        print(f'Making output directory {output_dir}...')
        output_dir.mkdir(parents=True)
    path_list = [a_dir, b_dir, a_annotation_dir, b_annotation_dir]
    for path in path_list:
        if path is not None and not path.glob('*'):
            raise ValueError(f'Path {path} is empty!')


def _group_list(ungrouped_list, group_size, padding=None):
    """Groups the input list into lists of specified size.

    Args:
        ungrouped_list: list to be grouped.
        group_size: size of groups to make.

    Keyword Args:
        padding: value with which to pad last group if not enough values to
            fill. Default is None.

    Returns:
        List of lists formed by grouping items in the input list into
            sub-lists of size group_size.
    """
    grouped_list = zip_longest(*[iter(ungrouped_list)] * group_size,
                               fillvalue=padding)
    num_groups = len(ungrouped_list) // group_size
    return grouped_list, num_groups


# The following functions can be used to convert a value to a type compatible
# with tf.Example.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
      value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, list):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    if isinstance(value, list):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    else:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if isinstance(value, list):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _read_fits(path):
    """Reads simple 1-hdu FITS file into a numpy array.

    Args:
        path: path to the FITS file.

    Returns:
        Image described by FITS file as a numpy array.
    """
    image = fits.getdata(path)
    image = image.astype(np.uint16)
    return image


def _partition_examples(examples, splits_dict):
    """Splits examples according to a split dictionary."""
    # Create a dict to hold examples.
    partitions = dict()
    # Store the total number of examples.
    num_examples = len(examples)
    # Iterate over the items specifying the partitions.
    for (split_name, split_fraction) in splits_dict.items():
        # Compute the size of this parition.
        num_split_examples = int(split_fraction * num_examples)
        # Pop the next partition elements.
        partition_examples = examples[:num_split_examples]
        examples = examples[num_split_examples:]
        # Map this paritions list of examples to this parition name.
        partitions[split_name] = partition_examples
    return partitions


def _parse_annotations(annotations, pad_amount=0., skip_empty=False):
    """Parses annotation file data into features.

    Args:
        annotations: annotation file contents.

    Keyword Args:
        pad_amount: padding amount used for satsim offsets. Default is 0.
        skip_empty: whether to skip empty frames. Default is False.

    Returns:
        Tuple of (class_id, y_min, y_max, y_center, x_min, x_max, x_center,
            magnitude, path_name)
    """

    class_id = [obj['class_id'] for obj in annotations['objects']]
    y_min = [obj['y_min'] - pad_amount for obj in annotations['objects']]
    y_max = [obj['y_max'] + pad_amount for obj in annotations['objects']]
    y_center = [obj['y_center'] for obj in annotations['objects']]
    x_min = [obj['x_min'] - pad_amount for obj in annotations['objects']]
    x_max = [obj['x_max'] + pad_amount for obj in annotations['objects']]
    x_center = [obj['x_center'] for obj in annotations['objects']]
    magnitude = [obj['magnitude'] for obj in annotations['objects']]
    dir_name = annotations['file']['dirname']
    file_name = annotations['file']['filename']
    path_name = (Path(dir_name) / Path(file_name)).as_posix()
    if skip_empty and not x_center:
        return None
    # Replace the unknown magnitude's with NaN's
    for i in range(len(magnitude)):
        if magnitude[i] is None:
            magnitude[i] = float('NaN')
    return (class_id, y_min, y_max, y_center, x_min, x_max, x_center,
            magnitude, path_name)


def _serialize_example(example, pad_for_satsim=False, skip_empty=False):
    """Builds a TFRecords Example object from the example data.

    Args:
        example: example structure with (a_path, b_path, annotation_path).

    Keyword Args:
        pad_for_satsim: whether to pad images for satsim. Default is False.
        skip_empty: whether to skip frames with no object. Default is False.
    """

    # Handle satsim offsets.
    if pad_for_satsim:
        pad_amount = 0.02
    else:
        pad_amount = 0.00

    # Load annotation data.
    (a_path, b_path, a_annotation, b_annotation) = example
    with open(a_annotation, 'r') as fp:
        a_annotations = json.load(fp)['data']
    with open(b_annotation, 'r') as fp:
        b_annotations = json.load(fp)['data']

    # Parse annotation data.
    (a_class_id, a_y_min, a_y_max, a_y_center, a_x_min, a_x_max, a_x_center, a_magnitude,
     a_path_name) = _parse_annotations(a_annotations, pad_amount, skip_empty)
    (b_class_id, b_y_min, b_y_max, b_y_center, b_x_min, b_x_max, b_x_center, b_magnitude,
     b_path_name) = _parse_annotations(b_annotations, pad_amount, skip_empty)

    # Load raw image data.
    a_data = _read_fits(a_path)
    b_data = _read_fits(b_path)

    # Create the features for this example
    features = {
        'a_raw': _bytes_feature([a_data.tostring()]),
        'b_raw': _bytes_feature([b_data.tostring()]),
        'a_filename': _bytes_feature([a_path_name.encode()]),
        'a_height': _int64_feature([a_annotations['sensor']['height']]),
        'a_width': _int64_feature([a_annotations['sensor']['width']]),
        'a_classes': _int64_feature(a_class_id),
        'a_ymin': _float_feature(a_y_min),
        'a_ymax': _float_feature(a_y_max),
        'a_ycenter': _float_feature(a_y_center),
        'a_xmin': _float_feature(a_x_min),
        'a_xmax': _float_feature(a_x_max),
        'a_xcenter': _float_feature(a_x_center),
        'a_magnitude': _float_feature(a_magnitude),
        'b_filename': _bytes_feature([b_path_name.encode()]),
        'b_height': _int64_feature([b_annotations['sensor']['height']]),
        'b_width': _int64_feature([b_annotations['sensor']['width']]),
        'b_classes': _int64_feature(b_class_id),
        'b_ymin': _float_feature(b_y_min),
        'b_ymax': _float_feature(b_y_max),
        'b_ycenter': _float_feature(b_y_center),
        'b_xmin': _float_feature(b_x_min),
        'b_xmax': _float_feature(b_x_max),
        'b_xcenter': _float_feature(b_x_center),
        'b_magnitude': _float_feature(b_magnitude),
    }
    # Create an example protocol buffer
    return tf.train.Example(features=tf.train.Features(feature=features))


def make_tf_records(args):
    """Make TFRecords files from images and annotations."""
    a_dir = Path(args.a_dir).resolve()
    b_dir = Path(args.b_dir).resolve()
    if args.a_annotation_dir is not None:
        a_annotation_dir = Path(args.a_annotation_dir).resolve()
    else:
        a_annotation_dir = None
    if args.b_annotation_dir is not None:
        b_annotation_dir = Path(args.b_annotation_dir).resolve()
    else:
        b_annotation_dir = None
    output_dir = Path(args.output_dir).resolve()
    a_paths = sorted(a_dir.glob('**/*.fits'))
    b_paths = sorted(b_dir.glob('**/*.fits'))
    if a_annotation_dir is not None:
        a_annotation_paths = sorted(a_annotation_dir.glob('**/Annotations/*.json'))
        if b_annotation_dir is None:
            b_annotation_paths = a_annotation_paths
    if b_annotation_dir is not None:
        b_annotation_paths = sorted(b_annotation_dir.glob('**/Annotations/*.json'))
        if a_annotation_dir is None:
            a_annotation_paths = b_annotation_paths
    examples = []
    for a_path, b_path, a_annotation, b_annotation in zip(a_paths, b_paths,
                                                          a_annotation_paths,
                                                          b_annotation_paths):
        examples.append((a_path, b_path, a_annotation, b_annotation))
    splits_dict = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
    partitions = _partition_examples(examples, splits_dict)
    # Make TFRecords from partitions.
    for name, examples in partitions.items():
        print(f'Writing partition "{name}" with {len(examples)} examples...')
        partition_dir = output_dir / name
        partition_dir.mkdir(parents=True)
        groups, num_groups = _group_list(examples, args.group_size)
        for i, example_group in tqdm(enumerate(groups), total=num_groups):
            tfrecords_name = f'{args.output_name}_{name}_{i}.tfrecords'
            output_path = partition_dir / tfrecords_name
            with tf.io.TFRecordWriter(output_path.as_posix()) as writer:
                for example in example_group:
                    # Make sure it's not empty.
                    if example:
                        tf_example = _serialize_example(
                            example,
                            skip_empty=args.skip_empty
                        )
                        if tf_example is not None:
                            writer.write(tf_example.SerializeToString())
    print('Done!')


def _serialize_example_one_domain(example, pad_for_satsim=False,
                                  skip_empty=False, generator=None):
    """Builds a TFRecords Example object from the example data.

    Args:
        example: example structure with (a_path, b_path, annotation_path).

    Keyword Args:
        pad_for_satsim: whether to pad images for satsim. Default is False.
        skip_empty: whether to skip frames with no object. Default is False.
        generator: generator model used to filter inputs.
    """

    # Handle satsim offsets.
    if pad_for_satsim:
        pad_amount = 0.02
    else:
        pad_amount = 0.00

    # Load annotation data.
    (a_path, a_annotation) = example
    with open(a_annotation, 'r') as fp:
        a_annotations = json.load(fp)['data']

    # Parse annotation data.
    (a_class_id, a_y_min, a_y_max, a_y_center, a_x_min, a_x_max, a_x_center, a_magnitude,
     a_path_name) = _parse_annotations(a_annotations, pad_amount, skip_empty)

    # Load raw image data.
    a_data = _read_fits(a_path)
    a_filtered = a_data
    if generator is not None:
        image = tf.cast(a_data, dtype=tf.float32)
        image = tf.expand_dims(image, axis=-1)
        image = tf.image.per_image_standardization(image)
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=1.0,
                                 dtype=tf.float32)
        gen_noise = generator(tf.expand_dims(noise, axis=0))
        a_filtered = image + tf.squeeze(gen_noise, axis=0)
        a_filtered = a_filtered.numpy()
        # a_filtered = tf.cast(a_filtered, dtype=tf.uin16)

    # Create the features for this example
    features = {
        'images_raw': _bytes_feature([a_filtered.tostring()]),
        'filename': _bytes_feature([a_path_name.encode()]),
        'height': _int64_feature([a_annotations['sensor']['height']]),
        'width': _int64_feature([a_annotations['sensor']['width']]),
        'classes': _int64_feature(a_class_id),
        'ymin': _float_feature(a_y_min),
        'ymax': _float_feature(a_y_max),
        'ycenter': _float_feature(a_y_center),
        'xmin': _float_feature(a_x_min),
        'xmax': _float_feature(a_x_max),
        'xcenter': _float_feature(a_x_center),
        'magnitude': _float_feature(a_magnitude),
    }
    # Create an example protocol buffer
    return tf.train.Example(features=tf.train.Features(feature=features))


def make_filtered_tf_records(args):
    """Make TFRecords files from images and annotations.

    Args:
        dataset: DataSet object representing source data.
        generator: generator model used to filter source data.
        output_dir: directory to which to save

    """

    # Set the visible devices to those specified:
    physical_devices = tf.config.list_physical_devices('GPU')
    used_devices = [physical_devices[i] for i in args.devices]
    try:
        # Specify enabled GPUs.
        tf.config.set_visible_devices(used_devices, 'GPU')
        logical_devices = tf.config.list_logical_devices('GPU')
        print(f'{len(physical_devices)} physical GPUs,',
              f'{len(logical_devices)} logical GPUs.')
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        print(f'{len(physical_devices)} physical GPUs.',
              'Could not set visible devices!')

    a_dir = Path(args.a_dir).resolve()
    a_annotation_dir = Path(args.a_annotation_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    generator_path = Path(args.generator_path).resolve()
    _ = mish(0.)  # take care of lazy mish init.
    generator = load_model(generator_path)
    a_paths = sorted(a_dir.glob('**/*.fits'))
    a_annotation_paths = sorted(a_annotation_dir.glob('**/Annotations/*.json'))
    examples = []
    for a_path, a_annotation in zip(a_paths, a_annotation_paths):
        examples.append((a_path, a_annotation))
    splits_dict = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
    partitions = _partition_examples(examples, splits_dict)
    # Make TFRecords from partitions.
    for name, examples in partitions.items():
        print(f'Writing partition "{name}" with {len(examples)} examples...')
        partition_dir = output_dir / name
        partition_dir.mkdir(parents=True)
        groups, num_groups = _group_list(examples, args.group_size)
        for i, example_group in tqdm(enumerate(groups), total=num_groups):
            tfrecords_name = f'{args.output_name}_{name}_{i}.tfrecords'
            output_path = partition_dir / tfrecords_name
            with tf.io.TFRecordWriter(output_path.as_posix()) as writer:
                for example in example_group:
                    # Make sure it's not empty.
                    if example:
                        tf_example = _serialize_example_one_domain(
                            example,
                            skip_empty=args.skip_empty,
                            generator=generator,
                        )
                        if tf_example is not None:
                            writer.write(tf_example.SerializeToString())
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate TFRecords files from FITS images.'
    )
    parser.add_argument('--a_dir', type=str,
                        help='Path to directory for data A.')
    parser.add_argument('--b_dir', type=str,
                        help='Path to directory for data B.')
    parser.add_argument('--a_annotation_dir', type=str, default=None,
                        help='Path to annotation files for data A.')
    parser.add_argument('--b_annotation_dir', type=str, default=None,
                        help='Path to annotation files for data B.')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory for TFRecords files.')
    parser.add_argument('--group_size', type=int, default=256,
                        help='Number of records per TFRecords file.')
    parser.add_argument('--output_name', default='tfrecords',
                        help='Name prepended to output TFRecords files.')
    parser.add_argument('--skip_empty', default=False, action='store_true',
                        help='Whether to skip empty frames in dataset.')
    parser.add_argument('--generator_path', type=str, default=None,
                        help='Path to generator for use in creating images.')
    parser.add_argument('--devices', nargs='+', type=int,
                        help='List of physical devices for TensorFlow to use.')
    args = parser.parse_args()
    _check_args(args)
    if args.generator_path is not None:
        make_filtered_tf_records(args)
    else:
        make_tf_records(args)
