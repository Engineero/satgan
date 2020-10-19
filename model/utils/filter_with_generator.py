"""Filters source domain input with a generator and produces tfrecords.

Created: 2020-10-19
Author: Nathan L. Toner
"""


import numpy as np
import argparse
import json
from pathlib import Path
from astropy.io import fits
from itertools import zip_longest
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow_addons.activations import mish


def _check_args(args):
    """Checks whether arguments are valid.

    Creates the output directory if it does not exist.

    Raises:
        NotADirectoryError if any input directory is None or invalid.
        ValueError if output_dir is *not* empty.
        ValueError if any other directory *is* empty.
    """
    a_dir = Path(args.data_dir).resolve()
    a_annotation_dir = Path(args.annotation_dir).resolve()
    generator_path = Path(args.generator_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not a_dir.is_dir():
        raise NotADirectoryError(f'Path A {args.a_dir} is not a directory!')
    if not a_annotation_dir.is_dir():
        raise NotADirectoryError(
            f'Annotation path {args.a_annotation_dir} is not a directory!'
        )
    if not generator_path.is_dir():
        raise NotADirectoryError(
            'Generator path must point to a directory containing a checkpoint file!'
        )
    if output_dir.is_dir():
        try:
            output_dir.rmdir()
        except:
            raise ValueError('Output directory already exists!')
    else:
        print(f'Making output directory {output_dir}...')
        output_dir.mkdir(parents=True)
    path_list = [a_dir, a_annotation_dir]
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


def _serialize_example(example, pad_for_satsim=False, skip_empty=False,
                       generator=None):
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
    noise = tf.random.normal(shape=tf.shape(a_data), mean=0.0,
                             stddev=1.0, dtype=tf.float32)
    a_filtered = tf.cast(a_data, dtype=tf.float32) + generator(noise)
    a_filtered = tf.cast(a_filtered, dtype=tf.uin16)
    
    # Create the features for this example
    features = {
        'images_original': _bytes_feature([a_data.tostring()]),
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


def make_tf_records(args):
    """Make TFRecords files from images and annotations."""
    a_dir = Path(args.data_dir).resolve()
    a_annotation_dir = Path(args.annotation_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    _ = mish(0.)  # take care of lazy mish init.
    generator = load_model(Path(args.generator_path))
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
        partition_dir.mkdir()
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
    parser.add_argument('--data_dir', type=str,
                        help='Path to directory for source data.')
    parser.add_argument('--annotation_dir', type=str, default=None,
                        help='Path to annotation files for source data.')
    parser.add_argument('--generator_path', type=str, default=None,
                        help='Path to generator checkpoint.')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory for TFRecords files.')
    parser.add_argument('--group_size', type=int, default=256,
                        help='Number of records per TFRecords file.')
    parser.add_argument('--output_name', default='tfrecords',
                        help='Name prepended to output TFRecords files.')
    parser.add_argument('--skip_empty', default=False, action='store_true',
                        help='Whether to skip empty frames in dataset.')
    args = parser.parse_args()
    _check_args(args)
    make_tf_records(args)