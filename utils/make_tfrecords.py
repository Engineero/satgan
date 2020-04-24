"""Creates TFRecords files for pix2pix task network.

Created: 2020-04-21
Author: Nathan L. Toner
"""


import tensorflow as tf
import numpy as np
import argparse
import os
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
    b_dir = Path(args.b_dir).resolve()
    annotation_dir = Path(args.annotation_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not a_dir.is_dir():
        raise NotADirectoryError(f'Path A {args.a_dir} is not a directory!')
    if not b_dir.is_dir():
        raise NotADirectoryError(f'Path B {args.b_dir} is not a directory!')
    if not annotation_dir.is_dir():
        raise NotADirectoryError(
            f'Annotation path {args.annotation_dir} is not a directory!'
        )
    if not output_dir.is_dir():
        print(f'Making output directory {output_dir}...')
        os.makedirs(output_dir)
    if not len(output_dir.glob('*')) == 0:
        raise ValueError('Output directory is not empty!')
    path_list = [a_dir, b_dir, annotation_dir]
    for path in path_list:
        if len(path.glob('*')) == 0:
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
    return grouped_list


# The following functions can be used to convert a value to a type compatible
# with tf.Example.
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
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


def _serialize_example(example, pad_for_satsim=False):
    """Builds a TFRecords Example object from the example data."""
    (a_path, b_path, annotation) = example
    a_data = _read_fits(a_path)
    b_data = _read_fits(b_path)
    with open(annotation, 'r') as fp:
        annotations = json.load(fp)['data']

    # Handle satsim offsets.
    if pad_for_satsim:
        pad_amount = 0.02
    else:
        pad_amount = 0.00

    # Read data from annotation file.
    # sequence = [file.encode() for file in annotations['file']['sequence']]
    # class_name = [obj['class_name'].encode() for obj in annotations['objects']]
    class_id = [obj['class_id'] for obj in annotations['objects']]
    y_min = [obj['y_min'] - pad_amount for obj in annotations['objects']]
    y_max = [obj['y_max'] + pad_amount for obj in annotations['objects']]
    y_center = [obj['y_center'] for obj in annotations['objects']]
    x_min = [obj['x_min'] - pad_amount for obj in annotations['objects']]
    x_max = [obj['x_max'] + pad_amount for obj in annotations['objects']]
    x_center = [obj['x_center'] for obj in annotations['objects']]
    # source = [obj['source'].encode() for obj in annotations['objects']]
    magnitude = [obj['magnitude'] for obj in annotations['objects']]
    dir_name = annotations['file']['dirname']
    file_name = annotations['file']['filename']
    path_name = (Path(dir_name) / Path(file_name)).as_posix()

    # Replace the unknown magnitude's with NaN's
    for i in range(len(magnitude)):
        if magnitude[i] is None:
            magnitude[i] = float('NaN')
    
    # Create the features for this example
    features = {
        'a_raw': _bytes_feature([a_data.tostring()]),
        'b_raw': _bytes_feature([b_data.tostring()]),
        'filename': _bytes_feature([path_name.encode()]),
        'height': _int64_feature([annotations['sensor']['height']]),
        'width': _int64_feature([annotations['sensor']['width']]),
        'classes': _int64_feature(class_id),
        'ymin': _float_feature(y_min),
        'ymax': _float_feature(y_max),
        'ycenter': _float_feature(y_center),
        'xmin': _float_feature(x_min),
        'xmax': _float_feature(x_max),
        'xcenter': _float_feature(x_center),
    }
    # Create an example protocol buffer
    return tf.train.Example(features=tf.train.Features(feature=features))


def make_tf_records(args):
    """Make TFRecords files from images and annotations."""
    a_dir = Path(args.a_dir).resolve()
    b_dir = Path(args.b_dir).resolve()
    annotation_dir = Path(args.annotation_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    a_paths = sorted(a_dir.glob('**/*.fits'))
    b_paths = sorted(b_dir.glob('**/*.fits'))
    annotation_paths = sorted(annotation_dir.glob('**/*.json'))
    examples = []
    for a_path, b_path, annotation in zip(a_paths, b_paths, annotation_paths):
        examples.append((a_path, b_path, annotation))
    splits_dict = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
    partitions = _partition_examples(examples, splits_dict)
    # Make TFRecords from partitions.
    for name, examples in partitions.items():
        print(f'Writing partition "{name}" with {len(examples)} examples...')
        partition_dir = output_dir / name
        os.mkdir(partition_dir)
        groups = _group_list(examples, args.group_size)
        for i, example_group in tqdm(enumerate(groups), total=len(groups)):
            tfrecords_name = f'{args.output_name}_{name}_{i}.tfrecords'
            output_path = partition_dir / tfrecords_name
            with tf.io.TFRecordWriter(output_path) as writer:
                for example in example_group:
                    # Make sure it's not empty.
                    if example:
                        tf_example = _serialize_example(example)
                        writer.write(tf_example.SerializeToString())
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate TFRecords files from FITS images.'
    )
    parser.add_argument('--a_dir', help='Path to directory for data A.')
    parser.add_argument('--b_dir', help='Path to directory for data B.')
    parser.add_argument('--annotation_dir', help='Path to annotation files.')
    parser.add_argument('--output_dir',
                        help='Output directory for TFRecords files.')
    parser.add_argument('--group_size', type=int, default=256,
                        help='Number of records per TFRecords file.')
    parser.add_argument('--output_name', deafult='tfrecords',
                        help='Name prepended to output TFRecords files.')
    args = parser.parse_args()
    _check_args(args)
    make_tf_records(args)