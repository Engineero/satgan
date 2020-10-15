"""Script for testing results of running task net on all domains."""


import time
import tensorflow as tf
import numpy as np
import argparse
from pathlib import Path
from tensorflow.keras.models import load_model
from model.data_loader import load_examples
from model.scatgan import create_model
from model.losses import (compute_total_loss, compute_apply_gradients,
                          calc_discriminator_loss, calc_generator_loss,
                          calc_task_loss)
from model.utils.plot_summaries import plot_summaries


# Enable eager execution.
tf.compat.v1.enable_eager_execution()


def main(a):
    # Set the visible devices to those specified:
    physical_devices = tf.config.list_physical_devices('GPU')
    used_devices = [physical_devices[i] for i in a.devices]
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

    # Set up the summary writer.
    output_path = Path(a.output_dir).resolve()
    tensorboard_path = Path(a.tensorboard_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    tensorboard_path.mkdir(parents=True, exist_ok=True)
    writer = tf.summary.create_file_writer(tensorboard_path.as_posix())

    # Build data generators.
    train_data, _, _ = load_examples(a)

    # Build the model.
    model, _, _ = create_model(a, train_data)
    model.summary()

    # Train the model.
    batches_seen = tf.Variable(0, dtype=tf.int64)
    with writer.as_default():
        # Create metrics for accumulating validation, test losses.
        tf.config.experimental_functions_run_eagerly()
        # Only a single epoch.
        print(f'Starting testing...')
        epoch_start = time.time()

        for _, batch in enumerate(train_data):
            # Save summary images, statistics.
            print(f'Writing outputs for test batch...')
            (inputs, noise, targets), (_, a_task_targets, b_task_targets) = batch
            gen_outputs, discrim_outputs, task_outputs = model(
                [inputs, noise, targets]
            )
            model_inputs = (inputs, targets, a_task_targets, b_task_targets,
                            noise)
            model_outputs = (gen_outputs, discrim_outputs, task_outputs)

            # Print raw task outputs.
            print(f'Task net outputs:\n{task_outputs}')
            print(f'\nA target classes:\n{a_task_targets[..., -2]}')
            print(f'\nA predicted classes:\n{task_outputs[1, ..., -2]}')
            print(f'\nB target classes:\n{b_task_targets[..., -2]}')
            print(f'\nB predicted classes:\n{task_outputs[0, ..., -2]}')

            # Plot all of the summary images.
            plot_summaries(a, model_inputs, model_outputs, batches_seen)

            batches_seen.assign_add(1)
            writer.flush()
            break

        epoch_time = time.time() - epoch_start
        print(f'Done evaluating in {epoch_time} sec.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dir",
        default=None,
        help="Path to folder containing TFRecords training files."
    )
    parser.add_argument(
        "--valid_dir",
        default=None,
        help="Path to folder containing TFRecords validation files."
    )
    parser.add_argument(
        "--test_dir",
        default=None,
        help="Path to folder containing TFRecords testing files."
    )
    parser.add_argument("--output_dir", required=True,
                        help="where to put output files")
    parser.add_argument("--tensorboard_dir", default=None,
                        help="Directory where tensorboard files are written.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="directory with checkpoint to resume training from or use for testing"
    )
    parser.add_argument("--separable_conv", action="store_true",
                        help="use separable convolutions in the generator")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="number of images in batch")
    parser.add_argument("--which_direction", type=str, default="AtoB",
                        choices=["AtoB", "BtoA"])
    parser.add_argument("--n_blocks_gen", type=int, default=8,
                        help="Number of ResNet blocks in generator. Must be even.")
    parser.add_argument("--n_layer_dsc", type=int, default=3,
                        help="Number of layers in discriminator.")
    parser.add_argument("--ngf", type=int, default=64,
                        help="number of generator filters in first conv layer")
    parser.add_argument("--ndf", type=int, default=64,
                        help="number of discriminator filters in first conv layer")
    parser.add_argument("--scale_size", type=int, default=286,
                        help="scale images to this size before cropping to 256x256")
    parser.add_argument("--flip", dest="flip", action="store_true",
                        help="flip images horizontally")
    parser.add_argument("--no_flip", dest="flip", action="store_false",
                        help="don't flip images horizontally")
    parser.set_defaults(flip=True)
    parser.add_argument("--lr_gen", type=float, default=4e-4,
                        help="initial learning rate for generator adam")
    parser.add_argument("--lr_dsc", type=float, default=1e-4,
                        help="initial learning rate for discriminator adam")
    parser.add_argument("--lr_task", type=float, default=1e-4,
                        help="initial learning rate for task adam")
    parser.add_argument("--lr_single", type=float, default=1e-4,
                        help="initial learning rate for single adam optimizer")
    parser.add_argument("--beta1_gen", type=float, default=0.5,
                        help="momentum term of generator adam")
    parser.add_argument("--beta1_dsc", type=float, default=0.5,
                        help="momentum term of discriminator adam")
    parser.add_argument("--beta1_task", type=float, default=0.5,
                        help="momentum term of task adam")
    parser.add_argument("--l1_weight", type=float, default=100.0,
                        help="weight on L1 term for generator gradient")
    parser.add_argument("--gan_weight", type=float, default=1.0,
                        help="weight on GAN term for generator gradient")
    parser.add_argument("--n_channels", type=int, default=3,
                        help="Number of channels in image.")
    parser.add_argument("--transform", action="store_true", default=False,
                        help="Whether to apply image transformations.")
    parser.add_argument("--crop_size", type=int, default=256,
                        help="Size of cropped image chunks.")
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Number of classes in the data.')
    parser.add_argument('--l1_reg_kernel', type=float, default=0.,
                        help='L1 regularization term for kernels')
    parser.add_argument('--l2_reg_kernel', type=float, default=0.,
                        help='L2 regularization term for kernels')
    parser.add_argument('--l1_reg_bias', type=float, default=0.,
                        help='L1 regularization term for bias')
    parser.add_argument('--l2_reg_bias', type=float, default=0.,
                        help='L2 regularization term for bias')
    parser.add_argument('--buffer_size', type=int, default=512,
                        help='Buffer size for shuffling input data.')
    parser.add_argument('--spec_norm', default=False, action='store_true',
                        help='Whether to perform spectral normalization.')
    parser.add_argument('--plot_models', default=False, action='store_true',
                        help='Whether to plot model architectures.')
    parser.add_argument('--max_inferences', default=20, type=int,
                        help='Max inferences per image. Default 100.')
    parser.add_argument('--num_pred_layers', default=1, type=int,
                        help='Number of predictor layers to use in network.')
    parser.add_argument('--gen_weight', default=1., type=float,
                        help='Relative weight of generator loss.')
    parser.add_argument('--dsc_weight', default=1., type=float,
                        help='Relative weight of discriminator loss.')
    parser.add_argument('--task_weight', default=1., type=float,
                        help='Relative weight of task loss.')
    parser.add_argument('--xy_weight', default=1., type=float,
                        help='Relative weight of xy task loss component.')
    parser.add_argument('--wh_weight', default=1., type=float,
                        help='Relative weight of wh task loss component.')
    parser.add_argument('--iou_weight', default=1., type=float,
                        help='Relative weight of IoU task loss component.')
    parser.add_argument('--class_weight', default=1., type=float,
                        help='Relative weight of class task loss component.')
    parser.add_argument('--obj_weight', default=1., type=float,
                        help='Relative weight of objectness task loss component.')
    parser.add_argument('--early_stop_patience', default=10, type=int,
                        help='Early stopping patience, epochs. Default 10.')
    parser.add_argument('--multi_optim', default=False, action='store_true',
                        help='Whether to use separate optimizers for each loss.')
    parser.add_argument('--ams_grad', default=False, action='store_true',
                        help='Whether to use AMS Grad variant of Adam optimizer.')
    parser.add_argument('--obj_threshold', type=float, default=0.5,
                        help='Objectness threshold, under which a detection is ignored.')
    parser.add_argument('--use_yolo', default=False, action='store_true',
                        help='Whether to use existing YOLO SatNet for task network.')
    parser.add_argument('--use_sagan', default=False, action='store_true',
                        help='Whether to use self-attending GAN architecture with ResNet blocks.')
    parser.add_argument('--checkpoint_load_path', type=str,
                        default=None,
                        help='Path to the model checkpoint to load.')
    parser.add_argument('--base_model_name', type=str,
                        default="DarkNet",
                        help='The name of the base network to be used.')
    parser.add_argument('--max_bbox_overlap', type=float,
                        default=1.0,
                        help='Maximum amount two inferred boxes can overlap.')
    parser.add_argument('--confidence_threshold', type=float,
                        default=0.0,
                        help='Minimum confidence required to infer a box.')
    parser.add_argument('--is_recurrent', action='store_true',
                        default=False,
                        help='Should we use a recurrent (Convolutional LSTM) '
                             'variant of the model')
    parser.add_argument('--devices', nargs='+', type=int,
                        help='List of physical devices for TensorFlow to use.')
    parser.add_argument('--activation', type=str, default='lrelu',
                        help='lrelu for leaky relu, mish for mish')
    parser.add_argument('--freeze_task', action='store_true',
                        default=False,
                        help='If specified, do not train task network, '
                             'just use its loss.')

    # export options
    parser.add_argument("--output_filetype", default="png",
                        choices=["png", "jpeg"])
    args = parser.parse_args()
    main(args)
