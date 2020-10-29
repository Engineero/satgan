from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
import argparse
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from model.data_loader import load_examples
from model.scatgan import create_model
from model.losses import (compute_total_loss, compute_apply_gradients,
                          calc_discriminator_loss, calc_generator_loss,
                          calc_task_loss)
from model.utils.plot_summaries import plot_summaries
from miss_data_generator import DatasetGenerator
from yolo_v3 import build_yolo_model


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

    # Create YOLO encoder to be used in making generator.
    _, _, encoder = build_yolo_model(
        base_model_name=a.base_model_name,
        is_recurrent=a.is_recurrent,
        num_predictor_heads=a.num_pred_layers,
        max_inferences_per_image=a.max_inferences,
        max_bbox_overlap=a.max_bbox_overlap,
        confidence_threshold=a.confidence_threshold,
    )

    # Build data generators for source domain.
    a_train_data = DatasetGenerator(
        a.a_train_dir,
        parse_function=encoder.parse_data,
        augment=False,
        shuffle=True,
        batch_size=a.batch_size,
        num_threads=a.num_parallel_calls,
        buffer=a.buffer_size,
        encoding_function=encoder.encode_for_yolo,
    )
    a_val_data = DatasetGenerator(
        a.a_valid_dir,
        parse_function=encoder.parse_data,
        augment=False,
        shuffle=False,
        batch_size=a.batch_size,
        num_threads=a.num_parallel_calls,
        buffer=a.buffer_size,
        encoding_function=encoder.encode_for_yolo,
    )
    if a.a_test_dir is not None:
        a_test_data = DatasetGenerator(
            a.a_test_dir,
            parse_function=encoder.parse_data,
            augment=False,
            shuffle=False,
            batch_size=a.batch_size,
            num_threads=a.num_parallel_calls,
            buffer=a.buffer_size,
            encoding_function=encoder.encode_for_yolo,
        )

    # a_train_data = load_examples(a, a.a_train_dir, shuffle=True,
    #                              pad_bboxes=a.pad_bboxes_a)
    # a_val_data = load_examples(a, a.a_valid_dir, pad_bboxes=a.pad_bboxes_a)
    # if a.a_test_dir is not None:
    #     a_test_data = load_examples(a, a.a_test_dir, pad_bboxes=a.pad_bboxes_a)

    # Build data generators for target domain.
    b_train_data = DatasetGenerator(
        a.b_train_dir,
        parse_function=encoder.parse_data,
        augment=False,
        shuffle=True,
        batch_size=a.batch_size,
        num_threads=a.num_parallel_calls,
        buffer=a.buffer_size,
        encoding_function=encoder.encode_for_yolo,
    )
    b_val_data = DatasetGenerator(
        a.b_valid_dir,
        parse_function=encoder.parse_data,
        augment=False,
        shuffle=False,
        batch_size=a.batch_size,
        num_threads=a.num_parallel_calls,
        buffer=a.buffer_size,
        encoding_function=encoder.encode_for_yolo,
    )
    if a.b_test_dir is not None:
        b_test_data = DatasetGenerator(
            a.b_test_dir,
            parse_function=encoder.parse_data,
            augment=False,
            shuffle=False,
            batch_size=a.batch_size,
            num_threads=a.num_parallel_calls,
            buffer=a.buffer_size,
            encoding_function=encoder.encode_for_yolo,
        )

    # b_train_data = load_examples(a, a.b_train_dir, shuffle=True,
    #                              pad_bboxes=a.pad_bboxes_b)
    # b_val_data = load_examples(a, a.b_valid_dir, pad_bboxes=a.pad_bboxes_b)
    # if a.b_test_dir is not None:
    #     b_test_data = load_examples(a, a.b_test_dir, pad_bboxes=a.pad_bboxes_b)

    # Build the model.
    model, generator, _ = create_model(a, a_train_data.dataset,
                                       b_train_data.dataset)
    model.summary()

    # Define the optimizer, losses, and weights.
    if a.multi_optim:
        with tf.device(f'/device:GPU:{a.devices[0]}'):
            optimizer_discrim = Adam(learning_rate=a.lr_dsc, amsgrad=a.ams_grad)
            optimizer_gen = Adam(learning_rate=a.lr_gen, amsgrad=a.ams_grad)
        with tf.device(f'/device:GPU:{a.devices[-1]}'):
            optimizer_task = Adam(learning_rate=a.lr_task, amsgrad=a.ams_grad)
        optimizer_list = [optimizer_discrim, optimizer_gen, optimizer_task]
        loss_list = [calc_discriminator_loss, calc_generator_loss, calc_task_loss]
    else:
        optimizer_list = [Adam(learning_rate=a.lr_single, amsgrad=a.ams_grad)]
        loss_list = [compute_total_loss]

    # Train the model.
    batches_seen = tf.Variable(0, dtype=tf.int64)
    with writer.as_default():
        # Create metrics for accumulating validation, test losses.
        mean_total = Mean()
        mean_discrim = Mean()
        mean_gen = Mean()
        mean_task = Mean()
        mean_list = [mean_total, mean_discrim, mean_gen, mean_task]
        tf.config.experimental_functions_run_eagerly()
        for epoch in range(a.max_epochs):
            print(f'Training epoch {epoch+1} of {a.max_epochs}...')
            epoch_start = time.time()

            for batch_num, (a_batch, b_batch) in enumerate(zip(a_train_data.dataset,
                                                               b_train_data.dataset)):

                # Generate noise batch for this step.
                inputs, _ = a_batch
                noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0,
                                         stddev=1.0, dtype=tf.float32)

                # Save summary images, statistics.
                if batch_num % a.summary_freq == 0:
                    print(f'Writing outputs for epoch {epoch+1}, batch {batch_num}.')
                    print(f'\nA batch:\n{a_batch}')
                    print(f'\nB batch:\n{b_batch}')
                    inputs, a_task_targets = a_batch
                    targets, b_task_targets = b_batch
                    gen_outputs, discrim_outputs, task_outputs = model(
                        [inputs, noise, targets]
                    )
                    model_inputs = (inputs, targets, a_task_targets,
                                    b_task_targets, noise)
                    model_outputs = (gen_outputs, discrim_outputs,
                                     task_outputs)

                    # Plot all of the summary images.
                    plot_summaries(a, model_inputs, model_outputs,
                                   batches_seen)

                    # Compute batch losses.
                    total_loss, discrim_loss, gen_loss, task_loss = \
                        compute_total_loss(
                            a,
                            model_inputs,
                            model_outputs,
                            batches_seen,
                            return_all=True
                        )
                    print(f'Batch {batch_num} performance\n',
                          f'total loss: {total_loss:.4f}\t',
                          f'discriminator loss: {discrim_loss:.4f}\t',
                          f'generator loss: {gen_loss:.4f}\t',
                          f'task loss: {task_loss:.4f}\t')

                # Update the model.
                compute_apply_gradients(
                    a,
                    model,
                    a_batch,
                    b_batch,
                    noise,
                    optimizer_list,
                    loss_list,
                    batches_seen
                )
                batches_seen.assign_add(1)
                writer.flush()

            epoch_time = time.time() - epoch_start
            print(f'Epoch {epoch+1} completed in {epoch_time} sec.\n')

            # Eval on validation data, save best model, early stopping...
            for m in mean_list:
                m.reset_states()
            for a_batch, b_batch in zip(a_val_data.dataset, b_val_data.dataset):
                inputs, a_task_targets = a_batch
                targets, b_task_targets = b_batch
                noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0,
                                         stddev=1.0, dtype=tf.float32)
                gen_outputs, discrim_outputs, task_outputs = model([inputs,
                                                                    noise,
                                                                    targets])
                model_inputs = (inputs, targets, a_task_targets, b_task_targets)
                model_outputs = (gen_outputs, discrim_outputs, task_outputs)

                total_loss, discrim_loss, gen_loss, task_loss = \
                    compute_total_loss(
                        a,
                        model_inputs,
                        model_outputs,
                        batches_seen,
                        return_all=True
                    )
                for m, loss in zip(mean_list, [total_loss,
                                               discrim_loss,
                                               gen_loss,
                                               task_loss]):
                    m.update_state([loss])
                if epoch == 0:
                    min_loss = mean_list[0].result().numpy()
                    epochs_without_improvement = 0
            print(f'Total validation loss: {mean_list[0].result().numpy()}')
            if mean_list[0].result().numpy() <= min_loss \
                and a.output_dir is not None:
                min_loss = mean_list[0].result().numpy()
                print(f'Saving model with total loss {min_loss:.4f} ',
                      f'to {a.output_dir}.')
                model.save(output_path / 'full_model')
                generator.save(output_path / 'generator')
                # task_net.save(output_path / 'task_net')
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            print(f'Epoch {epoch+1} performance\n',
                  f'total loss: {mean_list[0].result().numpy():.4f}\t',
                  f'discriminator loss: {mean_list[1].result().numpy():.4f}\t',
                  f'generator loss: {mean_list[2].result().numpy():.4f}\t',
                  f'task loss: {mean_list[3].result().numpy():.4f}\t')

            # Check for early stopping.
            if epochs_without_improvement >= a.early_stop_patience:
                print(f'{a.early_stop_patience} epochs passed without ',
                      'improvement. Stopping training.')
                break

        # Test the best saved model or current model if not saving.
        if a.output_dir is not None:
            model = load_model(output_path / 'full_model')
        for m in mean_list:
            m.reset_states()
        if a.a_test_dir is not None and a.b_test_dir is not None:
            for a_batch, b_batch in zip(a_test_data.dataset, b_test_data.dataset):
                inputs, a_task_targets = a_batch
                targets, b_task_targets = b_batch
                noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0,
                                         stddev=1.0, dtype=tf.float32)
                gen_outputs, discrim_outputs, task_outputs = model([inputs,
                                                                    noise,
                                                                    targets])
                model_inputs = (inputs, targets, a_task_targets,
                                b_task_targets)
                model_outputs = (gen_outputs, discrim_outputs, task_outputs)

                total_loss, discrim_loss, gen_loss, task_loss = \
                    compute_total_loss(
                        a,
                        model_inputs,
                        model_outputs,
                        batches_seen,
                        return_all=True
                    )
                for m, loss in zip(mean_list, [total_loss,
                                               discrim_loss,
                                               gen_loss,
                                               task_loss]):
                    m.update_state([loss])
            print(f'Test performance\n',
                  f'total loss: {mean_list[0].result().numpy():.4f}\t',
                  f'discriminator loss: {mean_list[1].result().numpy():.4f}\t',
                  f'generator loss: {mean_list[2].result().numpy():.4f}\t',
                  f'task loss: {mean_list[3].result().numpy():.4f}\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a_train_dir",
        default=None,
        help="Path to folder containing source TFRecords training files."
    )
    parser.add_argument(
        "--a_valid_dir",
        default=None,
        help="Path to folder containing source TFRecords validation files."
    )
    parser.add_argument(
        "--a_test_dir",
        default=None,
        help="Path to folder containing source TFRecords testing files."
    )
    parser.add_argument(
        "--b_train_dir",
        default=None,
        help="Path to folder containing target TFRecords training files."
    )
    parser.add_argument(
        "--b_valid_dir",
        default=None,
        help="Path to folder containing target TFRecords validation files."
    )
    parser.add_argument(
        "--b_test_dir",
        default=None,
        help="Path to folder containing target TFRecords testing files."
    )
    parser.add_argument("--mode", required=True,
                        choices=["train", "test", "export"])
    parser.add_argument("--output_dir", required=True,
                        help="where to put output files")
    parser.add_argument("--tensorboard_dir", default=None,
                        help="Directory where tensorboard files are written.")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="directory with checkpoint to resume training from or use for testing"
    )
    parser.add_argument("--max_steps", type=int, default=0,
                        help="number of training steps (0 to disable)")
    parser.add_argument("--max_epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--summary_freq", type=int, default=100,
                        help="Update summaries every summary_freq steps")
    parser.add_argument("--separable_conv", action="store_true",
                        help="use separable convolutions in the generator")
    parser.add_argument("--aspect_ratio", type=float, default=1.0,
                        help="aspect ratio of output images (width/height)")
    parser.add_argument("--lab_colorization", action="store_true",
                        help="split input image into brightness (A) and color (B)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="number of images in batch")
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
    parser.add_argument('--num_parallel_calls', default=None, type=int,
                        help='Number of parallel jobs for data mapping.')
    parser.add_argument('--pad_bboxes_a', action='store_true', default=False,
                        help='If specified, pads A-domain bboxes.')
    parser.add_argument('--pad_bboxes_b', action='store_true', default=False,
                        help='If specified, pads B-domain bboxes.')

    # export options
    parser.add_argument("--output_filetype", default="png",
                        choices=["png", "jpeg"])
    args = parser.parse_args()
    main(args)
