"""Defines the SCATGAN network."""


from .generator import create_generator
from .discriminator import create_discriminator
from .task_net import create_task_net

from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from yolo_v3 import build_yolo_model, load_yolo_model_weights


def create_model(a, train_data):

    (inputs, noise, targets), (_, a_task_targets, b_task_targets) = \
        next(iter(train_data))
    input_shape = inputs.shape.as_list()[1:]  # don't give Input the batch dim
    noise_shape = noise.shape.as_list()[1:]
    target_shape = targets.shape.as_list()[1:]
    a_task_targets_shape = a_task_targets.shape.as_list()[1:]
    b_task_targets_shape = b_task_targets.shape.as_list()[1:]
    inputs = Input(input_shape)
    noise = Input(noise_shape)
    targets = Input(target_shape)
    a_task_targets = Input(a_task_targets_shape)
    b_task_targets = Input(b_task_targets_shape)

    if a.checkpoint is not None:
        if a.activation == 'mish':
            # Take care of lazy init for tf-addons class.
            from tensorflow_addons.activations import mish
            _ = mish(0.)
        checkpoint_path = Path(a.checkpoint).resolve()
        generator = load_model(checkpoint_path / 'generator')
        print(f'Generator model summary:\n{generator.summary()}')
        model = load_model(checkpoint_path / 'full_model')
        task_net = None

    else:
        with tf.device(f'/device:GPU:{a.devices[0]}'):
            # Create the generator.
            with tf.name_scope("generator"):
                out_channels = target_shape[-1]
                generator = create_generator(a, input_shape, out_channels)
                generator.summary()
                gen_noise = generator(noise)
                fake_img = gen_noise + inputs
                gen_outputs = tf.stack([fake_img, gen_noise], axis=0,
                                       name='generator')

            # Create two copies of discriminator, one for real pairs and one for
            # fake pairs they share the same underlying variables.
            with tf.name_scope("discriminator"):
                # TODO (NLT): figure out discriminator loss, interaction with Keras changes.
                discriminator = create_discriminator(a, target_shape)
                discriminator.summary()
                predict_real = discriminator(targets)  # should -> [0, 1]
                predict_fake = discriminator(fake_img)  # should -> [1, 0]
                discrim_outputs = tf.stack([predict_real, predict_fake], axis=0,
                                           name='discriminator')

        # Create two copies of the task network, one for real images (targets
        # input to this method) and one for generated images (outputs from
        # generator). The task targets (detected objects) should be the same for
        # both.
        with tf.device(f'/device:GPU:{a.devices[-1]}'):
            with tf.name_scope('task_net'):
                if a.use_yolo:
                    task_net, _, _ = build_yolo_model(
                        base_model_name=a.base_model_name,
                        is_recurrent=a.is_recurrent,
                        num_predictor_heads=a.num_pred_layers,
                        max_inferences_per_image=a.max_inferences,
                        max_bbox_overlap=a.max_bbox_overlap,
                        confidence_threshold=a.confidence_threshold,
                    )
                    task_net = load_yolo_model_weights(task_net,
                                                       a.checkpoint_load_path)
                    task_net.name = 'task_net'
                    # task_net._set_inputs(targets)
                    if a.freeze_task:
                        for layer in task_net.get_layer('task_net').layers:
                            print(f'Freezing task net task net layer {layer}.')
                            layer.trainable = False
                else:
                    task_net = create_task_net(a, input_shape)
                pred_task = task_net(targets)
                pred_task_fake = task_net(fake_img)
                pred_task_noise = task_net(gen_noise)
            task_net.summary()
            task_outputs = tf.stack([pred_task, pred_task_fake, pred_task_noise],
                                    axis=0,
                                    name='task_net')

        model = Model(inputs=[inputs, noise, targets],
                      outputs=[gen_outputs, discrim_outputs, task_outputs],
                      name='scatgan')

        # Plot the sub-models and overall model.
        if a.plot_models:
            plot_model(generator, to_file='plots/generator.svg')
            plot_model(task_net, to_file='plots/task_net.svg')
            plot_model(discriminator, to_file='plots/discriminator.svg')
            plot_model(model, to_file='plots/full_model.svg')

    # Return the model. We'll define losses and a training loop back in the
    # main function.
    return model, generator, task_net
