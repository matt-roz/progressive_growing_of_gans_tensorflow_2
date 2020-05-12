import os
import gc
import time
import logging
from typing import Tuple, Union
from datetime import timedelta

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.utils import plot_model
from tensorflow.python.framework.composite_tensor import CompositeTensor

from config import conf
from data import get_dataset_pipeline
from model import generator_paper, discriminator_paper
from losses import wasserstein_discriminator_loss, wasserstein_generator_loss, discriminator_epsilon_drift, \
    wasserstein_gradient_penalty
from utils import save_eval_images, transfer_weights

"""
global variables for 'handier' access to changing objects references. Currently, while not pretty, this is the best way 
to deal with potential side-effects with graph mode execution via compiled tf.functions. For Progressive GANs object 
references for keras objects are updated, since models are growing progressively - graph execution does not support 
dynamically growing models out-of-the box. Functions, which are invoking member functions of newly created models, must 
be retraced and recompiled via AutoGraph. In this case, when a new stage is reached, train_step_fn is updated to the 
mentioned retraced graph execution function (retracing all underlying function invocations such as model.__call__).
"""
generator = None
final_gen = None
discriminator = None
optimizer_gen = None
optimizer_dis = None
global_batch_size = None
train_step_fn = None


def replica_train_step(batch: tf.Tensor, alpha: tf.Tensor) -> tf.Tensor:
    """Trains Progressive GAN for one batch in replica context. Loss functions used in this context must return per
     examples losses and are afterwards scaled by global_batch_size.

    Args:
        batch: a 4D-Tensor containing the images to train with. First dimension depicts per replica batch_size.
        alpha: a scalar depicting the current alpha for smoothing images of Progressive GANs

    Returns:
        A 1D-Tensor with shape=(4,) containing train information of local replica:
            gen_loss: wasserstein generator loss on replica
            disc_loss: wasserstein discriminator loss on replica
            disc_gradient_loss: gradient penalty loss of discriminator on replica
            disc_epsilon_loss: epsilon drift loss of discriminator on replica
    """
    # generate noise for projecting fake images
    replica_batch_size = tf.shape(batch)[0]
    noise = tf.random.normal([replica_batch_size, conf.model.noise_dim])
    with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
        # forward pass: inference through both models on tape to create predictions
        fake_images = generator([noise, alpha], training=True)
        real_image_guesses = discriminator([batch, alpha], training=True)
        fake_image_guesses = discriminator([fake_images, alpha], training=True)

        # compute gradient penalty
        if conf.train.use_gradient_penalty:
            disc_gradient_loss = wasserstein_gradient_penalty(discriminator, batch, fake_images, conf.train.wgan_target, conf.train.wgan_lambda, alpha)
            disc_gradient_loss = tf.nn.compute_average_loss(disc_gradient_loss, global_batch_size=global_batch_size)
        else:
            disc_gradient_loss = 0.0

        # compute drift penalty
        if conf.train.use_epsilon_penalty:
            disc_epsilon_loss = discriminator_epsilon_drift(real_image_guesses, conf.train.drift_epsilon)
            disc_epsilon_loss = tf.nn.compute_average_loss(disc_epsilon_loss, global_batch_size=global_batch_size)
        else:
            disc_epsilon_loss = 0.0

        # calculate losses
        gen_loss = wasserstein_generator_loss(fake_image_guesses)
        gen_loss = tf.nn.compute_average_loss(gen_loss, global_batch_size=global_batch_size)
        _disc_loss = wasserstein_discriminator_loss(real_image_guesses, fake_image_guesses)
        _disc_loss = tf.nn.compute_average_loss(_disc_loss, global_batch_size=global_batch_size)
        disc_stacked_loss = tf.stack((_disc_loss, disc_gradient_loss, disc_epsilon_loss))
        disc_loss = tf.reduce_sum(disc_stacked_loss)

    # collocate gradients from tapes
    gradients_generator = generator_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_discriminator = discriminator_tape.gradient(disc_loss, discriminator.trainable_variables)
    # backward pass: apply gradients via optimizers to update models
    optimizer_gen.apply_gradients(zip(gradients_generator, generator.trainable_variables))
    optimizer_dis.apply_gradients(zip(gradients_discriminator, discriminator.trainable_variables))
    return tf.stack((gen_loss, _disc_loss, disc_gradient_loss, disc_epsilon_loss))


def global_train_step(batch: Union[tf.Tensor, CompositeTensor], alpha: tf.Tensor) -> tf.Tensor:
    """Trains Progressive GAN for one batch over all replicas. Losses are reduced over all replicas with reduce_sum.
    This function expects all replica losses to be scaled with global_batch_size.

    Args:
        batch: Either a 4D-Tensor for single-replica training or a CompositeTensor for multi-replica training from a
            distributed dataset
        alpha: a scalar depicting the current alpha for smoothing images of Progressive GANs

    Returns:
        A 1D-Tensor with shape=(4,) containing train information:
            gen_loss: wasserstein generator loss for batch over all replicas
            disc_loss: wasserstein discriminator loss for batch over all replicas
            disc_gradient_loss: gradient penalty loss of discriminator for batch over all replicas
            disc_epsilon_loss: epsilon drift loss of discriminator for batch over all replicas
    """
    # log whenever this function is traced into a graph function (after each stage increase of Progressive GANs)
    if not conf.general.train_eagerly:
        logging.info(f'tf.function tracing train_step: batch={batch}, alpha={alpha}')
    per_replica_losses = conf.general.strategy.experimental_run_v2(replica_train_step, args=(batch, alpha))
    reduced_losses = conf.general.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    return reduced_losses


def epoch_step(dataset: tf.data.Dataset, current_epoch: int, num_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Trains Progressive GAN for one epoch. This function is always executed eagerly and can not be compiled into a
    graph function due to transfer_weights having to be executed within a python thread to keep an exponential moving
    average of the generator.
    
    Args:
        dataset: dataset to train one epoch on
        current_epoch: current epoch number
        num_steps: number of steps (batches to train for) in current epoch

    Returns:
        A tuple of length three containing epoch train information:
            gen_loss: mean generator loss over all replicas
            disc_loss: mean discriminator loss shape=(3,) with wasserstein_loss, gradient_loss, epsilon_loss
            image_count: number of images Progressive GAN was trained with in epoch
    """
    # return metrics
    _epoch_gen_loss, _epoch_dis_loss, _image_count = 0.0, 0.0, 0.0
    dataset = tqdm(iterable=dataset, desc=f"epoch-{current_epoch+1:04d}", unit="batch", total=num_steps, leave=False)

    for image_batch in dataset:
        batch_gen_loss, *batch_dis_loss = train_step_fn(image_batch, tf.constant(conf.model.alpha))
        batch_dis_loss = tf.stack(batch_dis_loss)

        # smooth available weights from current_stage generator into final generator
        transfer_weights(source_model=generator, target_model=final_gen, beta=conf.model.generator_ema)

        # compute moving average of loss metrics
        _size = tf.cast(global_batch_size, tf.float32)
        _epoch_gen_loss = (_image_count * _epoch_gen_loss + _size * batch_gen_loss) / (_image_count + _size)
        _epoch_dis_loss = (_image_count * _epoch_dis_loss + _size * batch_dis_loss) / (_image_count + _size)
        _image_count += _size

        # increase alpha
        if conf.model.use_alpha_smoothing:
            conf.model.alpha = tf.minimum(conf.model.alpha + conf.model.alpha_step * _size, 1.0)

        # additional chief tasks during training
        message = f"batch_gen_loss={batch_gen_loss:.3f}, batch_dis_loss={tf.reduce_sum(batch_dis_loss):.3f}"
        dataset.set_postfix_str(message)
        logging.debug(message)

    return _epoch_gen_loss, _epoch_dis_loss, _image_count


def instantiate_stage_objects(stage: int) -> \
        Tuple[tf.keras.Model, tf.keras.Model, tf.data.Dataset, tf.optimizers.Optimizer, tf.optimizers.Optimizer]:
    """Helper function that constructs and returns all tf/keras objects necessary for training stage 'stage'."""
    with conf.general.strategy.scope():
        # create optimizers with new learning rates
        optim_gen = tf.keras.optimizers.Adam(
            learning_rate=conf.optimizer.learning_rates[stage],
            beta_1=conf.optimizer.beta1,
            beta_2=conf.optimizer.beta2,
            epsilon=conf.optimizer.epsilon,
            name='adam_generator')
        optim_dis = tf.keras.optimizers.Adam(
            learning_rate=conf.optimizer.learning_rates[stage],
            beta_1=conf.optimizer.beta1,
            beta_2=conf.optimizer.beta2,
            epsilon=conf.optimizer.epsilon,
            name='adam_discriminator')

        # construct dataset
        batch_size = conf.data.replica_batch_sizes[stage] * conf.general.strategy.num_replicas_in_sync
        dataset = get_dataset_pipeline(name=f"{conf.data.registered_name}/{2**stage}", batch_size=batch_size,
                                       buffer_size=conf.data.buffer_sizes[stage], **conf.data)
        dataset = conf.general.strategy.experimental_distribute_dataset(dataset)

        # create models
        gen = generator_paper(stop_stage=stage, name=f"generator_stage_{stage}", **conf.model)
        dis = discriminator_paper(stop_stage=stage, name=f"discriminator_stage_{stage}", **conf.model)

    # logging, plotting, ship
    if conf.general.is_chief and conf.general.logging:
        logging.info(f"Successfully instantiated {dis.name} and {gen.name} for stage={stage}")
        gen.summary(print_fn=logging.info, line_length=150, positions=[.33, .55, .67, 1.])
        dis.summary(print_fn=logging.info, line_length=150, positions=[.33, .55, .67, 1.])
    if conf.general.is_chief and conf.general.save:
        plot_model(gen, os.path.join(conf.general.out_dir, f"net_{gen.name}.png"), True, False, dpi=178)
        plot_model(dis, os.path.join(conf.general.out_dir, f"net_{dis.name}.png"), True, False, dpi=178)
    return gen, dis, dataset, optim_gen, optim_dis


def train():
    global optimizer_gen, optimizer_dis, global_batch_size, final_gen, generator, discriminator, train_step_fn

    # instantiate target model that will be an exponential moving average of generator, log summary
    with conf.general.strategy.scope():
        final_gen = generator_paper(stop_stage=conf.model.final_stage, return_all_outputs=True, **conf.model)
    logging.info(f"Successfully instantiated the following final model {final_gen.name}")
    final_gen.summary(print_fn=logging.info, line_length=150, positions=[.33, .55, .67, 1.])
    if conf.general.is_chief and conf.general.save:
        plot_model(final_gen, os.path.join(conf.general.out_dir, f"net_{final_gen.name}.png"), True, False, dpi=178)

    # instantiate initial stage trainable models, optimizers and dataset
    current_stage = 2 if conf.model.use_stages else conf.model.final_stage
    generator, discriminator, train_dataset, optimizer_gen, optimizer_dis = instantiate_stage_objects(current_stage)
    transfer_weights(source_model=generator, target_model=final_gen, beta=0.0)  # force same initialization

    # extract image_shape from train_dataset's element_spec (which is depending on distribution strategy)
    if isinstance(train_dataset.element_spec, tf.TensorSpec):
        image_shape = train_dataset.element_spec.shape[1:]
    elif isinstance(train_dataset.element_spec, tf.python.distribute.values.PerReplicaSpec):
        image_shape = train_dataset.element_spec._value_specs[0].shape[1:]
    else:
        raise RuntimeError(f"dataset {train_dataset} returns unknown element_spec of type {type(train_dataset.element_spec)}")

    # variables, counters, timings, etc.
    total_image_count = 0
    replica_batch_size = conf.data.replica_batch_sizes[current_stage]
    global_batch_size = replica_batch_size * conf.general.strategy.num_replicas_in_sync
    steps_per_epoch = int(conf.data.num_examples // global_batch_size)
    stage_start_time = train_start_time = time.time()
    random_noise = tf.random.normal(shape=(16, conf.model.noise_dim), seed=1000)

    # create epoch iterator, compile train_step_fn depending on run mode configuration
    epochs = tqdm(range(conf.train.epochs), f"Progressive-GAN(stage={current_stage}, shape={image_shape}", unit='epoch')
    logging.info(f"Starting to train Stage {current_stage}")
    train_step_fn = global_train_step if conf.general.train_eagerly else tf.function(global_train_step, experimental_compile=conf.general.XLA)

    for epoch in epochs:
        # make an epoch step
        epoch_start_time = time.time()
        gen_loss, dis_loss, image_count = epoch_step(train_dataset, epoch, steps_per_epoch)
        epoch_duration = time.time() - epoch_start_time
        total_image_count += int(image_count.numpy())

        # chief does TensorBoard logging
        if conf.general.is_chief and conf.general.logging and conf.general.log_freq and (epoch + 1) % conf.general.log_freq == 0:
            with conf.general.summary.as_default():
                num_replicas = conf.general.strategy.num_replicas_in_sync
                num_replicas_per_node = num_replicas / conf.general.nnodes
                batches_per_second = tf.cast(steps_per_epoch, tf.float32) / epoch_duration
                images_per_second = image_count.numpy()/epoch_duration
                seconds_per_kimages = 1000*epoch_duration/image_count.numpy()
                disc_gradient_penalty = dis_loss[1] * (conf.train.wgan_target ** 2) / conf.train.wgan_lambda
                disc_gradient_mixed_norm = np.sqrt(disc_gradient_penalty + 1e-8) + conf.train.wgan_target
                tf.summary.scalar("train_speed/epoch", epoch, epoch)
                tf.summary.scalar("train_speed/total_image_count", total_image_count, epoch)
                tf.summary.scalar("train_speed/epoch_duration", epoch_duration, epoch)
                tf.summary.scalar("train_speed/train_duration", time.time() - train_start_time, epoch)
                tf.summary.scalar("train_speed/global/images_per_second", images_per_second, epoch)
                tf.summary.scalar("train_speed/global/batches_per_second", batches_per_second, epoch)
                tf.summary.scalar("train_speed/global/seconds_per_kimages", seconds_per_kimages, epoch)
                tf.summary.scalar("train_speed/replica/images_per_second", images_per_second/num_replicas, epoch)
                tf.summary.scalar("train_speed/replica/batches_per_second", batches_per_second/num_replicas, epoch)
                tf.summary.scalar("train_speed/replica/seconds_per_kimages", seconds_per_kimages/num_replicas, epoch)
                tf.summary.scalar("train_speed/node/images_per_second", num_replicas_per_node*images_per_second/num_replicas, epoch)
                tf.summary.scalar("train_speed/node/batches_per_second", num_replicas_per_node*batches_per_second/num_replicas, epoch)
                tf.summary.scalar("train_speed/node/seconds_per_kimages", num_replicas_per_node*seconds_per_kimages/num_replicas, epoch)
                tf.summary.scalar("losses/epoch/generator", gen_loss, epoch)
                tf.summary.scalar("losses/epoch/discriminator", tf.reduce_sum(dis_loss), epoch)
                tf.summary.scalar("losses/epoch/wasserstein_disc", dis_loss[0], epoch)
                tf.summary.scalar("losses/epoch/gradient_penalty_disc", dis_loss[1], epoch)
                tf.summary.scalar("losses/epoch/epsilon_penalty_disc", dis_loss[2], epoch)
                tf.summary.scalar("losses/epoch/mixed_norms_disc", disc_gradient_mixed_norm, epoch)
                tf.summary.scalar("model/epoch/current_stage", current_stage, epoch)
                tf.summary.scalar("model/epoch/alpha", conf.model.alpha, epoch)
                tf.summary.scalar("model/epoch/global_batch_size", global_batch_size, epoch)
                tf.summary.scalar("model/epoch/node_batch_size", num_replicas_per_node*replica_batch_size, epoch)
                tf.summary.scalar("model/epoch/replica_batch_size", replica_batch_size, epoch)
                tf.summary.scalar("model/epoch/buffer_size", conf.data.buffer_sizes[current_stage], epoch)
                tf.summary.scalar("model/epoch/steps_per_epoch", steps_per_epoch, epoch)
                tf.summary.scalar("model/epoch/num_replicas", num_replicas, epoch)
                tf.summary.scalar("model/epoch/num_nodes", conf.general.nnodes, epoch)
                tf.summary.scalar("optimizers/epoch/discriminator_learning_rate", optimizer_dis.lr, epoch)
                tf.summary.scalar("optimizers/epoch/generator_learning_rate", optimizer_gen.lr, epoch)

        # chief saves eval images
        if conf.general.is_chief and conf.general.evaluate and conf.general.eval_freq and (epoch + 1) % conf.general.eval_freq == 0:
            n = np.minimum(replica_batch_size, len(random_noise))  # else some GPUs might OOM for large resolutions
            save_eval_images(random_noise[:n], generator, epoch, conf.general.out_dir, tf.constant(conf.model.alpha), data_format=conf.model.data_format)
            save_eval_images(random_noise[:n], final_gen, epoch, conf.general.out_dir, tf.constant(1.0), current_stage, data_format=conf.model.data_format)

        # chief saves model checkpoints
        if conf.general.is_chief and conf.general.save and conf.general.checkpoint_freq and (epoch + 1) % conf.general.checkpoint_freq == 0:
            s = 'x'.join([str(x) for x in image_shape])
            gen_file = os.path.join(conf.general.out_dir, f"cp_{generator.name}_epoch-{epoch+1:04d}_shape-{s}.h5")
            dis_file = os.path.join(conf.general.out_dir, f"cp_{discriminator.name}_epoch-{epoch+1:04d}_shape-{s}.h5")
            fin_file = os.path.join(conf.general.out_dir, f"cp_{final_gen.name}_epoch-{epoch+1:04d}.h5")
            generator.save(filepath=gen_file)
            discriminator.save(filepath=dis_file)
            final_gen.save(filepath=fin_file)

        # all workers check for stage increase
        if (epoch + 1) % conf.train.epochs_per_stage == 0 and current_stage < conf.model.final_stage:
            # log progress, increment stage
            current_time = time.time()
            stage_duration = str(timedelta(seconds=current_time - stage_start_time))
            train_duration = str(timedelta(seconds=current_time - train_start_time))
            logging.info(f"Completed stage={current_stage} in {stage_duration}, total_train_time={train_duration}")
            stage_start_time = current_time
            current_stage += 1

            # get dataset pipeline for new images, instantiate next stage models, get new optimizers
            _gen, _dis, train_dataset, optimizer_gen, optimizer_dis = instantiate_stage_objects(current_stage)

            # extract image_shape from train_dataset's element_spec (which is depending on distribution strategy)
            if isinstance(train_dataset.element_spec, tf.TensorSpec):
                image_shape = train_dataset.element_spec.shape[1:]
            elif isinstance(train_dataset.element_spec, tf.python.distribute.values.PerReplicaSpec):
                image_shape = train_dataset.element_spec._value_specs[0].shape[1:]
            else:
                raise RuntimeError(f"dataset {train_dataset} returns unknown element_spec of type {type(train_dataset.element_spec)}")

            # transfer weights from previous stage models to current_stage models
            transfer_weights(source_model=generator, target_model=_gen, beta=0.0)
            transfer_weights(source_model=discriminator, target_model=_dis, beta=0.0)

            # clear previous stage models, collect with gc
            del generator
            del discriminator
            gc.collect()  # note: this only cleans the python runtime not keras/tensorflow backend nor GPU memory
            generator = _gen
            discriminator = _dis

            # compile train_step_fn depending on run mode configuration
            train_step_fn = global_train_step if conf.general.train_eagerly else tf.function(global_train_step, experimental_compile=conf.general.XLA)

            # update variables, counters/tqdm postfix
            replica_batch_size = conf.data.replica_batch_sizes[current_stage]
            global_batch_size = replica_batch_size * conf.general.strategy.num_replicas_in_sync
            steps_per_epoch = int(conf.data.num_examples // global_batch_size)
            conf.model.alpha = conf.train.alpha_init
            epochs.set_description_str(f"Progressive-GAN(stage={current_stage}, shape={image_shape}")
            logging.info(f"Starting to train Stage {current_stage}")

        # update log files and tqdm status message
        _str_duration = str(timedelta(seconds=epoch_duration))
        status_message = f"duration={_str_duration}, gen_loss={gen_loss:.3f}, dis_loss={tf.reduce_sum(dis_loss):.3f}"
        logging.info(f"Finished epoch-{epoch+1:04d} with {status_message}")
        epochs.set_postfix_str(status_message)

    # final
    current_time = time.time()
    stage_duration = str(timedelta(seconds=current_time - stage_start_time))
    train_duration = str(timedelta(seconds=current_time - train_start_time))
    logging.info(f"Completed stage={current_stage} in {stage_duration}, total_train_time={train_duration}")
    logging.info(f"Successfully completed training {conf.train.epochs} epochs in {train_duration}")
