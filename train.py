import os
import gc
import time
import logging
from typing import Tuple
from datetime import timedelta

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.utils import plot_model

from config import conf
from data import get_dataset_pipeline
from model import generator_paper, discriminator_paper
from losses import wasserstein_discriminator_loss, wasserstein_generator_loss, discriminator_epsilon_drift, \
    wasserstein_gradient_penalty
from utils import save_eval_images, transfer_weights

generator = None
final_gen = None
discriminator = None
optimizer_gen = None
optimizer_dis = None
global_batch_size = None
train_step_fn = None


def replica_train_step(batch: tf.Tensor, alpha: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Trains Progressive GAN for one batch.

    Args:
        batch: a 4D-Tensor containing the images to train with
        alpha: a scalar depicting the current alpha for smoothing images of Progressive GANs

    Returns:
        A tuple of length two containing batch train information:
            gen_loss (float): mean generator loss
            disc_loss (tf.Tensor): mean discriminator loss shape=(3,) with wasserstein_loss, gradient_loss, epsilon_loss
    """
    # generate noise for projecting fake images
    _batch_size = tf.shape(batch)[0]
    noise = tf.random.normal([_batch_size, conf.model.noise_dim])
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


def node_train_step(batch: tf.Tensor, alpha: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    if not conf.general.train_eagerly:
        logging.info(f'tf.function tracing train_step: batch={batch}, alpha={alpha}')
    per_replica_losses = conf.general.strategy.experimental_run_v2(replica_train_step, args=(batch, alpha))
    return conf.general.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


def epoch_step(
        # generator: tf.keras.Model,
        # final_generator: tf.keras.Model,
        # discriminator: tf.keras.Model,
        dataset: tf.data.Dataset,
        # optimizer_gen: tf.optimizers.Optimizer,
        # optimizer_dis: tf.optimizers.Optimizer,
        current_epoch: int,
        num_steps: int) -> Tuple[float, tf.Tensor, float]:
    """Trains Progressive GAN for one epoch.
    
    Args:
        generator: generator model of the GAN
        final_generator: exponential moving average of generator across all progressive stages
        discriminator: discriminator model of the GAN
        dataset: dataset to train one epoch on
        optimizer_gen: optimizer to train generator with
        optimizer_dis: optimizer to train discriminator with
        current_epoch: current epoch
        num_steps: number of steps (batches to train for) in current epoch

    Returns:
        A tuple of length three containing epoch train information:
            gen_loss (float): mean generator loss
            disc_loss (tf.Tensor): mean discriminator loss shape=(3,) with wasserstein_loss, gradient_loss, epsilon_loss
            image_count (float): number of images GAN was trained with in epoch
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
    global optimizer_gen, optimizer_dis, global_batch_size
    # create optimizers with new learning rates
    optimizer_gen = tf.keras.optimizers.Adam(
        learning_rate=conf.optimizer.learning_rates[stage],
        beta_1=conf.optimizer.beta1,
        beta_2=conf.optimizer.beta2,
        epsilon=conf.optimizer.epsilon,
        name='adam_generator')
    optimizer_dis = tf.keras.optimizers.Adam(
        learning_rate=conf.optimizer.learning_rates[stage],
        beta_1=conf.optimizer.beta1,
        beta_2=conf.optimizer.beta2,
        epsilon=conf.optimizer.epsilon,
        name='adam_discriminator')

    # construct dataset
    global_batch_size = conf.data.replica_batch_sizes[stage] * conf.general.strategy.num_replicas_in_sync
    dataset = get_dataset_pipeline(name=f"{conf.data.registered_name}/{2**stage}", batch_size=global_batch_size,
                                   buffer_size=conf.data.buffer_sizes[stage], **conf.data)
    # options = tf.data.Options()
    # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    # dataset = dataset.with_options(options)
    dataset = conf.general.strategy.experimental_distribute_dataset(dataset)

    # create models
    gen = generator_paper(stop_stage=stage, name=f"generator_stage_{stage}", **conf.model)
    dis = discriminator_paper(stop_stage=stage, name=f"discriminator_stage_{stage}", **conf.model)

    # logging, plotting, ship
    logging.info(f"Successfully instantiated {dis.name} and {gen.name} for stage={stage}")
    gen.summary(print_fn=logging.info, line_length=150, positions=[.33, .55, .67, 1.])
    dis.summary(print_fn=logging.info, line_length=150, positions=[.33, .55, .67, 1.])
    if conf.general.save:
        plot_model(gen, os.path.join(conf.general.out_dir, f"net_{gen.name}.png"), True, False, dpi=178)
        plot_model(dis, os.path.join(conf.general.out_dir, f"net_{dis.name}.png"), True, False, dpi=178)
    return gen, dis, dataset, optimizer_gen, optimizer_dis


def train():
    global optimizer_gen, optimizer_dis, global_batch_size, final_gen, generator, discriminator, train_step_fn
    # set optimizer settings
    # tf.config.optimizer.set_jit(conf.general.XLA)

    # instantiate target model that will be an exponential moving average of generator, log summary
    with conf.general.strategy.scope():
        final_gen = generator_paper(stop_stage=conf.model.final_stage, return_all_outputs=True, **conf.model)
    logging.info(f"Successfully instantiated the following final model {final_gen.name}")
    final_gen.summary(print_fn=logging.info, line_length=150, positions=[.33, .55, .67, 1.])
    if conf.general.save:
        plot_model(final_gen, os.path.join(conf.general.out_dir, f"net_{final_gen.name}.png"), True, False, dpi=178)

    # instantiate initial stage trainable models, optimizers and dataset
    current_stage = 2 if conf.model.use_stages else conf.model.final_stage
    with conf.general.strategy.scope():
        generator, discriminator, train_dataset, optimizer_gen, optimizer_dis = instantiate_stage_objects(current_stage)
    transfer_weights(source_model=generator, target_model=final_gen, beta=0.0)  # force same initialization
    if isinstance(train_dataset.element_spec, tf.TensorSpec):
        image_shape = train_dataset.element_spec.shape[1:]
    elif isinstance(train_dataset.element_spec, tf.python.distribute.values.PerReplicaSpec):
        image_shape = train_dataset.element_spec._value_specs[0].shape[1:]
    else:
        raise RuntimeError()

    # epoch iterator
    epochs = tqdm(range(conf.train.epochs), f"Progressive-GAN(stage={current_stage}, shape={image_shape}", unit='epoch')
    logging.info(f"Starting to train Stage {current_stage}")

    # variables, counters, timings, etc.
    total_image_count = 0
    replica_batch_size = conf.data.replica_batch_sizes[current_stage]
    global_batch_size = replica_batch_size * conf.general.strategy.num_replicas_in_sync
    steps_per_epoch = int(conf.data.num_examples // global_batch_size) + 1
    stage_start_time = train_start_time = time.time()
    random_noise = tf.random.normal(shape=(16, conf.model.noise_dim), seed=1000)

    # compile tf.function
    # input_signature = [tf.TensorSpec(shape=train_dataset.element_spec.shape, dtype=tf.float32), tf.TensorSpec(shape=tuple(), dtype=tf.float32)]
    train_step_fn = node_train_step if conf.general.train_eagerly else tf.function(node_train_step, experimental_compile=conf.general.XLA)

    for epoch in epochs:
        # make an epoch step
        epoch_start_time = time.time()
        gen_loss, dis_loss, image_count = epoch_step(train_dataset, epoch, steps_per_epoch)
        epoch_duration = time.time() - epoch_start_time
        total_image_count += int(image_count)

        # TensorBoard logging
        if conf.general.is_chief and conf.general.logging and conf.general.log_freq and (epoch + 1) % conf.general.log_freq == 0:
            with conf.general.summary.as_default():
                batches_per_second = tf.cast(steps_per_epoch, tf.float32) / epoch_duration
                disc_gradient_penalty = dis_loss[1] * (conf.train.wgan_target ** 2) / conf.train.wgan_lambda
                disc_gradient_mixed_norm = np.sqrt(disc_gradient_penalty + 1e-8) + conf.train.wgan_target
                tf.summary.scalar(name="train_speed/duration", data=epoch_duration, step=epoch)
                tf.summary.scalar(name="train_speed/images_per_second", data=image_count/epoch_duration, step=epoch)
                tf.summary.scalar(name="train_speed/seconds_per_kimages", data=1000*epoch_duration/image_count, step=epoch)
                tf.summary.scalar(name="train_speed/batches_per_second", data=batches_per_second, step=epoch)
                tf.summary.scalar(name="train_speed/total_image_count", data=total_image_count, step=epoch)
                tf.summary.scalar(name="losses/epoch/generator", data=gen_loss, step=epoch)
                tf.summary.scalar(name="losses/epoch/discriminator", data=tf.reduce_sum(dis_loss), step=epoch)
                tf.summary.scalar(name="losses/epoch/wasserstein_disc", data=dis_loss[0], step=epoch)
                tf.summary.scalar(name="losses/epoch/gradient_penalty_disc", data=dis_loss[1], step=epoch)
                tf.summary.scalar(name="losses/epoch/epsilon_penalty_disc", data=dis_loss[2], step=epoch)
                tf.summary.scalar(name="losses/epoch/mixed_norms_disc", data=disc_gradient_mixed_norm, step=epoch)
                tf.summary.scalar(name="model/epoch/current_stage", data=current_stage, step=epoch)
                tf.summary.scalar(name="model/epoch/alpha", data=conf.model.alpha, step=epoch)
                tf.summary.scalar(name="model/epoch/batch_size", data=global_batch_size, step=epoch)
                tf.summary.scalar(name="model/epoch/replica_batch_size", data=replica_batch_size, step=epoch)
                tf.summary.scalar(name="model/epoch/buffer_size", data=conf.data.buffer_sizes[current_stage], step=epoch)
                tf.summary.scalar(name="model/epoch/steps_per_epoch", data=steps_per_epoch, step=epoch)
                tf.summary.scalar(name="optimizers/epoch/discriminator_learning_rate", data=optimizer_dis.lr, step=epoch)
                tf.summary.scalar(name="optimizers/epoch/generator_learning_rate", data=optimizer_gen.lr, step=epoch)

        # save eval images
        if conf.general.is_chief and conf.general.evaluate and conf.general.eval_freq and (epoch + 1) % conf.general.eval_freq == 0:
            save_eval_images(random_noise, generator, epoch, conf.general.out_dir, tf.constant(conf.model.alpha))
            save_eval_images(random_noise, final_gen, epoch, conf.general.out_dir, tf.constant(1.0), stage=current_stage)

        # save model checkpoints
        if conf.general.is_chief and conf.general.save and conf.general.checkpoint_freq and (epoch + 1) % conf.general.checkpoint_freq == 0:
            shape = 'x'.join([str(x) for x in image_shape])
            gen_file = os.path.join(conf.general.out_dir, f"cp_{generator.name}_epoch-{epoch+1:04d}_shape-{shape}.h5")
            dis_file = os.path.join(conf.general.out_dir, f"cp_{discriminator.name}_epoch-{epoch+1:04d}_shape-{shape}.h5")
            fin_file = os.path.join(conf.general.out_dir, f"cp_{final_gen.name}_epoch-{epoch+1:04d}.h5")
            generator.save(filepath=gen_file)
            discriminator.save(filepath=dis_file)
            final_gen.save(filepath=fin_file)

        # update log files and tqdm status message
        _str_duration = str(timedelta(seconds=epoch_duration))
        status_message = f"duration={_str_duration}, gen_loss={gen_loss:.3f}, dis_loss={tf.reduce_sum(dis_loss):.3f}"
        logging.info(f"Finished epoch-{epoch+1:04d} with {status_message}")
        epochs.set_postfix_str(status_message)

        # check stage increase
        if (epoch + 1) % conf.train.epochs_per_stage == 0 and current_stage < conf.model.final_stage:
            # increment stages, log progress
            current_stage += 1
            current_time = time.time()
            stage_duration = str(timedelta(seconds=current_time - stage_start_time))
            train_duration = str(timedelta(seconds=current_time - train_start_time))
            logging.info(f"Completed stage={current_stage} in {stage_duration}, total_train_time={train_duration}")
            stage_start_time = current_time

            # get dataset pipeline for new images, instantiate next stage models, get new optimizers
            with conf.general.strategy.scope():
                _gen, _dis, train_dataset, optimizer_gen, optimizer_dis = instantiate_stage_objects(current_stage)
            if isinstance(train_dataset.element_spec, tf.TensorSpec):
                image_shape = train_dataset.element_spec.shape[1:]
            elif isinstance(train_dataset.element_spec, tf.python.distribute.values.PerReplicaSpec):
                image_shape = train_dataset.element_spec._value_specs[0].shape[1:]
            else:
                raise RuntimeError()

            # transfer weights from previous stage models to current_stage models
            transfer_weights(source_model=generator, target_model=_gen, beta=0.0)
            transfer_weights(source_model=discriminator, target_model=_dis, beta=0.0)

            # clear previous stage models, collect with gc
            del generator
            del discriminator
            gc.collect()  # note: this only cleans the python runtime not keras/tensorflow backend nor GPU memory
            generator = _gen
            discriminator = _dis

            # compile tf.function
            # input_signature = [tf.TensorSpec(shape=train_dataset.element_spec.shape, dtype=tf.float32), tf.TensorSpec(shape=tuple(), dtype=tf.float32)]
            train_step_fn = node_train_step if conf.general.train_eagerly else tf.function(node_train_step, experimental_compile=conf.general.XLA)

            # update variables, counters/tqdm postfix
            replica_batch_size = conf.data.replica_batch_sizes[current_stage]
            global_batch_size = replica_batch_size * conf.general.strategy.num_replicas_in_sync
            steps_per_epoch = int(conf.data.num_examples // global_batch_size) + 1
            conf.model.alpha = conf.train.alpha_init
            epochs.set_description_str(f"Progressive-GAN(stage={current_stage}, shape={image_shape}")
            logging.info(f"Starting to train Stage {current_stage}")

    # final
    current_time = time.time()
    stage_duration = str(timedelta(seconds=current_time - stage_start_time))
    train_duration = str(timedelta(seconds=current_time - train_start_time))
    logging.info(f"Completed stage={current_stage} in {stage_duration}, total_train_time={train_duration}")
    logging.info(f"Successfully completed training {conf.train.epochs} epochs in {train_duration}")
