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
from data import get_dataset_pipeline, celeb_a_hq_process_func
from model import generator_paper, discriminator_paper
from losses import discriminator_loss, generator_loss
from utils import save_eval_images, transfer_weights


def train():
    # set optimizer settings
    tf.config.optimizer.set_jit(conf.general.XLA)

    with conf.general.strategy.scope():
        # instantiate optimizers
        optimizer_gen = tf.keras.optimizers.Adam(
            learning_rate=conf.optimizer.learning_rates[2],
            beta_1=conf.optimizer.beta1,
            beta_2=conf.optimizer.beta2,
            epsilon=conf.optimizer.epsilon,
            name='adam_generator')
        optimizer_dis = tf.keras.optimizers.Adam(
            learning_rate=conf.optimizer.learning_rates[2],
            beta_1=conf.optimizer.beta1,
            beta_2=conf.optimizer.beta2,
            epsilon=conf.optimizer.epsilon,
            name='adam_discriminator')

        # instantiate target models to train, log summary
        alpha_step_per_image = (1.0 - conf.train.alpha_init) / (conf.train.epochs_per_stage * 30000 / 2)
        final_gen = generator_paper(stop_stage=conf.model.final_stage, return_all_outputs=True, **conf.model)
        model_dis = discriminator_paper(stop_stage=conf.model.final_stage, **conf.model)
        model_gen = final_gen
        logging.info(f"Successfully instantiated the following final models {model_dis.name} and {model_gen.name}")
        model_gen.summary(print_fn=logging.info, line_length=150, positions=[.33, .55, .67, 1.])
        model_dis.summary(print_fn=logging.info, line_length=150, positions=[.33, .55, .67, 1.])
        plot_model(final_gen, os.path.join(conf.general.out_dir, f"net_{final_gen.name}.png"), True, False, dpi=178)
        plot_model(model_dis, os.path.join(conf.general.out_dir, f"net_{model_dis.name}.png"), True, False, dpi=178)

        # random noise for image eval
        random_noise = tf.random.normal(shape=(16, conf.model.noise_dim), seed=1000)

    def train_step(image_batch: tf.Tensor, local_batch_size: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # generate noise for projecting fake images
        noise = tf.random.normal([local_batch_size, conf.model.noise_dim])

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            # forward pass: inference through both models on tape to create predictions
            fake_images = model_gen([noise, conf.model.alpha], training=True)
            real_image_guesses = model_dis([image_batch, conf.model.alpha], training=True)
            fake_image_guesses = model_dis([fake_images, conf.model.alpha], training=True)

            # compute gradient penalty: create mixed images for gradient loss
            if conf.train.use_gradient_penalty:
                mixing_factors = tf.random.uniform([local_batch_size, 1, 1, 1], 0.0, 1.0)
                mixed_images = image_batch + (fake_images - image_batch) * mixing_factors
                with tf.GradientTape(watch_accessed_variables=False) as mixed_tape:
                    mixed_tape.watch(mixed_images)
                    mixed_output = model_dis([mixed_images, conf.model.alpha], training=True)
                gradient_mixed = mixed_tape.gradient(mixed_output, mixed_images)
                gradient_mixed_norm = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(gradient_mixed), axis=[1, 2, 3]))
                gradient_penalty = tf.reduce_mean(tf.square(gradient_mixed_norm - conf.train.wgan_target))
                gradient_loss = gradient_penalty * (conf.train.wgan_lambda / (conf.train.wgan_target ** 2))
            else:
                gradient_loss = 0.0

            # calculate losses
            _gen_loss = generator_loss(fake_output=fake_image_guesses)
            _disc_ws_loss, _disc_eps_loss = discriminator_loss(
                real_output=real_image_guesses,
                fake_output=fake_image_guesses,
                wgan_epsilon=conf.train.drift_epsilon,
                use_epsilon_drift=conf.train.use_epsilon_penalty)
            _disc_stacked_loss = tf.stack((_disc_ws_loss, gradient_loss, _disc_eps_loss))
            _disc_loss = tf.reduce_sum(_disc_stacked_loss)

        # collocate gradients from tapes
        gradients_generator = generator_tape.gradient(_gen_loss, model_gen.trainable_variables)
        gradients_discriminator = discriminator_tape.gradient(_disc_loss, model_dis.trainable_variables)
        # backward pass: apply gradients via optimizers to update models
        optimizer_gen.apply_gradients(zip(gradients_generator, model_gen.trainable_variables))
        optimizer_dis.apply_gradients(zip(gradients_discriminator, model_dis.trainable_variables))
        return _gen_loss, _disc_stacked_loss

    def epoch_step(dataset: tf.data.Dataset, num_epoch: int, num_steps: int) -> Tuple[float, tf.Tensor, float]:
        # return metrics
        _epoch_gen_loss, _epoch_dis_loss, _image_count, _current_step = 0.0, 0.0, 0.0, 0

        # epoch iterable, chief iterates over tqdm for status prints - all other workers over tf.data.Dataset
        dataset = tqdm(iterable=dataset, desc=f"epoch-{num_epoch+1:04d}", unit="batch", total=num_steps, leave=False)

        for image_batch in dataset:
            batch_size = tf.shape(image_batch)[0]
            batch_gen_loss, batch_dis_loss = train_step(image_batch=image_batch, local_batch_size=batch_size)

            # smooth available weights from current_stage model_gen into final generator
            transfer_weights(source_model=model_gen, target_model=final_gen, beta=conf.model.generator_ema)

            # compute moving average of loss metrics
            _size = tf.cast(batch_size, tf.float32)
            _epoch_gen_loss = (_image_count * _epoch_gen_loss + _size * batch_gen_loss) / (_image_count + _size)
            _epoch_dis_loss = (_image_count * _epoch_dis_loss + _size * batch_dis_loss) / (_image_count + _size)
            _image_count += _size

            # TensorBoard logging
            if conf.general.logging and conf.general.log_freq == 'batch':
                _step = num_epoch * num_steps + _current_step
                tf.summary.scalar(name="losses/batch/generator", data=batch_gen_loss, step=_step)
                tf.summary.scalar(name="losses/batch/discriminator", data=tf.reduce_sum(batch_dis_loss), step=_step)
                tf.summary.scalar(name="losses/batch/discriminator_wgan", data=batch_dis_loss[0], step=_step)
                tf.summary.scalar(name="losses/batch/discriminator_gp", data=batch_dis_loss[1], step=_step)
                tf.summary.scalar(name="losses/batch/discriminator_eps", data=batch_dis_loss[2], step=_step)
                tf.summary.scalar(name="model/batch/alpha", data=conf.model.alpha, step=_step)

            # increase alpha
            if conf.model.use_alpha_smoothing:
                conf.model.alpha = tf.minimum(conf.model.alpha + alpha_step_per_image * _size, 1.0)

            # additional chief tasks during training
            message = f"batch_gen_loss={batch_gen_loss:.3f}, batch_dis_loss={tf.reduce_sum(batch_dis_loss):.3f}"
            dataset.set_postfix_str(message)
            logging.debug(message)
            _current_step += 1

        return _epoch_gen_loss, _epoch_dis_loss, _image_count

    # train loop
    current_stage = 2 if conf.model.use_stages else conf.model.stop_stage
    epochs = tqdm(iterable=range(conf.train.epochs), desc='Progressive-GAN', unit='epoch')
    global_batch_size = conf.data.replica_batch_sizes[current_stage] * conf.general.strategy.num_replicas_in_sync

    train_dataset, num_examples = get_dataset_pipeline(
        name=f"{conf.data.registered_name}/{2**current_stage}",
        batch_size=global_batch_size,
        buffer_size=conf.data.buffer_sizes[current_stage],
        process_func=celeb_a_hq_process_func,
        **conf.data)
    image_shape = train_dataset.element_spec.shape[1:]
    model_gen = generator_paper(stop_stage=current_stage, name=f"generator_stage_{current_stage}", **conf.model)
    model_dis = discriminator_paper(stop_stage=current_stage, name=f"discriminator_stage_{current_stage}", **conf.model)
    transfer_weights(source_model=model_gen, target_model=final_gen, beta=0.0)  # force same initialization
    logging.info(f"Successfully instantiated {model_dis.name} and {model_gen.name} for stage={current_stage}")
    model_gen.summary(print_fn=logging.info, line_length=150, positions=[.33, .55, .67, 1.])
    model_dis.summary(print_fn=logging.info, line_length=150, positions=[.33, .55, .67, 1.])
    plot_model(model_gen, os.path.join(conf.general.out_dir, f"net_{model_gen.name}.png"), True, False, dpi=178)
    plot_model(model_dis, os.path.join(conf.general.out_dir, f"net_{model_dis.name}.png"), True, False, dpi=178)
    epochs.set_description_str(f"Progressive-GAN(stage={current_stage}, shape={image_shape}")
    logging.info(f"Starting to train Stage {current_stage}")

    # counters, timings, etc.
    total_image_count = 0
    steps_per_epoch = int(num_examples // global_batch_size) + 1
    stage_start_time = time.time()
    train_start_time = stage_start_time

    for epoch in epochs:
        # make an epoch step
        epoch_start_time = time.time()
        gen_loss, dis_loss, image_count = epoch_step(train_dataset, epoch, steps_per_epoch)
        epoch_duration = time.time() - epoch_start_time
        total_image_count += int(image_count)

        # TensorBoard logging
        if conf.general.logging and conf.general.log_freq:
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
            tf.summary.scalar(name="losses/epoch/mixed_norms_disc", data=disc_gradient_mixed_norm, step=epoch)
            tf.summary.scalar(name="losses/epoch/epsilon_penalty_disc", data=dis_loss[2], step=epoch)
            tf.summary.scalar(name="model/epoch/current_stage", data=current_stage, step=epoch)
            tf.summary.scalar(name="model/epoch/alpha", data=conf.model.alpha, step=epoch)
            tf.summary.scalar(name="model/epoch/batch_size", data=global_batch_size, step=epoch)
            tf.summary.scalar(name="model/epoch/replica_batch_size", data=conf.data.replica_batch_sizes[current_stage], step=epoch)
            tf.summary.scalar(name="model/epoch/buffer_size", data=conf.data.buffer_sizes[current_stage], step=epoch)
            tf.summary.scalar(name="model/epoch/steps_per_epoch", data=steps_per_epoch, step=epoch)
            tf.summary.scalar(name="optimizers/epoch/discriminator_learning_rate", data=optimizer_dis.lr, step=epoch)
            tf.summary.scalar(name="optimizers/epoch/generator_learning_rate", data=optimizer_gen.lr, step=epoch)

        # save eval images
        if conf.general.evaluate and conf.general.eval_freq and (epoch + 1) % conf.general.eval_freq == 0:
            save_eval_images(random_noise, model_gen, epoch, conf.general.out_dir, alpha=conf.model.alpha)
            save_eval_images(random_noise, final_gen, epoch, conf.general.out_dir, stage=current_stage)

        # save model checkpoints
        if conf.general.save and conf.general.checkpoint_freq and (epoch + 1) % conf.general.checkpoint_freq == 0:
            shape = 'x'.join([str(x) for x in image_shape])
            gen_file = os.path.join(conf.general.out_dir, f"cp_{model_gen.name}_epoch-{epoch+1:04d}_shape-{shape}.h5")
            dis_file = os.path.join(conf.general.out_dir, f"cp_{model_dis.name}_epoch-{epoch+1:04d}_shape-{shape}.h5")
            fin_file = os.path.join(conf.general.out_dir, f"cp_{final_gen.name}_epoch-{epoch+1:04d}.h5")
            model_gen.save(filepath=gen_file)
            model_dis.save(filepath=dis_file)
            final_gen.save(filepath=fin_file)

        # update log files and tqdm status message
        _str_duration = str(timedelta(seconds=epoch_duration))
        status_message = f"duration={_str_duration}, gen_loss={gen_loss:.3f}, dis_loss={tf.reduce_sum(dis_loss):.3f}"
        logging.info(f"Finished epoch-{epoch+1:04d} with {status_message}")
        epochs.set_postfix_str(status_message)

        # check stage increase
        if (epoch + 1) % conf.train.epochs_per_stage == 0 and current_stage < conf.model.final_stage:
            current_time = time.time()
            stage_duration = str(timedelta(seconds=current_time - stage_start_time))
            train_duration = str(timedelta(seconds=current_time - train_start_time))
            logging.info(f"Completed stage={current_stage} in {stage_duration}, total_train_time={train_duration}")
            # increment stage, get dataset pipeline for new images, instantiate next stage models
            current_stage += 1
            global_batch_size = conf.data.replica_batch_sizes[current_stage]*conf.general.strategy.num_replicas_in_sync
            train_dataset, num_examples = get_dataset_pipeline(
                name=f"{conf.data.registered_name}/{2**current_stage}",
                batch_size=global_batch_size,
                buffer_size=conf.data.buffer_sizes[current_stage],
                process_func=celeb_a_hq_process_func,
                **conf.data)
            image_shape = train_dataset.element_spec.shape[1:]
            g = generator_paper(stop_stage=current_stage, name=f"generator_stage_{current_stage}", **conf.model)
            d = discriminator_paper(stop_stage=current_stage, name=f"discriminator_stage_{current_stage}", **conf.model)
            optimizer_gen = tf.keras.optimizers.Adam(
                learning_rate=conf.optimizer.learning_rates[current_stage],
                beta_1=conf.optimizer.beta1,
                beta_2=conf.optimizer.beta2,
                epsilon=conf.optimizer.epsilon,
                name='adam_generator')
            optimizer_dis = tf.keras.optimizers.Adam(
                learning_rate=conf.optimizer.learning_rates[current_stage],
                beta_1=conf.optimizer.beta1,
                beta_2=conf.optimizer.beta2,
                epsilon=conf.optimizer.epsilon,
                name='adam_discriminator')
            # transfer weights from previous stage models to current_stage models
            transfer_weights(source_model=model_gen, target_model=g)
            transfer_weights(source_model=model_dis, target_model=d)
            # clear previous stage models, collect with gc
            del model_gen
            del model_dis
            gc.collect()  # note: this only cleans the python runtime not keras/tensorflow backend nor GPU memory
            model_gen = g
            model_dis = d
            model_gen.summary(print_fn=logging.info, line_length=150, positions=[.33, .55, .67, 1.])
            model_dis.summary(print_fn=logging.info, line_length=150, positions=[.33, .55, .67, 1.])
            plot_model(model_gen, os.path.join(conf.general.out_dir, f"net_{model_gen.name}.png"), True, False, dpi=178)
            plot_model(model_dis, os.path.join(conf.general.out_dir, f"net_{model_dis.name}.png"), True, False, dpi=178)
            save_eval_images(random_noise, model_gen, epoch, conf.general.out_dir, prefix=f'stage-{current_stage}_')
            # update counters/tqdm postfix
            conf.model.alpha = conf.train.alpha_init
            steps_per_epoch = int(num_examples // global_batch_size) + 1
            stage_start_time = time.time()
            epochs.set_description_str(f"Progressive-GAN(stage={current_stage}, shape={image_shape}")
            logging.info(f"Starting to train Stage {current_stage}")

    # final
    train_duration = str(timedelta(seconds=time.time() - train_start_time))
    logging.info(f"Successfully completed training {conf.train.epochs} epochs in {train_duration}")
