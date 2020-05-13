import os
import gc
import time
import logging
from typing import Optional
from datetime import timedelta

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.utils import plot_model

from config import conf
from data import get_dataset_pipeline
from networks import generator_paper, discriminator_paper
from utils import save_eval_images, transfer_ema_weights
from losses import wasserstein_discriminator_loss, wasserstein_generator_loss, wasserstein_gradient_penalty, \
    discriminator_epsilon_drift


class ProgressiveGAN(tf.keras.Model):
    def __init__(self, start_stage: int = 2, final_stage: int = 10, **kwargs):
        super(ProgressiveGAN, self).__init__(**kwargs)
        self.final_stage = final_stage
        self.current_stage = start_stage
        self.alpha = tf.Variable(conf.train.alpha_init, False, dtype=tf.float32, name="model_alpha", aggregation=tf.VariableAggregation.MEAN)
        self.channel_axis = -1 if conf.model.data_format == 'channels_last' or conf.model.data_format == 'NHWC' else 1
        self.final_generator = generator_paper(stop_stage=self.final_stage, return_all_outputs=True, **conf.model)
        self.generator = self.discriminator = self.optimizer_dis = self.optimizer_gen = self.ema = None

    def compile(self, **kwargs):
        """Model is re-complied after each stage increment. Instantiates all new objects for current_stage. Transfers
        weights from previous stage models to new models."""
        # reinitialize optimizers
        self.optimizer_gen = tf.keras.optimizers.Adam(
            learning_rate=conf.optimizer.learning_rates[self.current_stage],
            beta_1=conf.optimizer.beta1,
            beta_2=conf.optimizer.beta2,
            epsilon=conf.optimizer.epsilon,
            name='adam_generator')
        self.optimizer_dis = tf.keras.optimizers.Adam(
            learning_rate=conf.optimizer.learning_rates[self.current_stage],
            beta_1=conf.optimizer.beta1,
            beta_2=conf.optimizer.beta2,
            epsilon=conf.optimizer.epsilon,
            name='adam_discriminator')

        # create new generator / discriminator for current_stage
        generator = generator_paper(stop_stage=self.current_stage, name=f'generator_stage_{self.current_stage}', **conf.model)
        discriminator = discriminator_paper(stop_stage=self.current_stage, name=f'discriminator_stage_{self.current_stage}', **conf.model)

        # transfer weights from previous stage to current stage
        if self.generator is not None and self.discriminator is not None:
            transfer_ema_weights(source_model=self.generator, target_model=generator)
            transfer_ema_weights(source_model=self.discriminator, target_model=discriminator)
            del self.generator
            del self.discriminator
            del self.ema
            gc.collect()  # note: this only cleans the python runtime not keras/tensorflow backend nor GPU memory
        else:
            # first time generator was created for start stage (use initialized weights from final_generator)
            transfer_ema_weights(source_model=self.final_generator, target_model=generator)

        # reset alpha for new stage, create a new EMA
        self.generator, self.discriminator = generator, discriminator
        self.ema = tf.train.ExponentialMovingAverage(decay=conf.model.generator_ema)
        self.alpha.assign(conf.train.alpha_init)
        self.ema.apply(self.generator.variables)
        super(ProgressiveGAN, self).compile()   # clears cached tf.function trace of train_step

    def train_step(self, data):
        if not self.run_eagerly:
            # log whenever this function is traced for graph mode execution
            logging.info(f'tf.function tracing train_step: data={data}')

        # generate noise for projecting fake images
        replica_batch_size = tf.shape(data)[0]
        noise = tf.random.normal([replica_batch_size, conf.model.noise_dim])

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            # forward pass: inference through both models on tape to create predictions
            fake_images = self.generator([noise, self.alpha], training=True)
            real_image_guesses = self.discriminator([data, self.alpha], training=True)
            fake_image_guesses = self.discriminator([fake_images, self.alpha], training=True)

            # compute gradient penalty
            if conf.train.use_gradient_penalty:
                disc_gradient_loss = wasserstein_gradient_penalty(self.discriminator, data, fake_images, conf.train.wgan_target, conf.train.wgan_lambda, self.alpha)
                disc_gradient_loss = tf.nn.compute_average_loss(disc_gradient_loss, global_batch_size=self.global_batch_size)
            else:
                disc_gradient_loss = 0.0

            # compute drift penalty
            if conf.train.use_epsilon_penalty:
                disc_epsilon_loss = discriminator_epsilon_drift(real_image_guesses, conf.train.drift_epsilon)
                disc_epsilon_loss = tf.nn.compute_average_loss(disc_epsilon_loss, global_batch_size=self.global_batch_size)
            else:
                disc_epsilon_loss = 0.0

            # calculate losses
            gen_loss = wasserstein_generator_loss(fake_image_guesses)
            gen_loss = tf.nn.compute_average_loss(gen_loss, global_batch_size=self.global_batch_size)
            _disc_loss = wasserstein_discriminator_loss(real_image_guesses, fake_image_guesses)
            _disc_loss = tf.nn.compute_average_loss(_disc_loss, global_batch_size=self.global_batch_size)
            disc_stacked_loss = tf.stack((_disc_loss, disc_gradient_loss, disc_epsilon_loss))
            disc_loss = tf.reduce_sum(disc_stacked_loss)

        # collocate gradients from tapes
        gradients_generator = generator_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_discriminator = discriminator_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        # backward pass: apply gradients via optimizers to update models
        self.optimizer_gen.apply_gradients(zip(gradients_generator, self.generator.trainable_variables))
        self.optimizer_dis.apply_gradients(zip(gradients_discriminator, self.discriminator.trainable_variables))

        # increment alpha, apply ema for moving average generator weights
        self.alpha.assign(tf.minimum(self.alpha + conf.model.alpha_step * self.global_batch_size, 1.0))
        self.ema.apply(self.generator.variables)
        return {'gen_loss': gen_loss, 'disc_loss': disc_loss, 'wgan_disc_loss': _disc_loss, 'gradient_loss': disc_gradient_loss, 'epsilon_loss': disc_epsilon_loss}

    @property
    def image_shape(self):
        resolution = 2**self.current_stage
        return (resolution, resolution, 3) if self.channel_axis == -1 else (3, resolution, resolution)

    @property
    def global_batch_size(self) -> int:
        return self.distribute_strategy.num_replicas_in_sync * conf.data.replica_batch_sizes[self.current_stage]

    @property
    def replica_batch_size(self) -> int:
        return conf.data.replica_batch_sizes[self.current_stage]


def train():
    # instantiate initial stage trainable models, optimizers and dataset
    current_stage = 2 if conf.model.use_stages else conf.model.final_stage
    with conf.general.strategy.scope():
        model = ProgressiveGAN(start_stage=current_stage, final_stage=conf.model.final_stage, name='Progressive-GAN')
        model.compile()
        train_dataset = get_dataset_pipeline(
            name=f"{conf.data.registered_name}/{2**current_stage}",
            batch_size=model.global_batch_size,
            buffer_size=conf.data.buffer_sizes[current_stage],
            **conf.data)

    # log newly created models
    if conf.general.is_chief and conf.general.logging:
        logging.info(f"Successfully instantiated {model.final_generator.name} for stage={model.final_stage}")
        model.final_generator.summary(print_fn=logging.info, line_length=150, positions=[.33, .55, .67, 1.])
        logging.info(f"Successfully instantiated {model.discriminator.name} and {model.generator.name} for stage={model.current_stage}")
        model.generator.summary(print_fn=logging.info, line_length=150, positions=[.33, .55, .67, 1.])
        model.discriminator.summary(print_fn=logging.info, line_length=150, positions=[.33, .55, .67, 1.])
    if conf.general.is_chief and conf.general.save:
        plot_model(model.final_generator, os.path.join(conf.general.out_dir, f"net_{model.final_generator.name}.png"), True, False, dpi=178)
        plot_model(model.generator, os.path.join(conf.general.out_dir, f"net_{model.generator.name}.png"), True, False, dpi=178)
        plot_model(model.discriminator, os.path.join(conf.general.out_dir, f"net_{model.discriminator.name}.png"), True, False, dpi=178)

    # iterators, variables, counters, timings, etc.
    epochs = tqdm(range(conf.train.epochs), f"{model.name}(stage={model.current_stage}, shape={model.image_shape}", unit='epoch')
    total_image_count = 0
    steps_per_epoch = int(conf.data.num_examples // model.global_batch_size)
    stage_start_time = train_start_time = time.time()
    random_noise = tf.random.normal(shape=(16, conf.model.noise_dim), seed=1000)
    logging.info(f"Starting to train Stage {model.current_stage}")

    for epoch in epochs:
        # make an epoch step
        epoch_start_time = time.time()
        losses = model.fit(train_dataset, epochs=1, steps_per_epoch=steps_per_epoch, verbose=0).history
        epoch_duration = time.time() - epoch_start_time

        # reduce losses
        for key, value in losses.items():
            losses[key] = value[0]

        # transfer weights from ema to final_generator
        transfer_ema_weights(source_model=model.generator, source_ema=model.ema, target_model=model.final_generator)
        total_image_count += conf.data.num_examples

        # chief does TensorBoard logging
        if conf.general.is_chief and conf.general.logging and conf.general.log_freq and (epoch + 1) % conf.general.log_freq == 0:
            with conf.general.summary.as_default():
                num_replicas = conf.general.strategy.num_replicas_in_sync
                num_replicas_per_node = num_replicas / conf.general.nnodes
                batches_per_second = tf.cast(steps_per_epoch, tf.float32) / epoch_duration
                images_per_second = conf.data.num_examples/epoch_duration
                seconds_per_kimages = 1000*epoch_duration/conf.data.num_examples
                disc_gradient_penalty = losses['gradient_loss'] * (conf.train.wgan_target ** 2) / conf.train.wgan_lambda
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
                tf.summary.scalar("losses/epoch/generator", losses['gen_loss'], epoch)
                tf.summary.scalar("losses/epoch/discriminator", losses['disc_loss'], epoch)
                tf.summary.scalar("losses/epoch/wasserstein_disc", losses['wgan_disc_loss'], epoch)
                tf.summary.scalar("losses/epoch/gradient_penalty_disc", losses['gradient_loss'], epoch)
                tf.summary.scalar("losses/epoch/epsilon_penalty_disc", losses['epsilon_loss'], epoch)
                tf.summary.scalar("losses/epoch/mixed_norms_disc", disc_gradient_mixed_norm, epoch)
                tf.summary.scalar("model/epoch/current_stage", model.current_stage, epoch)
                tf.summary.scalar("model/epoch/alpha", model.alpha, epoch)
                tf.summary.scalar("model/epoch/global_batch_size", model.global_batch_size, epoch)
                tf.summary.scalar("model/epoch/node_batch_size", model.replica_batch_size*num_replicas_per_node, epoch)
                tf.summary.scalar("model/epoch/replica_batch_size", model.replica_batch_size, epoch)
                tf.summary.scalar("model/epoch/buffer_size", conf.data.buffer_sizes[model.current_stage], epoch)
                tf.summary.scalar("model/epoch/steps_per_epoch", steps_per_epoch, epoch)
                tf.summary.scalar("model/epoch/num_replicas", num_replicas, epoch)
                tf.summary.scalar("model/epoch/num_nodes", conf.general.nnodes, epoch)
                tf.summary.scalar("optimizers/epoch/discriminator_learning_rate", model.optimizer_dis.lr, epoch)
                tf.summary.scalar("optimizers/epoch/generator_learning_rate", model.optimizer_gen.lr, epoch)

        # chief saves eval images TODO(M.Rozanski): rewrite as callback
        if conf.general.is_chief and conf.general.evaluate and conf.general.eval_freq and (epoch + 1) % conf.general.eval_freq == 0:
            n = np.minimum(model.replica_batch_size, len(random_noise))  # else some GPUs might OOM for large resolutions
            save_eval_images(random_noise[:n], model.generator, epoch, conf.general.out_dir, model.alpha, data_format=conf.model.data_format)
            save_eval_images(random_noise[:n], model.final_generator, epoch, conf.general.out_dir, tf.constant(1.0), current_stage, conf.model.data_format)

        # chief saves model checkpoints TODO(M.Rozanski): rewrite as callback
        if conf.general.is_chief and conf.general.save and conf.general.checkpoint_freq and (epoch + 1) % conf.general.checkpoint_freq == 0:
            s = 'x'.join([str(x) for x in model.image_shape])
            gen_file = os.path.join(conf.general.out_dir, f"cp_{model.generator.name}_epoch-{epoch+1:04d}_shape-{s}.h5")
            dis_file = os.path.join(conf.general.out_dir, f"cp_{model.discriminator.name}_epoch-{epoch+1:04d}_shape-{s}.h5")
            fin_file = os.path.join(conf.general.out_dir, f"cp_{model.final_generator.name}_epoch-{epoch+1:04d}.h5")
            model.generator.save(filepath=gen_file)
            model.discriminator.save(filepath=dis_file)
            model.final_generator.save(filepath=fin_file)

        # update log files and tqdm status message
        _str_duration = str(timedelta(seconds=epoch_duration))
        status_message = f"duration={_str_duration}, gen_loss={losses['gen_loss']:.3f}, dis_loss={losses['disc_loss']:.3f}"
        logging.info(f"Finished epoch-{epoch+1:04d} with {status_message}")
        epochs.set_postfix_str(status_message)

        # all workers check for stage increase
        if (epoch + 1) % conf.train.epochs_per_stage == 0 and model.current_stage < model.final_stage:
            # log progress, increment stage
            current_time = time.time()
            stage_duration = str(timedelta(seconds=current_time - stage_start_time))
            train_duration = str(timedelta(seconds=current_time - train_start_time))
            logging.info(f"Completed stage={current_stage} in {stage_duration}, total_train_time={train_duration}")
            stage_start_time = current_time

            # instantiate next stage models and new optimizers, get dataset pipeline for new images
            with conf.general.strategy.scope():
                model.current_stage += 1
                model.compile()
                train_dataset = get_dataset_pipeline(
                    name=f"{conf.data.registered_name}/{2**model.current_stage}",
                    batch_size=model.global_batch_size,
                    buffer_size=conf.data.buffer_sizes[model.current_stage],
                    **conf.data)

            # log newly created models
            if conf.general.is_chief and conf.general.logging:
                logging.info(f"Successfully instantiated {model.discriminator.name} and {model.generator.name} for stage={model.current_stage}")
                model.generator.summary(print_fn=logging.info, line_length=150, positions=[.33, .55, .67, 1.])
                model.discriminator.summary(print_fn=logging.info, line_length=150, positions=[.33, .55, .67, 1.])
            if conf.general.is_chief and conf.general.save:
                plot_model(model.generator, os.path.join(conf.general.out_dir, f"net_{model.generator.name}.png"), True, False, dpi=178)
                plot_model(model.discriminator, os.path.join(conf.general.out_dir, f"net_{model.discriminator.name}.png"), True, False, dpi=178)

            # update variables, counters/tqdm postfix
            steps_per_epoch = int(conf.data.num_examples // model.global_batch_size)
            epochs.set_description_str(f"Progressive-GAN(stage={model.current_stage}, shape={model.image_shape}")
            logging.info(f"Starting to train Stage {model.current_stage}")

    # final
    current_time = time.time()
    stage_duration = str(timedelta(seconds=current_time - stage_start_time))
    train_duration = str(timedelta(seconds=current_time - train_start_time))
    logging.info(f"Completed stage={model.current_stage} in {stage_duration}, total_train_time={train_duration}")
    logging.info(f"Successfully completed training {conf.train.epochs} epochs in {train_duration}")
