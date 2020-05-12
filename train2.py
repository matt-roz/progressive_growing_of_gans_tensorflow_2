import os
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
from model import generator_paper, ProgressiveGAN
from utils import save_eval_images, transfer_weights


def transfer_ema_weights(
        source_model: tf.keras.Model,
        target_model: tf.keras.Model,
        source_ema: Optional[tf.train.ExponentialMovingAverage] = None,
        layer_name_prefix: str = '') -> None:
    for source_layer in source_model.layers:
        source_vars = source_layer.variables
        if source_layer.name.startswith(layer_name_prefix) and source_vars:
            try:
                target_layer = target_model.get_layer(name=source_layer.name)
            except ValueError:
                continue
            for source_var, target_var in zip(source_vars, target_layer.variables):
                if source_ema is not None:
                    transfer_var = source_ema.average(source_var)
                else:
                    transfer_var = source_var
                target_var.assign(transfer_var)


def train():
    # instantiate target model that will be an exponential moving average of generator, log summary
    with conf.general.strategy.scope():
        final_gen = generator_paper(stop_stage=conf.model.final_stage, return_all_outputs=True, **conf.model)
    logging.info(f"Successfully instantiated the following final model {final_gen.name}")
    final_gen.summary(print_fn=logging.info, line_length=150, positions=[.33, .55, .67, 1.])
    if conf.general.is_chief and conf.general.save:
        plot_model(final_gen, os.path.join(conf.general.out_dir, f"net_{final_gen.name}.png"), True, False, dpi=178)

    # instantiate initial stage trainable models, optimizers and dataset
    current_stage = 2 if conf.model.use_stages else conf.model.final_stage
    with conf.general.strategy.scope():
        batch_size = conf.data.replica_batch_sizes[current_stage] * conf.general.strategy.num_replicas_in_sync
        train_dataset = get_dataset_pipeline(name=f"{conf.data.registered_name}/{2**current_stage}",
                                             batch_size=batch_size, buffer_size=conf.data.buffer_sizes[current_stage],
                                             **conf.data)
        # train_dataset = conf.general.strategy.experimental_distribute_dataset(train_dataset)
        model = ProgressiveGAN(
            model_kwargs=conf.model,
            optimizer_kwargs=conf.optimizer,
            replica_batch_sizes=conf.data.replica_batch_sizes,
            alpha_init=conf.train.alpha_init,
            alpha_step=conf.model.alpha_step,
            current_stage=current_stage,
            noise_dim=conf.model.noise_dim,
            wgan_lambda=conf.train.wgan_lambda,
            wgan_target=conf.train.wgan_target,
            drift_epsilon=conf.train.drift_epsilon,
            use_gradient_penalty=conf.train.use_gradient_penalty,
            use_epsilon_penalty=conf.train.use_epsilon_penalty,
            name='ProgressiveGAN'
        )
        model.compile()

    transfer_weights(source_model=model.generator, target_model=final_gen, beta=0.0)  # force same initialization

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

    for epoch in epochs:
        # make an epoch step
        epoch_start_time = time.time()
        ret = model.fit(train_dataset, epochs=1, steps_per_epoch=steps_per_epoch, verbose=0)
        ret = ret.history
        for key, value in ret.items():
            ret[key] = value[0]
        epoch_duration = time.time() - epoch_start_time

        transfer_ema_weights(source_model=model.generator, source_ema=model.ema, target_model=final_gen)
        total_image_count += int(conf.data.num_examples)

        # chief does TensorBoard logging
        if conf.general.is_chief and conf.general.logging and conf.general.log_freq and (epoch + 1) % conf.general.log_freq == 0:
            with conf.general.summary.as_default():
                num_replicas = conf.general.strategy.num_replicas_in_sync
                num_replicas_per_node = num_replicas / conf.general.nnodes
                batches_per_second = tf.cast(steps_per_epoch, tf.float32) / epoch_duration
                images_per_second = conf.data.num_examples/epoch_duration
                seconds_per_kimages = 1000*epoch_duration/conf.data.num_examples
                disc_gradient_penalty = ret['gradient_loss'] * (conf.train.wgan_target ** 2) / conf.train.wgan_lambda
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
                tf.summary.scalar("losses/epoch/generator", ret['gen_loss'], epoch)
                tf.summary.scalar("losses/epoch/discriminator", ret['disc_loss'], epoch)
                tf.summary.scalar("losses/epoch/wasserstein_disc", ret['wgan_disc_loss'], epoch)
                tf.summary.scalar("losses/epoch/gradient_penalty_disc", ret['gradient_loss'], epoch)
                tf.summary.scalar("losses/epoch/epsilon_penalty_disc", ret['epsilon_loss'], epoch)
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
                tf.summary.scalar("optimizers/epoch/discriminator_learning_rate", model.optimizer_dis.lr, epoch)
                tf.summary.scalar("optimizers/epoch/generator_learning_rate", model.optimizer_gen.lr, epoch)

        # chief saves eval images
        if conf.general.is_chief and conf.general.evaluate and conf.general.eval_freq and (epoch + 1) % conf.general.eval_freq == 0:
            n = np.minimum(replica_batch_size, len(random_noise))  # else some GPUs might OOM for large resolutions
            save_eval_images(random_noise[:n], model.generator, epoch, conf.general.out_dir, tf.constant(1.0), data_format=conf.model.data_format)
            save_eval_images(random_noise[:n], final_gen, epoch, conf.general.out_dir, tf.constant(1.0), current_stage, data_format=conf.model.data_format)

        # chief saves model checkpoints
        if conf.general.is_chief and conf.general.save and conf.general.checkpoint_freq and (epoch + 1) % conf.general.checkpoint_freq == 0:
            s = 'x'.join([str(x) for x in image_shape])
            gen_file = os.path.join(conf.general.out_dir, f"cp_{model.generator.name}_epoch-{epoch+1:04d}_shape-{s}.h5")
            dis_file = os.path.join(conf.general.out_dir, f"cp_{model.discriminator.name}_epoch-{epoch+1:04d}_shape-{s}.h5")
            fin_file = os.path.join(conf.general.out_dir, f"cp_{final_gen.name}_epoch-{epoch+1:04d}.h5")
            model.generator.save(filepath=gen_file)
            model.discriminator.save(filepath=dis_file)
            final_gen.save(filepath=fin_file)

        # update log files and tqdm status message
        _str_duration = str(timedelta(seconds=epoch_duration))
        status_message = f"duration={_str_duration}, gen_loss={ret['gen_loss']:.3f}, dis_loss={ret['disc_loss']:.3f}"
        logging.info(f"Finished epoch-{epoch+1:04d} with {status_message}")
        epochs.set_postfix_str(status_message)

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
            with conf.general.strategy.scope():
                batch_size = conf.data.replica_batch_sizes[current_stage] * conf.general.strategy.num_replicas_in_sync
                train_dataset = get_dataset_pipeline(name=f"{conf.data.registered_name}/{2 ** current_stage}",
                                                     batch_size=batch_size,
                                                     buffer_size=conf.data.buffer_sizes[current_stage],
                                                     **conf.data)
                # train_dataset = conf.general.strategy.experimental_distribute_dataset(train_dataset)
                model.current_stage = current_stage
                model.compile()

            if conf.general.is_chief and conf.general.logging:
                logging.info(f"Successfully instantiated {model.discriminator.name} and {model.generator.name} for stage={model.current_stage}")
                model.generator.summary(print_fn=logging.info, line_length=150, positions=[.33, .55, .67, 1.])
                model.discriminator.summary(print_fn=logging.info, line_length=150, positions=[.33, .55, .67, 1.])
            if conf.general.is_chief and conf.general.save:
                plot_model(model.generator, os.path.join(conf.general.out_dir, f"net_{model.generator.name}.png"), True, False, dpi=178)
                plot_model(model.discriminator, os.path.join(conf.general.out_dir, f"net_{model.discriminator.name}.png"), True, False, dpi=178)

            # extract image_shape from train_dataset's element_spec (which is depending on distribution strategy)
            if isinstance(train_dataset.element_spec, tf.TensorSpec):
                image_shape = train_dataset.element_spec.shape[1:]
            elif isinstance(train_dataset.element_spec, tf.python.distribute.values.PerReplicaSpec):
                image_shape = train_dataset.element_spec._value_specs[0].shape[1:]
            else:
                raise RuntimeError(f"dataset {train_dataset} returns unknown element_spec of type {type(train_dataset.element_spec)}")

            # update variables, counters/tqdm postfix
            replica_batch_size = conf.data.replica_batch_sizes[current_stage]
            global_batch_size = replica_batch_size * conf.general.strategy.num_replicas_in_sync
            steps_per_epoch = int(conf.data.num_examples // global_batch_size)
            conf.model.alpha = conf.train.alpha_init
            epochs.set_description_str(f"Progressive-GAN(stage={current_stage}, shape={image_shape}")
            logging.info(f"Starting to train Stage {current_stage}")

    # final
    current_time = time.time()
    stage_duration = str(timedelta(seconds=current_time - stage_start_time))
    train_duration = str(timedelta(seconds=current_time - train_start_time))
    logging.info(f"Completed stage={current_stage} in {stage_duration}, total_train_time={train_duration}")
    logging.info(f"Successfully completed training {conf.train.epochs} epochs in {train_duration}")
