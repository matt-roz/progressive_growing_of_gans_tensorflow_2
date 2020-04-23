import os
import gc
import time
import logging
from typing import Tuple
from datetime import timedelta

import tensorflow as tf
from tqdm import tqdm

from data import get_dataset_pipeline, celeb_a_hq_process_func
from model import generator_paper, discriminator_paper
from losses import discriminator_loss, generator_loss
from utils import save_eval_images, transfer_weights


def train(arguments):
    # set optimizer settings
    tf.config.optimizer.set_jit(arguments.XLA)

    with arguments.strategy.scope():
        # instantiate optimizers
        optimizer_gen = tf.keras.optimizers.Adam(
            learning_rate=arguments.learning_rate,
            beta_1=arguments.beta1,
            beta_2=arguments.beta2,
            epsilon=arguments.adam_epsilon,
            name='adam_generator')
        optimizer_dis = tf.keras.optimizers.Adam(
            learning_rate=arguments.learning_rate * arguments.disc_repeats,
            beta_1=arguments.beta1,
            beta_2=arguments.beta2,
            epsilon=arguments.adam_epsilon,
            name='adam_discriminator')

        # instantiate target models to train, log summary
        alpha_step_per_image = (1.0 - arguments.alpha) / (arguments.epochs_per_stage * 30000 / 2)
        final_gen = generator_paper(
            noise_dim=arguments.noise_dim,
            stop_stage=arguments.stop_stage,
            use_bias=arguments.use_bias,
            use_weight_scaling=arguments.use_weight_scaling,
            use_fused_scaling=arguments.use_fused_scaling,
            use_alpha_smoothing=arguments.use_alpha_smoothing,
            use_noise_normalization=arguments.use_noise_normalization,
            leaky_alpha=arguments.leaky_alpha,
            return_all_outputs=True,
            name='final_generator')
        model_gen = final_gen
        model_dis = discriminator_paper(
            stop_stage=arguments.stop_stage,
            use_bias=arguments.use_bias,
            use_weight_scaling=arguments.use_weight_scaling,
            use_fused_scaling=arguments.use_fused_scaling,
            use_alpha_smoothing=arguments.use_alpha_smoothing,
            leaky_alpha=arguments.leaky_alpha,
            name='final_discriminator')
        logging.info(f"Successfully instantiated the following final models {model_dis.name} and {model_gen.name}")
        model_gen.summary(print_fn=logging.info, line_length=170, positions=[.33, .55, .67, 1.])
        model_dis.summary(print_fn=logging.info, line_length=170, positions=[.33, .55, .67, 1.])

        # random noise for image eval
        random_noise = tf.random.normal(shape=(16, arguments.noise_dim), seed=1000)

    def train_step(image_batch: tf.Tensor, local_batch_size: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # generate noise for projecting fake images
        noise = tf.random.normal([local_batch_size, arguments.noise_dim])

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            # forward pass: inference through both models on tape to create predictions
            fake_images = model_gen([noise, arguments.alpha], training=True)
            real_image_guesses = model_dis([image_batch, arguments.alpha], training=True)
            fake_image_guesses = model_dis([fake_images, arguments.alpha], training=True)

            # compute gradient penalty: create mixed images for gradient loss
            if arguments.use_gradient_penalty:
                mixing_factors = tf.random.uniform([local_batch_size, 1, 1, 1], 0.0, 1.0)
                mixed_images = image_batch + (fake_images - image_batch) * mixing_factors
                with tf.GradientTape(watch_accessed_variables=False) as mixed_tape:
                    mixed_tape.watch(mixed_images)
                    mixed_output = model_dis([mixed_images, arguments.alpha], training=True)
                gradient_mixed = mixed_tape.gradient(mixed_output, mixed_images)
                gradient_mixed_norm = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(gradient_mixed), axis=[1, 2, 3]))
                gradient_penalty = tf.reduce_mean(tf.square(gradient_mixed_norm - arguments.wgan_target))
                gradient_loss = gradient_penalty * (arguments.wgan_lambda / (arguments.wgan_target ** 2))
            else:
                gradient_loss = 0.0

            # calculate losses
            _gen_loss = generator_loss(fake_output=fake_image_guesses)
            _disc_ws_loss, _disc_eps_loss = discriminator_loss(
                real_output=real_image_guesses,
                fake_output=fake_image_guesses,
                wgan_epsilon=arguments.wgan_epsilon,
                use_epsilon_drift=arguments.use_epsilon_drift)
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
        dataset = tqdm(iterable=dataset, desc=f"epoch-{num_epoch + 1:04d}", unit="batch", total=num_steps, leave=False)

        for image_batch in dataset:
            batch_size = tf.shape(image_batch)[0]
            batch_gen_loss, batch_dis_loss = train_step(image_batch=image_batch, local_batch_size=batch_size)

            # smooth available weights from current_stage model_gen into final generator
            transfer_weights(source_model=model_gen, target_model=final_gen, beta=arguments.generator_ema)

            # compute moving average of loss metrics
            _size = tf.cast(batch_size, tf.float32)
            _epoch_gen_loss = (_image_count * _epoch_gen_loss + _size * batch_gen_loss) / (_image_count + _size)
            _epoch_dis_loss = (_image_count * _epoch_dis_loss + _size * batch_dis_loss) / (_image_count + _size)
            _image_count += _size

            # TensorBoard logging
            if arguments.logging and arguments.log_freq == 'batch':
                _step = num_epoch * num_steps + _current_step
                tf.summary.scalar(name="losses/batch/generator", data=batch_gen_loss, step=_step)
                tf.summary.scalar(name="losses/batch/discriminator", data=tf.reduce_sum(batch_dis_loss), step=_step)
                tf.summary.scalar(name="losses/batch/discriminator_wgan", data=batch_dis_loss[0], step=_step)
                tf.summary.scalar(name="losses/batch/discriminator_gp", data=batch_dis_loss[1], step=_step)
                tf.summary.scalar(name="losses/batch/discriminator_eps", data=batch_dis_loss[2], step=_step)
                tf.summary.scalar(name="model/batch/alpha", data=arguments.alpha, step=_step)

            # increase alpha
            if arguments.use_alpha_smoothing:
                arguments.alpha = tf.minimum(arguments.alpha + alpha_step_per_image * _size, 1.0)

            # additional chief tasks during training
            message = f"batch_gen_loss={batch_gen_loss:.3f}, batch_dis_loss={tf.reduce_sum(batch_dis_loss):.3f}"
            dataset.set_postfix_str(message)
            logging.debug(message)
            _current_step += 1

        return _epoch_gen_loss, _epoch_dis_loss, _image_count

    # train loop
    current_stage = 2 if arguments.use_stages else arguments.stop_stage
    epochs = tqdm(iterable=range(arguments.epochs), desc='Progressive-GAN', unit='epoch')
    batch_sizes = {0: 16, 1: 16, 2: 16, 3: 16, 4: 16, 5: 16, 6: 16, 7: 16, 8: 14, 9: 6, 10: 3}
    buffer_sizes = {0: 5000, 1: 5000, 2: 5000, 3: 5000, 4: 2500, 5: 1250, 6: 500, 7: 400, 8: 300, 9: 250, 10: 200}

    train_dataset, num_examples = get_dataset_pipeline(
        name=f"celeb_a_hq/{2**current_stage}",
        split=arguments.split,
        data_dir=arguments.data_dir,
        batch_size=batch_sizes[current_stage],
        buffer_size=buffer_sizes[current_stage],
        process_func=celeb_a_hq_process_func,
        map_parallel_calls=arguments.map_calls,
        interleave_parallel_calls=arguments.interleave_calls,
        prefetch_parallel_calls=arguments.prefetch_calls,
        dataset_caching=arguments.caching,
        dataset_cache_file=arguments.cache_file)
    image_shape = train_dataset.element_spec.shape[1:]
    model_gen = generator_paper(
        noise_dim=arguments.noise_dim,
        stop_stage=current_stage,
        use_bias=arguments.use_bias,
        use_weight_scaling=arguments.use_weight_scaling,
        use_fused_scaling=arguments.use_fused_scaling,
        use_alpha_smoothing=arguments.use_alpha_smoothing,
        use_noise_normalization=arguments.use_noise_normalization,
        leaky_alpha=arguments.leaky_alpha,
        name=f"generator_stage_{current_stage}")
    model_dis = discriminator_paper(
        input_shape=image_shape,
        stop_stage=current_stage,
        use_bias=arguments.use_bias,
        use_weight_scaling=arguments.use_weight_scaling,
        use_fused_scaling=arguments.use_fused_scaling,
        use_alpha_smoothing=arguments.use_alpha_smoothing,
        leaky_alpha=arguments.leaky_alpha,
        name=f"discriminator_stage_{current_stage}")
    transfer_weights(source_model=model_gen, target_model=final_gen, beta=0.0)  # force same initialization
    logging.info(f"Successfully instantiated {model_dis.name} and {model_gen.name} for stage={current_stage}")
    model_gen.summary(print_fn=logging.info, line_length=170, positions=[.33, .55, .67, 1.])
    model_dis.summary(print_fn=logging.info, line_length=170, positions=[.33, .55, .67, 1.])
    epochs.set_description_str(f"Progressive-GAN(stage={current_stage}, shape={image_shape}")
    logging.info(f"Starting to train Stage {current_stage}")

    # counters, timings, etc.
    total_image_count = 0
    steps_per_epoch = int(num_examples // batch_sizes[current_stage]) + 1
    stage_start_time = time.time()

    for epoch in epochs:
        # make an epoch step
        epoch_start_time = time.time()
        gen_loss, dis_loss, image_count = epoch_step(train_dataset, epoch, steps_per_epoch)
        epoch_duration = time.time() - epoch_start_time
        total_image_count += int(image_count)

        # TensorBoard logging
        if arguments.logging and arguments.log_freq:
            batches_per_second = tf.cast(steps_per_epoch, tf.float32) / epoch_duration
            tf.summary.scalar(name="train_speed/duration", data=epoch_duration, step=epoch)
            tf.summary.scalar(name="train_speed/images_per_second", data=image_count/epoch_duration, step=epoch)
            tf.summary.scalar(name="train_speed/batches_per_second", data=batches_per_second, step=epoch)
            tf.summary.scalar(name="train_speed/total_image_count", data=total_image_count, step=epoch)
            tf.summary.scalar(name="losses/epoch/generator", data=gen_loss, step=epoch)
            tf.summary.scalar(name="losses/epoch/discriminator", data=tf.reduce_sum(dis_loss), step=epoch)
            tf.summary.scalar(name="losses/epoch/discriminator_wgan", data=dis_loss[0], step=epoch)
            tf.summary.scalar(name="losses/epoch/discriminator_gp", data=dis_loss[1], step=epoch)
            tf.summary.scalar(name="losses/epoch/discriminator_eps", data=dis_loss[2], step=epoch)
            tf.summary.scalar(name="model/epoch/alpha", data=arguments.alpha, step=epoch)

        # save eval images
        if arguments.evaluate and arguments.eval_freq and (epoch + 1) % arguments.eval_freq == 0:
            save_eval_images(random_noise, model_gen, epoch, arguments.out_dir, alpha=arguments.alpha)
            save_eval_images(random_noise, final_gen, epoch, arguments.out_dir, stage=current_stage)

        # save model checkpoints
        if arguments.save and arguments.checkpoint_freq and (epoch + 1) % arguments.checkpoint_freq == 0:
            str_shape = 'x'.join([str(x) for x in image_shape])
            gen_file = os.path.join(arguments.out_dir, f"{model_gen.name}-epoch-{epoch+1:04d}-shape-{str_shape}.h5")
            dis_file = os.path.join(arguments.out_dir, f"{model_dis.name}-epoch-{epoch+1:04d}-shape-{str_shape}.h5")
            fin_file = os.path.join(arguments.out_dir, f"{final_gen.name}-epoch-{epoch+1:04d}.h5")
            model_gen.save(filepath=gen_file)
            model_dis.save(filepath=dis_file)
            final_gen.save(filepath=fin_file)

        # update log files and tqdm status message
        _str_duration = str(timedelta(seconds=epoch_duration))
        status_message = f"duration={_str_duration}, gen_loss={gen_loss:.3f}, dis_loss={tf.reduce_sum(dis_loss):.3f}"
        logging.info(f"Finished epoch-{epoch+1:04d} with {status_message}")
        epochs.set_postfix_str(status_message)

        # check stage increase
        if (epoch + 1) % arguments.epochs_per_stage == 0 and current_stage < arguments.stop_stage:
            stage_duration = str(timedelta(seconds=time.time() - stage_start_time))
            logging.info(f"Successfully completed stage={current_stage} in {stage_duration}s after epoch={epoch+1}")
            # increment stage, get dataset pipeline for new images, instantiate next stage models
            current_stage += 1
            train_dataset, num_examples = get_dataset_pipeline(
                name=f"celeb_a_hq/{2**current_stage}",
                split=arguments.split,
                data_dir=arguments.data_dir,
                batch_size=batch_sizes[current_stage],
                buffer_size=buffer_sizes[current_stage],
                process_func=celeb_a_hq_process_func,
                map_parallel_calls=arguments.map_calls,
                interleave_parallel_calls=arguments.interleave_calls,
                prefetch_parallel_calls=arguments.prefetch_calls,
                dataset_caching=arguments.caching,
                dataset_cache_file=arguments.cache_file)
            image_shape = train_dataset.element_spec.shape[1:]
            _model_gen = generator_paper(
                noise_dim=arguments.noise_dim,
                stop_stage=current_stage,
                use_bias=arguments.use_bias,
                use_weight_scaling=arguments.use_weight_scaling,
                use_fused_scaling=arguments.use_fused_scaling,
                use_alpha_smoothing=arguments.use_alpha_smoothing,
                use_noise_normalization=arguments.use_noise_normalization,
                leaky_alpha=arguments.leaky_alpha,
                name=f"generator_stage_{current_stage}")
            _model_dis = discriminator_paper(
                input_shape=image_shape,
                stop_stage=current_stage,
                use_bias=arguments.use_bias,
                use_weight_scaling=arguments.use_weight_scaling,
                use_fused_scaling=arguments.use_fused_scaling,
                use_alpha_smoothing=arguments.use_alpha_smoothing,
                leaky_alpha=arguments.leaky_alpha,
                name=f"discriminator_stage_{current_stage}")
            optimizer_gen = tf.keras.optimizers.Adam(
                learning_rate=arguments.learning_rate,
                beta_1=arguments.beta1,
                beta_2=arguments.beta2,
                epsilon=arguments.adam_epsilon,
                name='adam_generator')
            optimizer_dis = tf.keras.optimizers.Adam(
                learning_rate=arguments.learning_rate * arguments.disc_repeats,
                beta_1=arguments.beta1,
                beta_2=arguments.beta2,
                epsilon=arguments.adam_epsilon,
                name='adam_discriminator')
            # transfer weights from previous stage models to current_stage models
            transfer_weights(source_model=model_gen, target_model=_model_gen)
            transfer_weights(source_model=model_dis, target_model=_model_dis)
            # clear previous stage models, collect with gc
            del model_gen
            del model_dis
            gc.collect()  # note: this only cleans the python runtime not keras/tensorflow backend nor GPU memory
            model_gen = _model_gen
            model_dis = _model_dis
            model_gen.summary(print_fn=logging.info, line_length=170, positions=[.33, .55, .67, 1.])
            model_dis.summary(print_fn=logging.info, line_length=170, positions=[.33, .55, .67, 1.])
            save_eval_images(random_noise, model_gen, epoch, arguments.out_dir, prefix=f'stage-{current_stage}-')
            # update counters/tqdm postfix
            arguments.alpha = 0.0
            steps_per_epoch = int(num_examples // batch_sizes[current_stage]) + 1
            stage_start_time = time.time()
            epochs.set_description_str(f"Progressive-GAN(stage={current_stage}, shape={image_shape}")
            logging.info(f"Starting to train Stage {current_stage}")
