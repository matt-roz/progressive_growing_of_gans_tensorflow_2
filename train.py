import os
import time
import logging
from typing import Union, Optional, Tuple, Callable

import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

from model import generator_paper, discriminator_paper
from utils import save_eval_images, transfer_weights


def get_dataset_pipeline(
        name: str,
        split: str,
        data_dir: Union[str, os.PathLike],
        as_supervised: bool = False,
        batch_size: Optional[int] = None,
        buffer_size: Optional[int] = None,
        process_func: Optional[Callable] = None,
        map_parallel_calls: Optional[int] = None,
        interleave_parallel_calls: Optional[int] = None,
        prefetch_parallel_calls: Optional[int] = None,
        epochs: Optional[int] = None,
        dataset_caching: bool = True,
        dataset_cache_file: Union[str, os.PathLike] = "") -> Tuple[tf.data.Dataset, int]:
    # load dataset from tensorflow_datasets, apply logical chain of transformations
    dataset, info = tfds.load(name=name, split=split, data_dir=data_dir, with_info=True, as_supervised=as_supervised)
    if process_func:
        dataset = dataset.map(map_func=process_func, num_parallel_calls=map_parallel_calls)
    if dataset_caching:
        dataset = dataset.cache(filename=dataset_cache_file)
    if buffer_size:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    if batch_size:
        dataset = dataset.batch(batch_size=batch_size)
    if epochs:
        dataset = dataset.repeat(epochs)
    if buffer_size:
        dataset = dataset.prefetch(buffer_size=prefetch_parallel_calls)
    logging.info(f"Successfully loaded dataset={name} with split={split} from data_dir={data_dir}")
    return dataset, info.splits[split].num_examples


@tf.function
def celeb_a_hq_process_func(entry, as_supervised=False):
    image = entry['image'] if not as_supervised else entry[0]
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image


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
            name='adam_generator',
            # clipvalue=0.01
        )
        optimizer_dis = tf.keras.optimizers.Adam(
            learning_rate=arguments.learning_rate * arguments.disc_repeats,
            beta_1=arguments.beta1,
            beta_2=arguments.beta2,
            epsilon=arguments.adam_epsilon,
            name='adam_discriminator',
            # clipvalue=0.01
        )

        # get model
        alpha_step_per_image = (1.0 - arguments.alpha) / (arguments.epochs_per_stage * 30000 / 2)
        final_gen = generator_paper(
            noise_dim=arguments.noise_dim,
            stop_stage=arguments.stop_stage,
            use_bias=arguments.use_bias,
            use_weight_scaling=arguments.use_weight_scaling,
            use_alpha_smoothing=arguments.use_alpha_smoothing,
            return_all_outputs=True,
            name='final_generator'
        )
        model_gen = final_gen
        model_dis = discriminator_paper(
            stop_stage=arguments.stop_stage,
            use_bias=arguments.use_bias,
            use_weight_scaling=arguments.use_weight_scaling,
            use_alpha_smoothing=arguments.use_alpha_smoothing,
            name='final_discriminator'
        )
        model_gen.summary(print_fn=logging.info, line_length=170, positions=[.33, .55, .67, 1.])
        model_dis.summary(print_fn=logging.info, line_length=170, positions=[.33, .55, .67, 1.])

        # random noise for image eval
        random_noise = tf.random.normal(shape=(16, arguments.noise_dim), seed=1000)

    # local tf.function definitions for fast graphmode execution
    @tf.function
    def discriminator_loss(real_output: tf.Tensor, fake_output: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # wasserstein loss
        wasserstein_loss = tf.reduce_mean(fake_output - real_output)

        # epsilon drift penalty
        if arguments.use_epsilon_drift:
            epsilon_loss = tf.reduce_mean(tf.square(real_output)) * arguments.wgan_epsilon
        else:
            epsilon_loss = 0.0
        return wasserstein_loss, epsilon_loss

    @tf.function
    def generator_loss(fake_output: tf.Tensor) -> tf.Tensor:
        return -tf.reduce_mean(fake_output)

    def train_step(image_batch: tf.Tensor, local_batch_size: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # generate noise for projecting fake images
        noise = tf.random.normal([local_batch_size, arguments.noise_dim])

        # forward pass: inference through both models on tape to create predictions
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            fake_images = model_gen([noise, arguments.alpha], training=True)
            real_image_guesses = model_dis([image_batch, arguments.alpha], training=True)
            fake_image_guesses = model_dis([fake_images, arguments.alpha], training=True)

            # create mixed images for gradient loss
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
            _gen_loss = generator_loss(fake_image_guesses)
            _disc_ws_loss, _disc_eps_loss = discriminator_loss(real_image_guesses, fake_image_guesses)
            _disc_stacked_loss = tf.stack((_disc_ws_loss, gradient_loss, _disc_eps_loss))
            _disc_loss = tf.reduce_sum(_disc_stacked_loss)

        # collocate gradients from tapes
        gradients_generator = generator_tape.gradient(_gen_loss, model_gen.trainable_variables)
        gradients_discriminator = discriminator_tape.gradient(_disc_loss, model_dis.trainable_variables)
        gen_grad_vars = zip(gradients_generator, model_gen.trainable_variables)
        dis_grad_vars = zip(gradients_discriminator, model_dis.trainable_variables)
        # backward pass: apply gradients via optimizers to update models
        optimizer_gen.apply_gradients(gen_grad_vars)
        optimizer_dis.apply_gradients(dis_grad_vars)
        return _gen_loss, _disc_stacked_loss

    def epoch_step(dataset: tf.data.Dataset, num_epoch: int, num_steps: int) -> Tuple[float, tf.Tensor, float]:
        # return metrics
        _epoch_gen_loss, _epoch_dis_loss, _image_count, _current_step = 0.0, 0.0, 0.0, 0

        # epoch iterable, chief iterates over tqdm for status prints - all other workers over tf.data.Dataset
        dataset = tqdm(iterable=dataset, desc=f"epoch-{num_epoch + 1:04d}", unit="batch", total=num_steps, leave=False)

        for image_batch in dataset:
            batch_size = tf.shape(image_batch)[0]
            batch_gen_loss, batch_dis_loss = train_step(image_batch=image_batch, local_batch_size=batch_size)

            # smooth weights into final generator
            transfer_weights(source_model=model_gen, target_model=final_gen, prefix='', beta=0.999, log_info=False)

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
            batch_status_message = f"batch_gen_loss={batch_gen_loss:.3f}, batch_dis_loss={tf.reduce_sum(batch_dis_loss):.3f}"
            dataset.set_postfix_str(batch_status_message)
            logging.debug(batch_status_message)
            _current_step += 1

        return _epoch_gen_loss, _epoch_dis_loss, _image_count

    # train loop
    current_stage = 2 if arguments.use_stages else arguments.stop_stage
    epochs = tqdm(iterable=range(arguments.epochs), desc='Progressive-GAN', unit='epoch')
    batch_sizes = {0: 512, 1: 512, 2: 512, 3: 384, 4: 384, 5: 256, 6: 178, 7: 128, 8: 64, 9: 32, 10: 16}

    train_dataset, num_examples = get_dataset_pipeline(
        name=f"celeb_a_hq/{2**current_stage}",
        split=arguments.split,
        data_dir=arguments.data_dir,
        batch_size=batch_sizes[current_stage],
        buffer_size=arguments.buffer_size,
        process_func=celeb_a_hq_process_func,
        map_parallel_calls=arguments.map_calls,
        interleave_parallel_calls=arguments.interleave_calls,
        prefetch_parallel_calls=arguments.prefetch_calls,
        dataset_caching=arguments.caching,
        dataset_cache_file=arguments.cache_file
    )
    image_shape = train_dataset.element_spec.shape[1:]
    model_gen = generator_paper(
        noise_dim=arguments.noise_dim,
        stop_stage=current_stage,
        use_bias=arguments.use_bias,
        use_weight_scaling=arguments.use_weight_scaling,
        use_alpha_smoothing=arguments.use_alpha_smoothing,
        name=f"generator_stage_{current_stage}"
    )
    model_dis = discriminator_paper(
        input_shape=image_shape,
        stop_stage=current_stage,
        use_bias=arguments.use_bias,
        use_weight_scaling=arguments.use_weight_scaling,
        use_alpha_smoothing=arguments.use_alpha_smoothing,
        name=f"discriminator_stage_{current_stage}"
    )
    transfer_weights(source_model=model_gen, target_model=final_gen, prefix='', beta=0.0, log_info=True)
    model_gen.summary(print_fn=logging.info, line_length=170, positions=[.33, .55, .67, 1.])
    model_dis.summary(print_fn=logging.info, line_length=170, positions=[.33, .55, .67, 1.])
    epochs.set_description_str(f"Progressive-GAN(stage={current_stage}, shape={image_shape}")

    # metrics
    total_image_count = 0
    steps_per_epoch = int(num_examples // batch_sizes[current_stage]) + 1

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
        status_message = f"sec={epoch_duration:.3f}, gen_loss={gen_loss:.3f}, dis_loss={tf.reduce_sum(dis_loss):.3f}"
        logging.info(f"finished epoch-{epoch+1:04d} with {status_message}")
        epochs.set_postfix_str(status_message)

        # check stage increase
        if (epoch + 1) % arguments.epochs_per_stage == 0 and current_stage < arguments.stop_stage:
            arguments.alpha = 0.0
            current_stage += 1
            train_dataset, num_examples = get_dataset_pipeline(
                name=f"celeb_a_hq/{2**current_stage}",
                split=arguments.split,
                data_dir=arguments.data_dir,
                batch_size=batch_sizes[current_stage],
                buffer_size=arguments.buffer_size,
                process_func=celeb_a_hq_process_func,
                map_parallel_calls=arguments.map_calls,
                interleave_parallel_calls=arguments.interleave_calls,
                prefetch_parallel_calls=arguments.prefetch_calls,
                dataset_caching=arguments.caching,
                dataset_cache_file=arguments.cache_file
            )
            image_shape = train_dataset.element_spec.shape[1:]
            steps_per_epoch = int(num_examples // batch_sizes[current_stage]) + 1
            epochs.set_description_str(f"Progressive-GAN(stage={current_stage}, shape={image_shape}")
            _model_gen = generator_paper(
                noise_dim=arguments.noise_dim,
                stop_stage=current_stage,
                use_bias=arguments.use_bias,
                use_weight_scaling=arguments.use_weight_scaling,
                use_alpha_smoothing=arguments.use_alpha_smoothing,
                name=f"generator_stage_{current_stage}"
            )
            _model_dis = discriminator_paper(
                input_shape=image_shape,
                stop_stage=current_stage,
                use_bias=arguments.use_bias,
                use_weight_scaling=arguments.use_weight_scaling,
                use_alpha_smoothing=arguments.use_alpha_smoothing,
                name=f"discriminator_stage_{current_stage}"
            )
            transfer_weights(source_model=model_gen, target_model=_model_gen, prefix='', log_info=True)
            transfer_weights(source_model=model_dis, target_model=_model_dis, prefix='', log_info=True)
            model_gen = _model_gen
            model_dis = _model_dis
            model_gen.summary(print_fn=logging.info, line_length=170, positions=[.33, .55, .67, 1.])
            model_dis.summary(print_fn=logging.info, line_length=170, positions=[.33, .55, .67, 1.])
            save_eval_images(random_noise, model_gen, epoch, arguments.out_dir, prefix=f'stage-{current_stage}-')
            optimizer_gen = tf.keras.optimizers.Adam(
                learning_rate=arguments.learning_rate,
                beta_1=arguments.beta1,
                beta_2=arguments.beta2,
                epsilon=arguments.adam_epsilon,
                name='adam_generator',
                # clipvalue=0.01
            )
            optimizer_dis = tf.keras.optimizers.Adam(
                learning_rate=arguments.learning_rate * arguments.disc_repeats,
                beta_1=arguments.beta1,
                beta_2=arguments.beta2,
                epsilon=arguments.adam_epsilon,
                name='adam_discriminator',
                # clipvalue=0.01
            )
