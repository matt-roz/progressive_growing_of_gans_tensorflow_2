import os
import time
import logging
from typing import Union, Optional, Tuple, Callable

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
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
            learning_rate=arguments.learningrate,
            beta_1=arguments.beta1,
            beta_2=arguments.beta2,
            epsilon=arguments.epsilon,
            name='adam_generator',
            clipvalue=0.01
        )
        optimizer_dis = tf.keras.optimizers.Adam(
            learning_rate=arguments.learningrate * arguments.discrepeats,
            beta_1=arguments.beta1,
            beta_2=arguments.beta2,
            epsilon=arguments.epsilon,
            name='adam_discriminator',
            clipvalue=0.01
        )
        tf_loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # get model
        alpha_step_per_image = 1.0 / (arguments.epochsperstage * arguments.numexamples / 2)
        model_gen = generator_paper(
            stop_stage=arguments.stopstage,
            use_bias=arguments.usebias,
            use_weight_scaling=arguments.useweightscaling,
            use_alpha_smoothing=arguments.usealphasmoothing
        )
        model_dis = discriminator_paper(
            stop_stage=arguments.stopstage,
            use_bias=arguments.usebias,
            use_weight_scaling=arguments.useweightscaling,
            use_alpha_smoothing=arguments.usealphasmoothing
        )
        model_gen.summary(print_fn=logging.info, line_length=170, positions=[.33, .55, .67, 1.])
        model_dis.summary(print_fn=logging.info, line_length=170, positions=[.33, .55, .67, 1.])

        # random noise for image eval
        random_noise = tf.random.normal(shape=(16, arguments.noisedim), seed=1000)

    # local tf.function definitions for fast graphmode execution
    @tf.function
    def discriminator_loss(real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
        real_loss = tf_loss_obj(tf.ones_like(real_output), real_output)
        fake_loss = tf_loss_obj(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        # return total_loss
        # real_loss = tf.reduce_mean(real_output)
        # fake_loss = tf.reduce_mean(fake_output)
        # total_loss = tf.reduce_mean(fake_output - real_output)
        return total_loss

    @tf.function
    def generator_loss(fake_output: tf.Tensor) -> tf.Tensor:
        return tf_loss_obj(tf.ones_like(fake_output), fake_output)
        # return -tf.reduce_mean(fake_output)

    def train_step(image_batch: tf.Tensor, local_batch_size: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # generate noise for projecting fake images
        noise = tf.random.normal([local_batch_size, arguments.noisedim])

        # forward pass: inference through both models on tape, compute losses
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            fake_images = model_gen([noise, arguments.alpha], training=True)

            real_image_guesses = model_dis([image_batch, arguments.alpha], training=True)
            fake_image_guesses = model_dis([fake_images, arguments.alpha], training=True)

            _gen_loss = generator_loss(fake_image_guesses)
            _disc_loss = discriminator_loss(real_image_guesses, fake_image_guesses)

        # collocate gradients from tapes
        gradients_generator = generator_tape.gradient(_gen_loss, model_gen.trainable_variables)
        gradients_discriminator = discriminator_tape.gradient(_disc_loss, model_dis.trainable_variables)
        # backward pass: apply gradients via optimizers to update models
        optimizer_gen.apply_gradients(zip(gradients_generator, model_gen.trainable_variables))
        optimizer_dis.apply_gradients(zip(gradients_discriminator, model_dis.trainable_variables))
        return _gen_loss, _disc_loss

    def epoch_step(dataset: tf.data.Dataset, num_epoch: int, num_steps: int) -> Tuple[float, float, float]:
        # return metrics
        _epoch_gen_loss, _epoch_dis_loss, _image_count, _current_step = 0.0, 0.0, 0.0, 0

        # epoch iterable, chief iterates over tqdm for status prints - all other workers over tf.data.Dataset
        dataset = tqdm(iterable=dataset, desc=f"epoch-{num_epoch + 1:04d}", unit="batch", total=num_steps, leave=False)

        for image_batch in dataset:
            batch_size = tf.shape(image_batch)[0]
            batch_gen_loss, batch_dis_loss = train_step(image_batch=image_batch, local_batch_size=batch_size)

            # compute moving average of loss metrics
            _size = tf.cast(batch_size, tf.float32)
            _epoch_gen_loss = (_image_count * _epoch_gen_loss + _size * batch_gen_loss) / (_image_count + _size)
            _epoch_dis_loss = (_image_count * _epoch_dis_loss + _size * batch_dis_loss) / (_image_count + _size)
            _image_count += _size

            # TensorBoard logging
            if arguments.logging and arguments.logfrequency == 'batch':
                _step = num_epoch * num_steps + _current_step
                tf.summary.scalar(name="losses/batch/generator", data=batch_gen_loss, step=_step)
                tf.summary.scalar(name="losses/batch/discriminator", data=batch_dis_loss, step=_step)
                tf.summary.scalar(name="losses/batch/moving_epoch_mean/generator", data=_epoch_gen_loss, step=_step)
                tf.summary.scalar(name="losses/batch/moving_epoch_mean/discriminator", data=_epoch_dis_loss, step=_step)
                tf.summary.scalar(name="model/batch/alpha", data=arguments.alpha, step=_step)

            # increase alpha
            if arguments.usealphasmoothing:
                arguments.alpha += alpha_step_per_image * _size
                arguments.alpha = np.minimum(arguments.alpha, 1.0)

            # additional chief tasks during training
            batch_status_message = f"batch_gen_loss={batch_gen_loss:.3f}, batch_dis_loss={batch_dis_loss:.3f}"
            dataset.set_postfix_str(batch_status_message)
            logging.debug(batch_status_message)
            _current_step += 1

        return _epoch_gen_loss, _epoch_dis_loss, _image_count

    # train loop
    current_stage = arguments.startstage
    epochs = tqdm(iterable=range(arguments.epochs), desc='Progressive-GAN', unit='epoch')
    batch_sizes = {0: 512, 1: 512, 2: 512, 3: 384, 4: 256, 5: 128, 6: 32, 7: 20, 8: 16, 9: 10, 10: 6}
    steps_per_epoch = int(arguments.numexamples // batch_sizes[current_stage]) + 1

    train_dataset, _ = get_dataset_pipeline(
        name=f"celeb_a_hq/{2**current_stage}",
        split=arguments.split,
        data_dir=arguments.datadir,
        batch_size=batch_sizes[current_stage],
        buffer_size=arguments.buffersize,
        process_func=celeb_a_hq_process_func,
        map_parallel_calls=arguments.mapcalls,
        interleave_parallel_calls=arguments.interleavecalls,
        prefetch_parallel_calls=arguments.prefetchcalls,
        dataset_caching=arguments.caching,
        dataset_cache_file=arguments.cachefile
    )
    image_shape = train_dataset.element_spec.shape[1:]
    model_gen = generator_paper(
        noise_dim=arguments.noisedim,
        stop_stage=current_stage,
        use_bias=arguments.usebias,
        use_weight_scaling=arguments.useweightscaling,
        use_alpha_smoothing=arguments.usealphasmoothing,
        name=f"pgan_celeb_a_hq_generator_{current_stage}"
    )
    model_dis = discriminator_paper(
        input_shape=image_shape,
        stop_stage=current_stage,
        use_bias=arguments.usebias,
        use_weight_scaling=arguments.useweightscaling,
        use_alpha_smoothing=arguments.usealphasmoothing,
        name=f"pgan_celeb_a_hq_discriminator_{current_stage}"
    )
    model_gen.summary(print_fn=logging.info, line_length=170, positions=[.33, .55, .67, 1.])
    model_dis.summary(print_fn=logging.info, line_length=170, positions=[.33, .55, .67, 1.])
    epochs.set_description_str(f"Progressive-GAN(stage={current_stage}, shape={image_shape}")

    for epoch in epochs:
        epoch_start_time = time.time()
        gen_loss, dis_loss, image_count = epoch_step(train_dataset, epoch, steps_per_epoch)
        epoch_duration = time.time() - epoch_start_time

        # TensorBoard logging
        if arguments.logging and arguments.logfrequency:
            batches_per_second = tf.cast(steps_per_epoch, tf.float32) / epoch_duration
            tf.summary.scalar(name="train_speed/duration", data=epoch_duration, step=epoch)
            tf.summary.scalar(name="train_speed/images_per_second", data=image_count/epoch_duration, step=epoch)
            tf.summary.scalar(name="train_speed/batches_per_second", data=batches_per_second, step=epoch)
            tf.summary.scalar(name="losses/epoch/generator", data=gen_loss, step=epoch)
            tf.summary.scalar(name="losses/epoch/discriminator", data=dis_loss, step=epoch)
            tf.summary.scalar(name="model/epoch/alpha", data=arguments.alpha, step=epoch)

        # save eval images
        if arguments.evaluate and arguments.evalfrequency and (epoch + 1) % arguments.evalfrequency == 0:
            save_eval_images(random_noise, model_gen, epoch, arguments.outdir, alpha=arguments.alpha)

        # save model checkpoints
        if arguments.saving and arguments.checkpointfrequency and (epoch + 1) % arguments.checkpointfrequency == 0:
            str_shape = 'x'.join([str(x) for x in image_shape])
            gen_file = os.path.join(arguments.outdir, f"{model_gen.name}-epoch-{epoch+1:04d}-shape-{str_shape}.h5")
            dis_file = os.path.join(arguments.outdir, f"{model_dis.name}-epoch-{epoch+1:04d}-shape-{str_shape}.h5")
            model_gen.save(filepath=gen_file)
            model_dis.save(filepath=dis_file)

        # update log files and tqdm status message
        status_message = f"sec={epoch_duration:.3f}, gen_loss={gen_loss:.3f}, dis_loss={dis_loss:.3f}"
        logging.info(f"finished epoch-{epoch+1:04d} with {status_message}")
        epochs.set_postfix_str(status_message)

        # check stage increase
        if (epoch + 1) % arguments.epochsperstage == 0 and current_stage < arguments.stopstage:
            arguments.alpha = 0.0
            current_stage += 1
            train_dataset, _ = get_dataset_pipeline(
                name=f"celeb_a_hq/{2**current_stage}",
                split=arguments.split,
                data_dir=arguments.datadir,
                batch_size=batch_sizes[current_stage],
                buffer_size=arguments.buffersize,
                process_func=celeb_a_hq_process_func,
                map_parallel_calls=arguments.mapcalls,
                interleave_parallel_calls=arguments.interleavecalls,
                prefetch_parallel_calls=arguments.prefetchcalls,
                dataset_caching=arguments.caching,
                dataset_cache_file=arguments.cachefile
            )
            image_shape = train_dataset.element_spec.shape[1:]
            steps_per_epoch = int(arguments.numexamples // batch_sizes[current_stage]) + 1
            epochs.set_description_str(f"Progressive-GAN(stage={current_stage}, shape={image_shape}")
            _model_gen = generator_paper(
                noise_dim=arguments.noisedim,
                stop_stage=current_stage,
                use_bias=arguments.usebias,
                use_weight_scaling=arguments.useweightscaling,
                use_alpha_smoothing=arguments.usealphasmoothing,
                name=f"pgan_celeb_a_hq_generator_{current_stage}"
            )
            _model_dis = discriminator_paper(
                input_shape=image_shape,
                stop_stage=current_stage,
                use_bias=arguments.usebias,
                use_weight_scaling=arguments.useweightscaling,
                use_alpha_smoothing=arguments.usealphasmoothing,
                name=f"pgan_celeb_a_hq_discriminator_{current_stage}"
            )
            transfer_weights(source_model=model_gen, target_model=_model_gen, prefix='')
            transfer_weights(source_model=model_dis, target_model=_model_dis, prefix='')
            model_gen = _model_gen
            model_dis = _model_dis
            model_gen.summary(print_fn=logging.info, line_length=170, positions=[.33, .55, .67, 1.])
            model_dis.summary(print_fn=logging.info, line_length=170, positions=[.33, .55, .67, 1.])
            save_eval_images(random_noise, model_gen, epoch, arguments.outdir, prefix=f'stage-{current_stage}-')
            optimizer_gen = tf.keras.optimizers.Adam(
                learning_rate=arguments.learningrate,
                beta_1=arguments.beta1,
                beta_2=arguments.beta2,
                epsilon=arguments.epsilon,
                name='adam_generator',
                clipvalue=0.01
            )
            optimizer_dis = tf.keras.optimizers.Adam(
                learning_rate=arguments.learningrate * arguments.discrepeats,
                beta_1=arguments.beta1,
                beta_2=arguments.beta2,
                epsilon=arguments.epsilon,
                name='adam_discriminator',
                clipvalue=0.01
            )
