import os
import time
import logging
from typing import Union, Optional, Tuple, Callable, Dict

import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

from model import celeb_a_discriminator, celeb_a_generator


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
    dataset, info = tfds.load(name=name, split=split, data_dir=data_dir, with_info=True, as_supervised=False)
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
def celeb_a_hq_process_func(entry):
    image = entry['image']
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image


def train(arguments):
    # set optimizer settings
    tf.config.optimizer.set_jit(arguments.XLA)

    with arguments.strategy.scope():
        # get tensorflow data set
        train_dataset, num_examples = get_dataset_pipeline(
            name=f"celeb_a_hq/{arguments.maxresolution}",
            split=arguments.split,
            data_dir=arguments.datadir,
            batch_size=arguments.globalbatchsize,
            buffer_size=arguments.buffersize,
            process_func=celeb_a_hq_process_func,
            map_parallel_calls=arguments.mapcalls,
            interleave_parallel_calls=arguments.interleavecalls,
            prefetch_parallel_calls=arguments.prefetchcalls,
            dataset_caching=arguments.caching,
            dataset_cache_file=arguments.cachefile
        )
        image_shape = train_dataset.element_spec.shape[1:]
        if arguments.is_chief:
            logging.info(f"{arguments.host}: train_dataset contains {num_examples} images with shape={image_shape}")

        # instantiate optimizers
        optimizer_gen = tf.keras.optimizers.Adam(
            learning_rate=arguments.learningrate,
            beta_1=arguments.beta1,
            beta_2=arguments.beta2,
            epsilon=arguments.epsilon,
            name='adam_generator'
        )
        optimizer_dis = tf.keras.optimizers.Adam(
            learning_rate=arguments.learningrate * arguments.discrepeats,
            beta_1=arguments.beta1,
            beta_2=arguments.beta2,
            epsilon=arguments.epsilon,
            name='adam_discriminator'
        )
        tf_loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # get model
        model_gen = celeb_a_generator(input_tensor=None, input_shape=(arguments.noisedim,))
        model_dis = celeb_a_discriminator(input_tensor=None, input_shape=image_shape, noise_stddev=0.0)
        if arguments.is_chief:
            model_gen.summary(print_fn=logging.info, line_length=170, positions=[.33, .55, .67, 1.])
            model_dis.summary(print_fn=logging.info, line_length=170, positions=[.33, .55, .67, 1.])

    # local tf.function definitions for fast graphmode execution
    @tf.function
    def discriminator_loss(real_output, fake_output):
        real_loss = tf_loss_obj(tf.ones_like(real_output), real_output)
        fake_loss = tf_loss_obj(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @tf.function
    def generator_loss(fake_output):
        return tf_loss_obj(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(image_batch, local_batch_size):
        # generate noise for projecting fake images
        noise = tf.random.normal([local_batch_size, arguments.noisedim])

        # forward pass: inference through both models on tape, compute losses
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            fake_images = model_gen(noise, training=True)

            real_image_guesses = model_dis(image_batch, training=True)
            fake_image_guesses = model_dis(fake_images, training=True)

            _gen_loss = generator_loss(fake_image_guesses)
            _disc_loss = discriminator_loss(real_image_guesses, fake_image_guesses)

        # collocate gradients from tapes
        gradients_generator = generator_tape.gradient(_gen_loss, model_gen.trainable_variables)
        gradients_discriminator = discriminator_tape.gradient(_disc_loss, model_dis.trainable_variables)
        # backward pass: apply gradients via optimizers to update models
        optimizer_gen.apply_gradients(zip(gradients_generator, model_gen.trainable_variables))
        optimizer_dis.apply_gradients(zip(gradients_discriminator, model_dis.trainable_variables))
        return _gen_loss, _disc_loss

    def epoch_step(dataset, num_epoch, num_steps):
        # return metrics
        _epoch_gen_loss, _epoch_dis_loss, _image_count = 0.0, 0.0, 0.0

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

            # additional chief tasks during training
            batch_status_message = f"batch_gen_loss={batch_gen_loss:.3f}, batch_dis_loss={batch_dis_loss:.3f}"
            dataset.set_postfix_str(batch_status_message)
            logging.debug(batch_status_message)

        return _epoch_gen_loss, _epoch_dis_loss, tf.cast(_image_count, tf.int32)

    def train_loop():
        epochs = tqdm(iterable=range(arguments.epochs), desc='Progressive-GAN', unit='epoch')
        steps_per_epoch = int(num_examples // arguments.globalbatchsize) + 1

        for epoch in epochs:
            epoch_start_time = time.time()
            gen_loss, dis_loss, image_count = epoch_step(train_dataset, epoch, steps_per_epoch)
            epoch_duration = time.time() - epoch_start_time

            # save eval images
            if arguments.evaluate and arguments.evalfrequency and (epoch + 1) % arguments.evalfrequency == 0:
                pass

            # save model checkpoints
            if arguments.saving and arguments.checkpointfrequency and (epoch + 1) % arguments.checkpointfrequency == 0:
                str_image_shape = 'x'.join([str(x) for x in image_shape])
                model_gen.save(filepath=f"generator-{epoch + 1:04d}-shape-{str_image_shape}.hdf5")
                model_dis.save(filepath=f"discriminator-{epoch + 1:04d}-shape-{str_image_shape}.hdf5")

            # update log files and tqdm status message
            status_message = f"sec={epoch_duration:.3f}, gen_loss={gen_loss:.3f}, dis_loss={dis_loss:.3f}"
            logging.info(f"finished epoch-{epoch + 1:04d} with {status_message}")
            epochs.set_postfix_str(status_message)

    # train loop
    train_loop()
