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
        batch_size: Optional[int] = None,
        buffer_size: Optional[int] = None,
        process_func: Optional[Callable] = None,
        num_parallel_calls: Optional[int] = None,
        epochs: Optional[int] = None,
        dataset_caching: bool = True,
        dataset_cache_file: Union[str, os.PathLike] = "") -> Tuple[tf.data.Dataset, int]:
    # assert data directory exists
    assert os.path.isdir(data_dir), f"data_dir={data_dir} is not a valid directory"

    # resolve caching file, log configuration for user (incorrect configuration might lead to OOM)
    if dataset_caching:
        if dataset_cache_file:
            if os.path.exists(dataset_cache_file):
                raise FileExistsError(f"--cache-file {dataset_cache_file} already exists")
            logging.info(f"using dataset_cache_file={dataset_cache_file} for dataset caching")
        else:
            msg = f"dataset caching is activated with --cache and --cache-file was specified as \"\". TensorFlow will "\
                  f"attempt to load the entire dataset into memory. In case of OOM specify a temporary cachefile!"
            logging.warning(msg)

    # load dataset from tensorflow_datasets, apply logical chain of transformations
    dataset, info = tfds.load(name=name, split=split, data_dir=data_dir, with_info=True, as_supervised=False)
    if process_func:
        dataset = dataset.map(map_func=process_func, num_parallel_calls=num_parallel_calls)
    if dataset_caching:
        dataset = dataset.cache(filename=dataset_cache_file)
    if buffer_size:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    if batch_size:
        dataset = dataset.batch(batch_size=batch_size)
    if epochs:
        dataset = dataset.repeat(epochs)
    if buffer_size:
        dataset = dataset.prefetch(buffer_size=num_parallel_calls)
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
            dataset_caching=arguments.caching,
            dataset_cache_file=arguments.cachefile
        )
        image_shape = train_dataset.element_spec.shape[1:]
        if arguments.is_chief:
            logging.info(f"{arguments.host}: train_dataset contains {num_examples} images with shape={image_shape}")

        # instantiate optimizers and loss
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
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # get model
        model_gen = celeb_a_generator(input_tensor=None, input_shape=(arguments.noisedim,))
        model_dis = celeb_a_discriminator(input_tensor=None, input_shape=image_shape, noise_stddev=0.0)
        if arguments.is_chief:
            model_gen.summary(print_fn=logging.info, line_length=170, positions=[.33, .55, .67, 1.])
            model_dis.summary(print_fn=logging.info, line_length=170, positions=[.33, .55, .67, 1.])

        # epoch iterable, chief iterates over tqdm for status prints
        epochs = range(arguments.epochs)
        if arguments.is_chief:
            epochs = tqdm(iterable=epochs, desc='Progressive-GAN', unit='epoch')

        # train loop
        for epoch in epochs:
            epoch_start_time = time.time()
            gen_loss, dis_loss, image_count = epoch_step(
                generator=model_gen,
                discriminator=model_dis,
                optimizer_generator=optimizer_gen,
                optimizer_discriminator=optimizer_dis,
                dataset=train_dataset,
                tf_loss=loss,
                noise_dim=arguments.noisedim,
                epoch=epoch,
                steps_per_epoch=int(num_examples // arguments.globalbatchsize) + 1,
                is_chief=arguments.is_chief
            )
            epoch_duration = time.time() - epoch_start_time

            # additional chief tasks during training
            if arguments.is_chief:
                # save eval images
                if arguments.evaluate and arguments.evalfrequency and (epoch + 1) % arguments.evalfrequency == 0:
                    pass
                # save model checkpoints
                if arguments.saving and arguments.checkpointfrequency and (epoch + 1) % arguments.checkpointfrequency == 0:
                    pass
                # update log files and tqdm status message
                status_message = f"sec={epoch_duration:.3f}, gen_loss={gen_loss:.3f}, dis_loss={dis_loss:.3f}"
                logging.info(f"finished epoch-{epoch + 1:04d} with {status_message}")
                epochs.set_postfix_str(status_message)


def epoch_step(
        generator: tf.keras.Model,
        discriminator: tf.keras.Model,
        optimizer_generator: tf.keras.optimizers.Optimizer,
        optimizer_discriminator: tf.keras.optimizers.Optimizer,
        dataset: tf.data.Dataset,
        tf_loss: tf.keras.losses.Loss,
        noise_dim: int,
        epoch: int,
        steps_per_epoch: int,
        is_chief: bool) -> Tuple[float, float, int]:
    # return metrics
    epoch_gen_loss, epoch_dis_loss, image_count = 0.0, 0.0, 0

    # epoch iterable, chief iterates over tqdm for status prints - all other workers over tf.data.Dataset
    if is_chief:
        dataset = tqdm(iterable=dataset, desc=f"epoch-{epoch+1:04d}", unit="batch", total=steps_per_epoch, leave=False)

    for image_batch in dataset:
        batch_size = len(image_batch)
        batch_gen_loss, batch_dis_loss = train_step(
            generator=generator,
            discriminator=discriminator,
            optimizer_generator=optimizer_generator,
            optimizer_discriminator=optimizer_discriminator,
            image_batch=image_batch,
            noise_dim=noise_dim,
            tf_loss=tf_loss,
        )

        # compute moving average of loss metrics
        epoch_gen_loss = (image_count * epoch_gen_loss + batch_size * batch_gen_loss) / (image_count + batch_size)
        epoch_dis_loss = (image_count * epoch_dis_loss + batch_size * batch_dis_loss) / (image_count + batch_size)
        image_count += batch_size

        # additional chief tasks during training
        if is_chief:
            status_message = f"batch_gen_loss={batch_gen_loss:.3f}, batch_dis_loss={batch_dis_loss:.3f}"
            dataset.set_postfix_str(status_message)
            logging.debug(status_message)

    return epoch_gen_loss, epoch_dis_loss, image_count


@tf.function
def discriminator_loss(tf_loss, real_output, fake_output):
    real_loss = tf_loss(tf.ones_like(real_output), real_output)
    fake_loss = tf_loss(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


@tf.function
def generator_loss(tf_loss, fake_output):
    return tf_loss(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(generator, discriminator, optimizer_generator, optimizer_discriminator, image_batch, noise_dim, tf_loss):
    # generate noise for projecting fake images
    batch_size = tf.shape(image_batch)[0]
    noise = tf.random.normal([batch_size, noise_dim])

    # forward pass: inference through both models on tape, compute losses
    with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
        fake_images = generator(noise, training=True)

        real_image_guesses = discriminator(image_batch, training=True)
        fake_image_guesses = discriminator(fake_images, training=True)

        gen_loss = generator_loss(tf_loss, fake_output=fake_image_guesses)
        disc_loss = discriminator_loss(tf_loss, real_output=real_image_guesses, fake_output=fake_image_guesses)

    # collocate gradients from tapes
    gradients_generator = generator_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_discriminator = discriminator_tape.gradient(disc_loss, discriminator.trainable_variables)
    # backward pass: apply gradients via optimizers to update models
    optimizer_generator.apply_gradients(zip(gradients_generator, generator.trainable_variables))
    optimizer_discriminator.apply_gradients(zip(gradients_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss
