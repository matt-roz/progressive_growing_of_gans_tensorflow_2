import tensorflow as tf


@tf.function
def wasserstein_discriminator_loss(real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
    """Computes the wasserstein loss for a discriminator according to https://arxiv.org/abs/1701.07875.

    Args:
        real_output: 1D-Tensor depicting discriminator predictions on real data
        fake_output: 1D-Tensor depicting discriminator predictions on fake data (from the generator)

    Returns:
        1D-Tensor depicting the per example wasserstein_loss
    """
    return fake_output - real_output


@tf.function
def wasserstein_generator_loss(fake_output: tf.Tensor) -> tf.Tensor:
    """Computes the wasserstein loss for a generator according to https://arxiv.org/abs/1701.07875.

    Args:
        fake_output: 1D-Tensor depicting discriminator predictions on fake data (from the generator)

    Returns:
        1D-Tensor depicting the per example wasserstein_loss
    """
    return -fake_output


def wasserstein_gradient_penalty(discriminator: tf.keras.Model, real_images: tf.Tensor, fake_images: tf.Tensor,
                                 wgan_target: float, wgan_lambda: float, *args) -> tf.Tensor:
    """Compute the wasserstein gradient penalty for a discriminator according to https://arxiv.org/abs/1704.00028.

    This function must be called within the context of an active tf.GradientTape of the discriminator.

    Example:
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            # forward pass
            fake_images = generator([noise, *args], training=True)
            real_guesses = discriminator([image_batch, *args], training=True)
            fake_guesses = discriminator([fake_images, *args], training=True)

            # gradient loss - must be computed within activate discriminator_tape context
            gradient_loss = wasserstein_gradient_penalty(...)

            # other losses
            generator_loss = ...
            discriminator_loss = ...

    Args:
        discriminator: model instance to inference on
        real_images: 4D-Tensor depicting ground truth images from dataset
        fake_images: 4D-Tensor depicting generated images by generator
        wgan_target: target value according to paper
        wgan_lambda: lambda value according to paper
        *args: additional arguments to pass into model.call() for inference

    Returns:
        1D-Tensor depicting the per example gradient penalty
    """
    # create mixed images
    local_batch_size = tf.shape(real_images)[0]
    mixing_factors = tf.random.uniform([local_batch_size, 1, 1, 1], 0.0, 1.0)
    mixed_images = real_images + (fake_images - real_images) * mixing_factors

    # apply forward pass, compute mixed norm and penalty
    with tf.GradientTape(watch_accessed_variables=False) as mixed_tape:
        mixed_tape.watch(mixed_images)
        mixed_output = discriminator([mixed_images, *args], training=True)
    gradient_mixed = mixed_tape.gradient(mixed_output, mixed_images)
    gradient_mixed_norm = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(gradient_mixed), axis=[1, 2, 3]))
    gradient_penalty = tf.square(gradient_mixed_norm - wgan_target)
    return gradient_penalty * (wgan_lambda / (wgan_target ** 2))


@tf.function
def discriminator_epsilon_drift(real_output: tf.Tensor, wgan_epsilon: float) -> tf.Tensor:
    """Compute epsilon drift penalty for discriminator according to https://arxiv.org/abs/1710.10196-

    Args:
        real_output: 1D-Tensor depicting discriminator predictions on real data
        wgan_epsilon: epsilon value to drift predictions with

    Returns:
        1D-Tensor depicting the per example epsilon drift
    """
    return tf.square(real_output) * wgan_epsilon
