from typing import Tuple

import tensorflow as tf


@tf.function
def discriminator_loss(
        real_output: tf.Tensor,
        fake_output: tf.Tensor,
        wgan_epsilon: float = 0.001,
        use_epsilon_drift: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
    """Computes the wasserstein loss and epsilon drift loss for a discriminator. If use_epsilon_drift is deactivated,
    epsilon_loss is set to 0.0.

    Args:
        real_output: discriminator predictions on real data
        fake_output: discriminator predictions on fake data (from the generator)
        wgan_epsilon: scaling factor for epsilon drift
        use_epsilon_drift: flag depicting whether or not epsilon drift should be used

    Returns:
        tuple: containing both losses
            wasserstein_loss: tf.Tensor with shape=(), dtype=tf.float32 depicting the wasserstein_loss
            epsilon_loss: tf.Tensor with shape=(), dtype=tf.float32 depicting the epsilon_loss
    """
    # wasserstein loss
    wasserstein_loss = tf.reduce_mean(fake_output - real_output)

    # epsilon drift penalty
    if use_epsilon_drift:
        epsilon_loss = tf.reduce_mean(tf.square(real_output)) * wgan_epsilon
    else:
        epsilon_loss = tf.constant(valoe=0.0, dtype=tf.float32)
    return wasserstein_loss, epsilon_loss


@tf.function
def generator_loss(fake_output: tf.Tensor) -> tf.Tensor:
    """Computes the wasserstein loss for a generator.

    Args:
        fake_output: discriminator predictions on fake data (from the generator)

    Returns:
        tf.Tensor: shape=(), dtype=tf.float32 depicting the wasserstein_loss
    """
    return -tf.reduce_mean(fake_output)

