import os
import warnings
from typing import Union, Tuple

import tensorflow as tf
import numpy as np
from PIL import Image


# ----------------------------------------------------------------------------------------------------------------------
# utils for general purpose operating system handling
# ----------------------------------------------------------------------------------------------------------------------

def get_environment_variable(identifier: str) -> str:
    """Returns the environment variable for identifier.

    Args:
        identifier: string depicting to environment variable

    Returns:
        string stored in environment variable for identifier

    Raises:
        TypeError: if identifier is not of type string
        ValueError: if identifier is empty string
        EnvironmentError: if environment variable is not set
    """
    if not isinstance(identifier, str):
        raise TypeError(f"identifier must be of type str but found type(identifier)={type(identifier)} instead.")
    if identifier == "":
        raise ValueError(f"identifier must be non-empty string, yet received empty string.")
    env_variable = os.environ.get(identifier)
    if env_variable is None:
        raise EnvironmentError(f"environment variable ${identifier} is not set")
    return env_variable


def create_directory(directory: Union[str, bytes, os.PathLike], *args, **kwargs) -> None:
    """Creates a directory.

    Args:
        directory: directory to be created
        args: arguments passed down to os.makedirs
        kwargs: keyword arguments passed down to os.makedirs

    Raises:
        TypeError: if directory is not of appropriate input type
        FileExistsError: if directory already exists as a non-directory
    """
    if not isinstance(directory, (str, bytes, os.PathLike)):
        raise TypeError(f"directory must be str, bytes or os.PathLike, but found {type(directory)} instead")
    if os.path.exists(directory):
        if os.path.isfile(directory) or os.path.islink(directory):
            raise FileExistsError(f"{directory} already exists as a file/link and can't be created")
    else:
        os.makedirs(directory, *args, **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
# utils for general purpose images/tensorflow functions
# ----------------------------------------------------------------------------------------------------------------------

def save_eval_images(
        static_noise: tf.Tensor,
        generator: tf.keras.Model,
        epoch: int,
        output_dir: Union[str, bytes, os.PathLike],
        alpha: tf.Tensor,
        stage: int = 0,
        data_format: str = "channels_last") -> None:
    """Inferences twice through generator and stores a png-image consisting of two rows. First row consists of images 
    generated via static_noise; second row consists of images rolled randomly. Images are stored in output_dir.

    Args:
        static_noise: a 2D-Tensor of shape=(num_images, noise_dim) depicting the noise to inference on
        generator: generator model to synthesize images with
        epoch: current epoch number generator was trained with
        output_dir: output location for images
        alpha: smoothing value of all alpha-smoothed layers (image compositions)
        stage: if generator has multiple outputs, stage depicts which output to use
        data_format: data_format that generator returns

    Raises:
        ValueError: if data_format is not of appropriate value
        IndexError: if generator does not have output for depicted stage
    """
    if data_format not in ['NCHW', 'NHWC', 'channels_first', 'channels_last']:
        raise ValueError(f'undefined data_format={data_format}')
    noise_shape = tf.shape(static_noise)
    channel_axis = -1 if data_format == 'NHWC' or data_format == 'channels_last' else 1

    # inference on generator to get images
    if len(generator.outputs) == 1:  # generator only has output for current stage
        fixed_predictions = generator([static_noise, alpha], training=False).numpy()
        rand_predictions = generator([tf.random.normal(shape=noise_shape), alpha], training=False).numpy()
    else:  # generator has output for all stages
        if not 2 <= stage <= 10:
            raise IndexError(f"generator has output for stages 2 to {2+len(generator.outputs)}: "
                             f"attempted to access output for stage {stage}")
        fixed_predictions = generator([static_noise, alpha], training=False)[stage - 2].numpy()
        rand_predictions = generator([tf.random.normal(shape=noise_shape), alpha], training=False)[stage - 2].numpy()

    # from tf.float32 [-1, 1] to np.uint8 [0, 255]
    fixed_predictions = 255 * ((fixed_predictions + 1.0) / 2.0)
    rand_predictions = 255 * ((rand_predictions + 1.0) / 2.0)
    fixed_predictions.astype(dtype=np.uint8, copy=False)
    rand_predictions.astype(dtype=np.uint8, copy=False)

    # get output dims, swap axes in case of channels_first
    if channel_axis == -1:
        num_images, height, width, channels = fixed_predictions.shape
    else:
        num_images, channels, height, width = fixed_predictions.shape
        fixed_predictions = np.einsum('nchw->nhwc', fixed_predictions)
        rand_predictions = np.einsum('nchw->nhwc', rand_predictions)

    # output container for image, first column are fixed_random_seed pictures, second are not seeded-random
    predictions = np.empty(shape=[2 * height, num_images * width,  channels], dtype=np.uint8)
    for index in range(len(fixed_predictions)):
        predictions[:height, index * width:(index + 1) * width, :] = fixed_predictions[index]
        predictions[height:, index * width:(index + 1) * width, :] = rand_predictions[index]
    image = Image.fromarray(predictions)
    name = f"{generator.name}_epoch-{epoch+1:04d}_alpha-{alpha.numpy():.3f}_shape-{width}x{height}x{channels}.png"
    image.save(os.path.join(output_dir, name))

    # clean up
    image.close()
    del fixed_predictions
    del rand_predictions
    del predictions


def transfer_weights(
        source_model: tf.keras.Model,
        target_model: tf.keras.Model,
        is_cloned: bool = False,
        layer_name_prefix: str = '',
        beta: float = 0.0) -> None:
    """Linear beta-interpolation of weights from source_model to target_model.

    Can be used to maintain a shadow exponential moving average of source_model. Only weights of layers with the same
    name in both models and both starting with 'layer_name_prefix' are transferred.

    If target_model and source_model are clones and share the exact same topology a significantly faster implementation
    is used. If is_cloned is False, this function assumes source_model is a topological sub-network of target_model; in
    that case missing layers in either target_model or source_model are silently ignored.

    Args:
        source_model: the source to transfer weights from
        target_model: the target to transfer weights to
        is_cloned: whether or not source and target are exact clones (significantly speeds up computation)
        layer_name_prefix: only layers, which names start with layer_name_prefix, are transferred
        beta: value for linear interpolation; must be within [0.0, 1.0)

    Raises:
        ValueError: if beta exceeds interval [0.0, 1.0)
    """
    if not 0.0 <= beta < 1.0:
        raise ValueError(f'beta must be within [0.0, 1.0) but received beta={beta} instead')

    if is_cloned:  # same exact layer order and topology in both models
        for source_layer, target_layer in zip(source_model.layers, target_model.layers):
            if source_layer.name == target_layer.name and source_layer.name.startswith(layer_name_prefix):
                for source_var, target_var in zip(source_layer.variables, target_layer.variables):
                    delta_value = (1 - beta) * (target_var - source_var)
                    target_var.assign_sub(delta_value)
    else:  # iterate source_model.layers and transfer to target_layer, if target_layer exists
        for source_layer in source_model.layers:
            source_vars = source_layer.variables
            if source_layer.name.startswith(layer_name_prefix) and source_vars:
                try:
                    target_layer = target_model.get_layer(name=source_layer.name)
                except ValueError:
                    continue
                for source_var, target_var in zip(source_vars, target_layer.variables):
                    delta_value = (1 - beta) * (target_var - source_var)
                    target_var.assign_sub(delta_value)


def he_initializer_scale(kernel_shape, gain: float = 2.0):
    """float: he_normal according to the original contribution https://arxiv.org/abs/1502.01852"""
    fan_in = np.prod(kernel_shape[:-1])
    return np.sqrt(gain / fan_in)


# ----------------------------------------------------------------------------------------------------------------------
# deprecated functions (kept for backwards compatibility for trained and serialized h5 models)
# ----------------------------------------------------------------------------------------------------------------------

def he_kernel_initializer(kernel_shape, gain: float = 2.0, use_weight_scaling: bool = True) \
        -> Tuple[float, tf.initializers.Initializer]:
    warnings.warn("he_kernel_initializer is deprecated. Use he_initializer_scale instead.", DeprecationWarning, 2)
    he_scale = he_initializer_scale(kernel_shape=kernel_shape, gain=gain)
    if use_weight_scaling:
        op_scale = he_scale
        kernel_initializer = tf.keras.initializers.RandomNormal()
    else:
        op_scale = 1.0
        kernel_initializer = tf.keras.initializers.RandomNormal(0, he_scale)
    return op_scale, kernel_initializer
