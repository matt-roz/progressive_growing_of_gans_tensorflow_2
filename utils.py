import os
from typing import Union
import logging

import tensorflow as tf
import numpy as np
from PIL import Image


def get_environment_variable(identifier: str) -> str:
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


def save_eval_images(random_noise: tf.Tensor, generator: tf.keras.Model, epoch: int, output_dir, prefix: str = "",
                     alpha: float = 0.0, stage: int = 0) -> None:
    assert isinstance(stage, int) and (stage == 0 or stage >= 2)
    # ToDo M.Rozanski: adapt flag for data_format 'NCHW' (right now 'NHWC' only)
    # inference on generator to get images
    _shape = tf.shape(random_noise)
    if not stage:
        fixed_predictions = generator([random_noise, alpha], training=False).numpy()
        rand_predictions = generator([tf.random.normal(shape=_shape), alpha], training=False).numpy()
    else:
        fixed_predictions = generator([random_noise, alpha], training=False)[stage - 2].numpy()
        rand_predictions = generator([tf.random.normal(shape=_shape), alpha], training=False)[stage - 2].numpy()
    num_images, width, height, channels = fixed_predictions.shape

    # from tf.float32 [-1, 1] to np.uint8 [0, 244]
    fixed_predictions = 255 * ((fixed_predictions + 1.0) / 2.0)
    rand_predictions = 255 * ((rand_predictions + 1.0) / 2.0)
    fixed_predictions.astype(dtype=np.uint8, copy=False)
    rand_predictions.astype(dtype=np.uint8, copy=False)

    # output container for image, first column are fixed_random_seed pictures, second are not seeded-random
    predictions = np.empty(shape=[2 * height, num_images * width,  channels], dtype=np.uint8)
    for index in range(len(fixed_predictions)):
        predictions[:height, index * width:(index + 1) * width, :] = fixed_predictions[index]
        predictions[height:, index * width:(index + 1) * width, :] = rand_predictions[index]
    image = Image.fromarray(predictions)

    # file name
    if not stage:
        name = f"{prefix}epoch-{epoch+1:04d}_alpha-{alpha:.3f}_shape-{width}x{height}x{channels}.png"
    else:
        name = f"{prefix}final_gen_epoch-{epoch+1:04d}_alpha-{alpha:.3f}_shape-{width}x{height}x{channels}.png"
    image.save(os.path.join(output_dir, name))

    # clean up
    image.close()
    del fixed_predictions
    del rand_predictions
    del predictions


def transfer_weights(source_model: tf.keras.Model, target_model: tf.keras.Model, prefix: str = 'block',
                     beta: float = 0.0, log_info: bool = False):
    transferred_name_list = []
    missing_target_list = []

    # iterate layers and transfer to target_layer, if existent
    for layer in source_model.layers:
        source_vars = layer.variables
        if layer.name.startswith(prefix) and source_vars:
            try:
                target_layer = target_model.get_layer(name=layer.name)
            except ValueError:
                missing_target_list.append(layer.name)
                continue
            # assign source to target for each var in layer, discount var by beta-momentum if var.trainable
            for source_var, target_var in zip(source_vars, target_layer.variables):
                assert source_var.shape == target_var.shape
                assert source_var.dtype == target_var.dtype
                _beta = beta if source_var.trainable else 0.0
                new_value = source_var + (target_var - source_var) * _beta
                target_var.assign(new_value)
                transferred_name_list.append(f"{source_var.name} -> {target_var.name}")

    # log and clear
    if log_info:
        logging.info(f"Transferred variables with beta={beta} from {source_model.name} to {target_model.name}: "
                     f"{transferred_name_list}. The following layers were not found in {target_model.name}: "
                     f"{missing_target_list}")
    del transferred_name_list
    del missing_target_list
    # ToDo M. Rozanski: implement missing source_layer.name logic ?
