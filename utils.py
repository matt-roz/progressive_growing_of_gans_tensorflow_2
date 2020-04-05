import os
from typing import Union

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


def save_eval_images(random_noise, generator, epoch , output_dir):
    fixed_predictions = generator(random_noise, training=False).numpy()
    rand_predictions = generator(tf.random.normal(shape=tf.shape(random_noise)), training=False).numpy()
    num_images, width, height, channels = fixed_predictions.shape  # 16, 128, 128, 3

    fixed_predictions = 255 * ((fixed_predictions + 1.0) / 2.0)
    rand_predictions = 255 * ((rand_predictions + 1.0) / 2.0)

    fixed_predictions.astype(dtype=np.uint8, copy=False)
    rand_predictions.astype(dtype=np.uint8, copy=False)

    predictions = np.empty(shape=[2 * height, num_images * width,  channels], dtype=np.uint8)
    for index in range(len(fixed_predictions)):
        predictions[:height, index * width:(index + 1) * width, :] = fixed_predictions[index]
        predictions[height:, index * width:(index + 1) * width, :] = rand_predictions[index]

    image = Image.fromarray(predictions)
    image.save(os.path.join(output_dir, f"epoch-{epoch:04d}_shape-{width}x{height}x{channels}.png"))

    image.close()
    del fixed_predictions
    del rand_predictions
    del predictions
