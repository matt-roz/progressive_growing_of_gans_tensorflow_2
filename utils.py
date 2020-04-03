import os
from typing import Union


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
