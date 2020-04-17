import os
import logging
from typing import Union, Optional, Callable, Tuple, Dict, Sequence

import tensorflow as tf
import tensorflow_datasets as tfds


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
    """Builds a tf.data.Dataset pipeline via tensorflow_datasets. Applies a logical chain of transformations based on
    input arguments. Returns the configured tf.data.Dataset instance and its number of entries. tf.data.Dataset is
    implicitly distribution-aware and can be used both in eager as well as graph mode.

    Args:
        name: name of dataset for tensorflow_dataset loader
        split: split to load from dataset with tensorflow_dataset loader
        data_dir: directory to which tensorflow_dataset should download to and load from, if files are pre-existent
        as_supervised: if true, tf.data.Dataset returns a mapping for each entry instead of a Sequence
        batch_size: global batch size used to
        buffer_size: size of buffer to shuffle the dataset in
        process_func: python callable to apply transformations on each data entry within the pipeline
        map_parallel_calls: number of parallel entries to apply 'process_func' to asynchronously
        interleave_parallel_calls: number of parallel threads to access dataset shards/files concurrently
        prefetch_parallel_calls: number of parallel threads to prefetch entries with
        epochs: number of epochs the dataset should loop for
        dataset_caching: whether or not the dataset should be cached
        dataset_cache_file: location of cache_file. If set to empty string "" the entire dataset will be loaded into
            system memory (assert you have enough memory else this setting will run OOM).

    Returns:
        tuple: containing the tf.data.Dataset instance and its number of entries as an integer:
            dataset: tf.data.Dataset instance
            num_examples: number of entries
    """
    # load dataset from tensorflow_datasets, apply logical chain of transformations
    dataset, info = tfds.load(name=name, split=split, data_dir=data_dir, with_info=True, as_supervised=as_supervised)
    if process_func:
        dataset = dataset.map(map_func=process_func, num_parallel_calls=map_parallel_calls)
    if batch_size:
        dataset = dataset.batch(batch_size=batch_size)
    if dataset_caching:
        dataset = dataset.cache(filename=dataset_cache_file)
    if buffer_size:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    if epochs:
        dataset = dataset.repeat(epochs)
    if buffer_size:
        dataset = dataset.prefetch(buffer_size=prefetch_parallel_calls)
    logging.info(f"Successfully loaded dataset={name} with split={split} from data_dir={data_dir}")
    return dataset, info.splits[split].num_examples


@tf.function
def celeb_a_hq_process_func(
        entry: Union[Dict[str, tf.Tensor], Sequence[tf.Tensor]],
        as_supervised: bool = False) -> tf.Tensor:
    """Transforms celeb_a_hq image from tf.uint8 with range [0, 255] to tf.float32 with range [-1, 1].

    Expects entry to be a sequence of tf.Tensors, if as_supervised is True; else a dictionary mapping with its keys
    according to the info provided with the dataset.

    Args:
        entry: a single entry in a tf.data.Dataset pipeline
        as_supervised: whether or not the entry is a sequence (supervised) or a mapping

    Returns:
        A normalized image as a tf.Tensor.
    """
    image = entry['image'] if not as_supervised else entry[0]
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image
