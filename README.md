## Progressive Growing of GANs - TensorFlow 2 Implementation

![Representative image](res/representative_image_512x1792x3.png)

This is a TensorFlow 2 implementation of *Progressive Growing of GANs*. The original implementation was provided by the authors
**Tero Karras** (NVIDIA), **Timo Aila** (NVIDIA), **Samuli Laine** (NVIDIA), **Jaakko Lehtinen** (NVIDIA and Aalto University).
Please cite the original authors and their work (not this repository):

[Paper (arXiv)](http://arxiv.org/abs/1710.10196) <br>
[TensorFlow 1 Implementation (github)](https://github.com/tkarras/progressive_growing_of_gans)

##### Overview

- [x] Configurable: adapt [`config.py`](config.py)
- [x] Readability: written to for long-term maintenance
- [x] Documentation: fully documented in [Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- [x] Dataset: `tf.data.Dataset` pipeline built via [tensorflow_datasets](https://www.tensorflow.org/datasets)
- [x] AutoGraph: graph mode execution via `tf.function` for train_step with dynamic models
- [x] Distribute: supporting multiple distribution strategies
  - [x] Default
  - [x] [MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy)
  - [x] [MultiWorkerMirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy)


![Example Gif](res/inter3.gif) ![Example Gif](res/inter2.gif) ![Example Gif](res/inter1.gif) 
