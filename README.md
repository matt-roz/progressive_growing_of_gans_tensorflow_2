## Progressive Growing of GANs - TensorFlow 2 Implementation

![Representative image](res/representative_image_512x1792x3.png)

This is a TensorFlow 2 implementation of *Progressive Growing of GANs*. The original implementation was provided by the authors
**Tero Karras** (NVIDIA), **Timo Aila** (NVIDIA), **Samuli Laine** (NVIDIA), **Jaakko Lehtinen** (NVIDIA and Aalto University).
Please cite the original authors and their work (**not** this repository):

[Paper (arXiv)](http://arxiv.org/abs/1710.10196) <br>
[TensorFlow 1 Implementation (github)](https://github.com/tkarras/progressive_growing_of_gans)

#### Overview

The repository at hand was written to get myself more comfortable and familiar with TensorFlow 2. It aims to provide a maintainable and (hopefully) well-written implementation of Progressive GANs in TensorFlow 2. It follows the best practices for **distributed computing with custom training loops and dynamic models** according to [TensorFlow's API](https://www.tensorflow.org/api_docs/python/). This repository aims to use the *highest level API* available in TensorFlow 2 for each building block (dataset, model, layer, etc.), for example:

* [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset): a `celeb_a_hq` pipeline built via [tensorflow_datasets](https://www.tensorflow.org/datasets)
* [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model): functional API implementations of models (for shape inference at `model.build()` time)
* [tf.autograph](https://www.tensorflow.org/api_docs/python/tf/autograph): tracing/compiling python functions for faster graph mode execution 
  * using `tf.function` as a function annotation where appropriate (e.g. [`losses.py`](losses.py)) for static functions
  * using `tf.function` as a function call to manually determine re-tracing of python functions at runtime (necessary to execute dynamic models in graph mode)
* [tf.keras.layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers):
  * subclassing [`Layer`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) for custom layers defined in the original implementation ([`PixelNormalization`](https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120), [`StandardDeviationLayer`](https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L127))
  * subclassing [`Wrapper`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Wrapper) to realize the *weight scaling trick* for any `tf.keras.layers.Layer` as proposed in the original paper
* [tf.distribute.Strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy): allowing the same code base to be run executed with different distribution stratgies **without** code repetition (`DefaultStrategy`, [`MirroredStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy), [`MultiWorkerMirroredStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy))

Additionally this repository aims to provide:

* **Documentation**: fully documented in [Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
* **Readability**: written *for the reader* to be easily understandable and allow long-term maintainability
* **Configurability**: adaptable via [`config.py`](config.py)

![Example Gif](res/inter3.gif) ![Example Gif](res/inter2.gif) ![Example Gif](res/inter1.gif) 

#### Confguration
> global configuration

| identifier | dtype | choices | default | meaning |
|---|---|---|---|---|
| save | bool  | [True, False] | True | de-/activates model saving and checkpointing |
| evaluate | bool  | [True, False] | True | de-/activates model  evaluation|
| logging | bool  | [True, False] | True | de-/activates file logging (incl. TensorBoard) |
| out_dir | str, os.PathLike |  | /media/storage/outs/ | directory for output files (images, models) |
| log_dir | str, os.PathLike |  | /media/storage/outs/ | directory for logging (logfile, tensorboard) |
| data_dir | str, os.PathLike |  | ~/tensorflow_datasets | directory to load tensorflow_datasets from |
| save | bool  | [True, False] | True | de-/activates model saving and checkpointing |
| train_eagerly | bool  | [True, False] | False | de-/activates execution of train_step in graph mode |
| XLA | bool | [True, False] | False | de-/activates XLA JIT compilation for train_step |
| strategy | bool | [True, False] | True | de-/activates file logging (incl. TensorBoard) |
| checkpoint_freq | uint |  | 1 | epoch frequency to checkpoint models with (0 = disabled) |
| eval_freq | uint |  | 1 | epoch frequency to evaluate models with (0 = disabled) |
| log_freq | uint |  | 1 | epoch frequency to log with (0 = disabled) |

> model configuration

| identifier | dtype | choices | default | meaning |
|---|---|---|---|---|
| leaky_alpha | float  |  | 0.2 | leakiness of LeakyReLU activations |
| generator_ema | float  |  | 0.999 | de-/activates model saving and checkpointing |
| resolution | uint  | [4, 8, 16, 32, 64, 128, 256, 512, 1024] | 256 | final resolution |
| noise_dim | uint  | | 512 | noise_dim generator projects from |
| epsilon | float  | | 1e-8 | small constant for numerical stability in model layers |
| use_bias | bool  | [True, False] | True | de-/activates usage of biases in all trainable layers |
| use_stages | bool  | [True, False] | True | de-/activates progressive training of model in stages |
| use_fused_scaling | bool  | [True, False] | True | de-/activates up- and downsampling of images via strides=(2, 2) in Conv2D and Conv2DTranspose |
| use_weight_scaling | bool  | [True, False] | True | de-/activates weight scaling trick |
| use_alpha_smoothing | bool  | [True, False] | True | de-/activates smoothing in an image from a previous block after increasing the model to a new stage |
| use_noise_normalization | bool  | [True, False] | True | de-/activates pixel_normalization on noise input at generator start |

> train loop configuration

| identifier | dtype | choices | default | meaning |
|---|---|---|---|---|
| epochs | uint  |  | 432 | number of epochs to train for |
| epochs_per_stage | uint  |  | 54 | number of epochs per stage |
| alpha_init | float  |  | 0.0 |  initial alpha value to smooth in images from previous block |
| use_epsilon_penalty | bool | [True, False] | True | de-/activates epsilon_drift_penalty applied to discriminator loss |
| drift_epsilon | float  |  | 0.001 |  epsilon scalar for epsilon_drift_penalty |
| use_gradient_penalty | bool | [True, False] | True | de-/activates gradient_penalty applied to discriminator loss |
| wgan_lambda | float  |  | 10.0 | wasserstein lambda scalar for gradient_penalty |
| wgan_target | float  |  | 1.0 | wasserstein target scalar for gradient_penalty |

> data pipeline configuration

tbd

> optimizer configuration

tbd

> logging configuration

tbd

#### Roadmap

- [ ] support for NCHW (channel_first) data format
  - [x] make custom layers [`layers.py`](layers.py) data_format aware
  - [ ] configurable via [`config.py`](config.py)
  - [ ] make models  [`model.py`](model.py) data_format aware
- [ ] implement metrics 
  - [ ] 
  - [ ] MS-SIM
  - [ ] FID
  - [ ] R&R

#### Personal Note
Looking for (job/internship) opportunities world wide: matthiasrozanski[at]gmail.com