from typing import Optional, Dict, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dense, Reshape, UpSampling2D, Flatten, BatchNormalization, \
    Conv2DTranspose, AveragePooling2D

from layers import PixelNormalization, StandardDeviationLayer, CustomConv2D, CustomDense, WeightScalingWrapper


def generator_example(
        input_shape: Optional[Sequence] = None,
        noise_dim: int = 512,
        stop_stage: int = 10,
        use_bias: bool = True,
        use_alpha_smoothing: bool = True,
        return_all_outputs: bool = False,
        leaky_alpha: float = 0.2,
        normalize_latents: bool = False,
        num_features: Optional[Dict] = None,
        name: str = 'generator_example',
        **kwargs) -> tf.keras.Model:
    if num_features is None:
        num_features = {0: 512, 1: 512, 2: 512, 3: 512, 4: 512, 5: 512, 6: 256, 7: 128, 8: 64, 9: 32, 10: 16}
    if input_shape is None:
        input_shape = (noise_dim,)
    inputs = tf.keras.layers.Input(shape=input_shape, name='noise_input', dtype=tf.float32)
    outputs = []
    alpha = tf.keras.layers.Input(shape=tuple(), batch_size=1, name='alpha_input', dtype=tf.float32)

    # define building blocks
    def to_rgb(value: tf.Tensor, stage: int):
        _x = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), use_bias=use_bias, kernel_initializer='he_normal',
                    name=f'block_{stage}/toRGB')(value)
        return _x

    def block(value: tf.Tensor, stage: int):
        _x = Conv2DTranspose(filters=num_features[stage], kernel_size=(3, 3), strides=(2, 2), padding='same',
                             use_bias=use_bias, kernel_initializer='he_normal', name=f'block_{stage}/conv2d_1')(value)
        _x = BatchNormalization(name=f'block_{stage}/batch_norm_1')(_x)
        _x = LeakyReLU(alpha=leaky_alpha, name=f'block_{stage}/activation_1')(_x)
        return _x

    # noise input
    x = inputs
    if normalize_latents:
        x = PixelNormalization(name='block_s/pixel_norm_noise')(x)

    # project from noise to minimum features, apply block 2 to features
    _target_shape = (4, 4, num_features[2])
    _units = np.prod(_target_shape)
    features = Dense(units=_units, use_bias=use_bias, kernel_initializer='he_normal', input_shape=input_shape,
                     name='block_2/dense_projector')(x)
    features = Reshape(target_shape=_target_shape, input_shape=(_units,), name='block_2/feature_reshape')(features)
    features = Conv2D(filters=num_features[2], kernel_size=(3, 3), strides=(1, 1), padding='same',
                      use_bias=use_bias, kernel_initializer='he_normal', name='block_2/conv2d_1')(features)
    features = BatchNormalization(name='block_2/batch_norm_1')(features)
    features = LeakyReLU(alpha=leaky_alpha, name='block_2/activation_2')(features)
    image_out = to_rgb(value=features, stage=2)
    outputs.append(tf.nn.tanh(image_out, name=f'block_2/final_image_activation'))

    # build 3 - till end
    for current_stage in range(3, stop_stage + 1):
        # upscale toRGB image from previous layer (image_out)
        image_out = UpSampling2D(name=f'block_{current_stage}/upsample_to_{2**current_stage}x{2**current_stage}')(image_out)

        # apply block on upsampled features with new stage, transform current features to image
        features = block(value=features, stage=current_stage)
        image = to_rgb(value=features, stage=current_stage)

        # alpha smooth features from current block into features from previous block image
        if use_alpha_smoothing and current_stage == stop_stage:
            image_out = image_out + (image - image_out) * alpha
        else:
            image_out = image

        # append to outputs
        outputs.append(tf.nn.tanh(image_out, name=f'block_{current_stage}/final_image_activation'))

    outputs = outputs[-1] if not return_all_outputs else outputs
    return tf.keras.models.Model(inputs=[inputs, alpha], outputs=outputs, name=name)


def discriminator_example(
        input_shape: Optional[Sequence] = None,
        stop_stage: int = 10,
        leaky_alpha: float = 0.2,
        use_bias: bool = True,
        use_alpha_smoothing: bool = True,
        num_features: Optional[Dict] = None,
        name: str = 'discriminator_example',
        **kwargs) -> tf.keras.Model:
    # default values
    if num_features is None:
        num_features = {0: 512, 1: 512, 2: 512, 3: 512, 4: 512, 5: 512, 6: 256, 7: 128, 8: 64, 9: 32, 10: 16}
    if input_shape is None:
        input_shape = (2 ** stop_stage, 2 ** stop_stage, 3)
    inputs = tf.keras.layers.Input(shape=input_shape, name='image_input', dtype=tf.float32)
    alpha = tf.keras.layers.Input(shape=tuple(), batch_size=1, name='alpha_input', dtype=tf.float32)

    def from_rgb(value: tf.Tensor, stage: int):
        _x = Conv2D(filters=num_features[stage], kernel_size=(1, 1), strides=(1, 1), use_bias=use_bias,
                    kernel_initializer='he_normal', name=f'block_{stage}/fromRGB')(value)
        _x = LeakyReLU(alpha=leaky_alpha, name=f'block_{stage}/activation_rgb')(_x)
        return _x

    def block(value: tf.Tensor, stage: int):
        _x = Conv2D(filters=num_features[stage - 1], kernel_size=(3, 3), strides=(2, 2), padding='same',
                    use_bias=use_bias, kernel_initializer='he_normal', name=f'block_{stage}/conv2d_1')(value)
        _x = BatchNormalization(name=f'block_{stage}/batch_norm_1')(_x)
        _x = LeakyReLU(alpha=leaky_alpha, name=f'block_{stage}/activation_1')(_x)
        return _x

    # input block stop_stage
    image = inputs
    features = from_rgb(value=image, stage=stop_stage)

    # build from highest block till block 3
    for current_stage in range(stop_stage, 2, -1):
        # apply block on previous features with new stage
        features = block(value=features, stage=current_stage)

        # downsample image features from current block and image from previous block
        image = AveragePooling2D(name=f'block_{current_stage}/avgpool_to_{2**(current_stage-1)}x{2**(current_stage-1)}')(image)

        # alpha smooth features from current block into features from previous block image
        if use_alpha_smoothing and current_stage == stop_stage:
            features_image = from_rgb(value=image, stage=current_stage - 1)
            features = features_image + (features - features_image) * alpha

    # final block 2
    x = StandardDeviationLayer(name=f'block_2/stddev_layer')(features)
    x = Conv2D(filters=num_features[2], kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=use_bias,
               kernel_initializer='he_normal', name=f'block_2/conv2d_1')(x)
    x = BatchNormalization(name=f'block_2/batch_norm_1')(x)
    x = LeakyReLU(alpha=leaky_alpha, name=f'block_2/activation_1')(x)
    x = Flatten(name='block_2/flatten')(x)
    x = Dense(units=512, use_bias=use_bias, kernel_initializer='he_normal', name='block_2/dense_1')(x)
    x = BatchNormalization(name=f'block_2/batch_norm_2')(x)
    x = LeakyReLU(alpha=leaky_alpha, name=f'block_2/activation_2')(x)
    x = Dense(units=1, use_bias=use_bias, kernel_initializer='he_normal', activation='linear', name='block_2/dense_2')(x)

    return tf.keras.models.Model(inputs=[inputs, alpha], outputs=x, name=name)


def generator_paper(
        input_shape: Optional[Sequence] = None,
        noise_dim: int = 512,
        stop_stage: int = 10,
        use_bias: bool = True,
        use_weight_scaling: bool = True,
        use_fused_scaling: bool = True,
        use_alpha_smoothing: bool = True,
        return_all_outputs: bool = False,
        leaky_alpha: float = 0.2,
        use_noise_normalization: bool = True,
        num_features: Optional[Dict[int, int]] = None,
        name: str = 'pgan_celeb_a_hq_generator',
        **kwargs) -> tf.keras.Model:
    """Functional API implementation of generator for Progressive-GAN as described in https://arxiv.org/abs/1710.10196.
    original implementation: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L144

    This generator is built and invoked for each consecutive stage (up to 9 times). Trainable variables from a previous
    stage are transferred to a newly created instance.

    Args:
        input_shape: optional input_shape without batch_dimension. Defaults to (noise_dim,)
        noise_dim: dimensionality of noise vector to project generated images from
        stop_stage: the stage of the generator; output will be (2**stop_stage, 2**stop_stage, 3)
        use_bias: whether or not layers should use biases or not
        use_weight_scaling: whether or not the weight scaling trick should be applied
        use_alpha_smoothing: whether or not layer new stages should be linearly interpolated into the current model
        return_all_outputs: whether or not all image outputs (including previous stages) should be connected to the
            output. By default only the current stage image (stop_stage) is returned.
        leaky_alpha: alpha for LeakyReLU configuration
        use_noise_normalization: whether or not the noise vector should be pixel_normalized
        num_features: mapping of stage to features; all Convolutions will output num_features[stage] at stage
        name: name of keras model
        **kwargs: unused

    Returns:
        tf.keras.Model built via functional API depicting the generator at stage 'stop_stage'
    """
    if num_features is None:
        num_features = {0: 512, 1: 512, 2: 512, 3: 512, 4: 512, 5: 512, 6: 256, 7: 128, 8: 64, 9: 32, 10: 16}
    if input_shape is None:
        input_shape = (noise_dim,)
    inputs = tf.keras.layers.Input(shape=input_shape, name='noise_input', dtype=tf.float32)
    alpha = tf.keras.layers.Input(shape=tuple(), batch_size=1, name='alpha_input', dtype=tf.float32)
    outputs = []

    # define building blocks
    def _conv(filters, kernel_size, gain=2.0, **_kwargs):
        _layer = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=use_bias,
                        kernel_initializer='random_normal', **_kwargs)
        if use_weight_scaling:
            _layer = WeightScalingWrapper(layer=_layer, gain=gain)
        return _layer

    def _deconv(filters, kernel_size, gain=2.0, **_kwargs):
        _layer = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=(2, 2), padding='same',
                                 use_bias=use_bias, kernel_initializer='random_normal', **_kwargs)
        if use_weight_scaling:
            _layer = WeightScalingWrapper(layer=_layer, gain=gain)
        return _layer

    def _dense(units, gain=2.0, **_kwargs):
        _layer = Dense(units=units, use_bias=use_bias, kernel_initializer='random_normal', **_kwargs)
        if use_weight_scaling:
            _layer = WeightScalingWrapper(layer=_layer, gain=gain)
        return _layer

    def to_rgb(value: tf.Tensor, stage: int):
        _x = _conv(filters=3, kernel_size=(1, 1), gain=1.0, name=f'block_{stage}/toRGB')(value)
        return _x

    def block(value: tf.Tensor, stage: int):
        if use_fused_scaling:
            _x = _deconv(filters=num_features[stage], kernel_size=(3, 3), gain=2.0, name=f'block_{stage}/conv2d_1')(value)
        else:
            _x = _conv(filters=num_features[stage], kernel_size=(3, 3), gain=2.0, name=f'block_{stage}/conv2d_1')(value)
        _x = LeakyReLU(leaky_alpha, name=f'block_{stage}/activation_1')(_x)
        _x = PixelNormalization(name=f'block_{stage}/pixel_norm_1')(_x)
        _x = _conv(filters=num_features[stage], kernel_size=(3, 3), gain=2.0, name=f'block_{stage}/conv2d_2')(_x)
        _x = LeakyReLU(leaky_alpha, name=f'block_{stage}/activation_2')(_x)
        _x = PixelNormalization(name=f'block_{stage}/pixel_norm_2')(_x)
        return _x

    # noise input
    x = inputs
    if use_noise_normalization:
        x = PixelNormalization(name='block_s/pixel_norm_noise')(x)

    # project from noise to minimum features, apply block 2 to features
    _target_shape = (4, 4, num_features[2])
    _units = int(np.prod(_target_shape))
    features = _dense(units=_units, gain=0.125, name='block_2/dense_projector')(x)
    features = Reshape(_target_shape, name='block_2/feature_reshape')(features)
    features = LeakyReLU(leaky_alpha, name='block_2/activation_1')(features)
    features = PixelNormalization(name='block_2/pixel_norm_1')(features)
    features = _conv(filters=num_features[2], kernel_size=(3, 3), gain=2.0, name=f'block_2/conv2d_1')(features)
    features = LeakyReLU(leaky_alpha, name='block_2/activation_2')(features)
    features = PixelNormalization(name='block_2/pixel_norm_2')(features)
    image_out = to_rgb(value=features, stage=2)
    outputs.append(tf.nn.tanh(image_out, name=f'block_2/final_image_activation'))

    # build block 3 - till end
    for current_stage in range(3, stop_stage + 1):
        # upscale current features and toRGB image from previous layer (image_out)
        up = UpSampling2D(name=f'block_{current_stage}/upsample_to_{2**current_stage}x{2**current_stage}')
        if not use_fused_scaling:
            features = up(features)
        image_out = up(image_out)

        # apply block on upsampled features with new stage, transform current features to image
        features = block(value=features, stage=current_stage)
        image = to_rgb(value=features, stage=current_stage)

        # alpha smooth features from current block into features from previous block image
        if use_alpha_smoothing and current_stage == stop_stage:
            image_out = image_out + (image - image_out) * alpha
        else:
            image_out = image

        # append to outputs
        outputs.append(tf.nn.tanh(image_out, name=f'block_{current_stage}/final_image_activation'))

    outputs = outputs[-1] if not return_all_outputs else outputs
    return tf.keras.models.Model(inputs=[inputs, alpha], outputs=outputs, name=name)


def discriminator_paper(
        input_shape: Optional[Sequence] = None,
        stop_stage: int = 10,
        leaky_alpha: float = 0.2,
        use_bias: bool = True,
        use_weight_scaling: bool = True,
        use_fused_scaling: bool = True,
        use_alpha_smoothing: bool = True,
        num_features: Optional[Dict] = None,
        name: str = 'pgan_celeb_a_hq_discriminator',
        **kwargs) -> tf.keras.Model:
    # default values
    if num_features is None:
        num_features = {0: 512, 1: 512, 2: 512, 3: 512, 4: 512, 5: 512, 6: 256, 7: 128, 8: 64, 9: 32, 10: 16}
    if input_shape is None:
        input_shape = (2 ** stop_stage, 2 ** stop_stage, 3)
    inputs = tf.keras.layers.Input(shape=input_shape, name='image_input', dtype=tf.float32)
    alpha = tf.keras.layers.Input(shape=tuple(), batch_size=1, name='alpha_input', dtype=tf.float32)

    # define building blocks
    def _conv(filters, kernel_size, strides=(1, 1), gain=2.0, **_kwargs):
        _layer = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=use_bias,
                        kernel_initializer='random_normal', **_kwargs)
        if use_weight_scaling:
            _layer = WeightScalingWrapper(layer=_layer, gain=gain)
        return _layer

    def _dense(units, gain=2.0, **_kwargs):
        _layer = Dense(units=units, use_bias=use_bias, kernel_initializer='random_normal', **_kwargs)
        if use_weight_scaling:
            _layer = WeightScalingWrapper(layer=_layer, gain=gain)
        return _layer

    def from_rgb(value: tf.Tensor, stage: int):
        _x = _conv(filters=num_features[stage], kernel_size=(1, 1), gain=2.0, name=f'block_{stage}/fromRGB')(value)
        _x = LeakyReLU(leaky_alpha, name=f'block_{stage}/activation_rgb')(_x)
        return _x

    def block(value: tf.Tensor, stage: int):
        strides = (2, 2) if use_fused_scaling else (1, 1)
        _x = _conv(filters=num_features[stage], kernel_size=(3, 3), strides=strides, gain=2.0,  name=f'block_{stage}/conv2d_1')(value)
        _x = LeakyReLU(leaky_alpha, name=f'block_{stage}/activation_1')(_x)
        _x = _conv(filters=num_features[stage-1], kernel_size=(3, 3), gain=2.0, name=f'block_{stage}/conv2d_2')(_x)
        _x = LeakyReLU(leaky_alpha, name=f'block_{stage}/activation_2')(_x)
        return _x

    # input block stop_stage
    image = inputs
    features = from_rgb(value=image, stage=stop_stage)

    # build from highest block till block 3
    for current_stage in range(stop_stage, 2, -1):
        # apply block on previous features with new stage
        features = block(value=features, stage=current_stage)

        # downsample image features from current block and image from previous block
        down = AveragePooling2D(name=f'block_{current_stage}/avgpool_to_{2**(current_stage-1)}x{2**(current_stage-1)}')
        if not use_fused_scaling:
            features = down(features)
        image = down(image)

        # alpha smooth features from current block into features from previous block image
        if use_alpha_smoothing and current_stage == stop_stage:
            features_image = from_rgb(value=image, stage=current_stage - 1)
            features = features_image + (features - features_image) * alpha

    # final block 2
    x = StandardDeviationLayer(name=f'block_2/stddev_layer')(features)
    x = _conv(filters=num_features[2], kernel_size=(3, 3), gain=2.0, name=f'block_2/conv2d_1')(x)
    x = LeakyReLU(leaky_alpha, name=f'block_2/activation_1')(x)
    _units = x.shape[-1]
    x = Flatten(name='block_2/flatten')(x)
    x = _dense(units=_units, gain=2.0, name='block_2/dense_1')(x)
    x = LeakyReLU(leaky_alpha, name=f'block_2/activation_2')(x)
    x = _dense(units=1, gain=1.0, name='block_2/dense_2')(x)

    return tf.keras.models.Model(inputs=[inputs, alpha], outputs=x, name=name)
