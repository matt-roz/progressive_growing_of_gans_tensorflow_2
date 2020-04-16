from typing import Optional, Dict, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dense, Reshape, UpSampling2D, Flatten, BatchNormalization, \
    Conv2DTranspose

from layers import PixelNormalization, DownSampling2D, StandardDeviationLayer, custom_conv2d, custom_dense


def generator_example(
        input_shape: Optional[Sequence] = None,
        noise_dim: int = 512,
        stop_stage: int = 10,
        use_bias: bool = True,
        use_weight_scaling: bool = True,
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
        use_weight_scaling: bool = True,
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
        image = DownSampling2D(name=f'block_{current_stage}/downsample_to_{2**(current_stage-1)}x{2**(current_stage-1)}')(image)

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
        use_alpha_smoothing: bool = True,
        return_all_outputs: bool = False,
        leaky_alpha: float = 0.2,
        normalize_latents: bool = False,
        num_features: Optional[Dict] = None,
        name: str = 'pgan_celeb_a_hq_generator',
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
        # _x = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), use_bias=use_bias, kernel_initializer='he_normal',
        #            name=f'block_{stage}/toRGB')(value)
        _x = custom_conv2d(x=value, filters=3, kernel_size=1, name=f'block_{stage}/toRGB', strides=(1, 1),
                           use_weight_scaling=use_weight_scaling)
        return _x

    def block(value: tf.Tensor, stage: int):
        # _x = Conv2D(filters=num_features[stage], kernel_size=(3, 3), strides=(1, 1), padding='same',
        #            use_bias=use_bias, kernel_initializer='he_normal', name=f'block_{stage}/conv2d_1')(value)
        _x = custom_conv2d(x=value, filters=num_features[stage], kernel_size=3, strides=(1, 1),
                           name=f'block_{stage}/conv2d_1', he_initializer_slope=0.0,
                           use_weight_scaling=use_weight_scaling)
        _x = LeakyReLU(alpha=leaky_alpha, name=f'block_{stage}/activation_1')(_x)
        _x = PixelNormalization(name=f'block_{stage}/pixel_norm_1')(_x)
        # _x = Conv2D(filters=num_features[stage], kernel_size=(3, 3), strides=(1, 1), padding='same',
        #             use_bias=use_bias, kernel_initializer='he_normal', name=f'block_{stage}/conv2d_2')(_x)
        _x = custom_conv2d(x=_x, filters=num_features[stage], kernel_size=3, strides=(1, 1),
                           name=f'block_{stage}/conv2d_2', he_initializer_slope=0.0,
                           use_weight_scaling=use_weight_scaling)
        _x = LeakyReLU(alpha=leaky_alpha, name=f'block_{stage}/activation_2')(_x)
        _x = PixelNormalization(name=f'block_{stage}/pixel_norm_2')(_x)
        return _x

    # noise input
    x = inputs
    if normalize_latents:
        x = PixelNormalization(name='block_s/pixel_norm_noise')(x)

    # project from noise to minimum features, apply block 2 to features
    _target_shape = (4, 4, num_features[2])
    _units = np.prod(_target_shape)
    # features = Dense(units=_units, use_bias=use_bias, kernel_initializer='he_normal', input_shape=input_shape,
    #                 name='block_2/dense_projector')(x)
    features = custom_dense(x=x, units=_units, use_weight_scaling=use_weight_scaling, name='block_2/dense_projector')
    features = Reshape(target_shape=_target_shape, input_shape=(_units,), name='block_2/feature_reshape')(features)
    features = LeakyReLU(alpha=leaky_alpha, name='block_2/activation_1')(features)
    features = PixelNormalization(name='block_2/pixel_norm_1')(features)
    # features = Conv2D(filters=num_features[2], kernel_size=(3, 3), strides=(1, 1), padding='same',
    #                  use_bias=use_bias, kernel_initializer='he_normal', name='block_2/conv2d_1')(features)
    features = custom_conv2d(x=features, filters=num_features[2], kernel_size=3, strides=(1, 1),
                             name=f'block_2/conv2d_1', he_initializer_slope=0.0,
                             use_weight_scaling=use_weight_scaling)
    features = LeakyReLU(alpha=leaky_alpha, name='block_2/activation_2')(features)
    features = PixelNormalization(name='block_2/pixel_norm_2')(features)
    image_out = to_rgb(value=features, stage=2)
    outputs.append(tf.nn.tanh(image_out, name=f'block_2/final_image_activation'))

    # build 3 - till end
    for current_stage in range(3, stop_stage + 1):
        # upscale current features and toRGB image from previous layer (image_out)
        up = UpSampling2D(name=f'block_{current_stage}/upsample_to_{2**current_stage}x{2**current_stage}')
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

    def from_rgb(value: tf.Tensor, stage: int):
        # _x = Conv2D(filters=num_features[stage], kernel_size=(1, 1), strides=(1, 1), use_bias=use_bias,
        #            kernel_initializer='he_normal', name=f'block_{stage}/fromRGB')(value)
        _x = custom_conv2d(x=value, filters=num_features[stage], kernel_size=1, strides=(1, 1),
                           he_initializer_slope=0.0, name=f'block_{stage}/fromRGB',
                           use_weight_scaling=use_weight_scaling)
        _x = LeakyReLU(alpha=leaky_alpha, name=f'block_{stage}/activation_rgb')(_x)
        return _x

    def block(value: tf.Tensor, stage: int):
        # _x = Conv2D(filters=num_features[stage], kernel_size=(3, 3), strides=(1, 1), padding='same',
        #            use_bias=use_bias, kernel_initializer='he_normal', name=f'block_{stage}/conv2d_1')(value)
        _x = custom_conv2d(x=value, filters=num_features[stage], kernel_size=3, strides=(1, 1),
                           he_initializer_slope=0.0, name=f'block_{stage}/conv2d_1',
                           use_weight_scaling=use_weight_scaling)
        _x = LeakyReLU(alpha=leaky_alpha, name=f'block_{stage}/activation_1')(_x)
        # _x = Conv2D(filters=num_features[stage - 1], kernel_size=(3, 3), strides=(1, 1), padding='same',
        #            use_bias=use_bias, kernel_initializer='he_normal', name=f'block_{stage}/conv2d_2')(_x)
        _x = custom_conv2d(x=_x, filters=num_features[stage - 1], kernel_size=3, strides=(1, 1),
                           he_initializer_slope=0.0, name=f'block_{stage}/conv2d_2',
                           use_weight_scaling=use_weight_scaling)
        _x = LeakyReLU(alpha=leaky_alpha, name=f'block_{stage}/activation_2')(_x)
        return _x

    # input block stop_stage
    image = inputs
    features = from_rgb(value=image, stage=stop_stage)

    # build from highest block till block 3
    for current_stage in range(stop_stage, 2, -1):
        # apply block on previous features with new stage
        features = block(value=features, stage=current_stage)

        # downsample image features from current block and image from previous block
        down = DownSampling2D(name=f'block_{current_stage}/downsample_to_{2**(current_stage-1)}x{2**(current_stage-1)}')
        features = down(features)
        image = down(image)

        # alpha smooth features from current block into features from previous block image
        if use_alpha_smoothing and current_stage == stop_stage:
            features_image = from_rgb(value=image, stage=current_stage - 1)
            features = features_image + (features - features_image) * alpha

    # final block 2
    x = StandardDeviationLayer(name=f'block_2/stddev_layer')(features)
    # x = Conv2D(filters=num_features[2], kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=use_bias,
    #           kernel_initializer='he_normal', name=f'block_2/conv2d_1')(x)
    x = custom_conv2d(x=x, filters=num_features[2], kernel_size=3, strides=(1, 1),
                      name=f'block_2/conv2d_1', use_weight_scaling=use_weight_scaling)
    x = LeakyReLU(alpha=leaky_alpha, name=f'block_2/activation_1')(x)
    x = Flatten(name='block_2/flatten')(x)
    # x = Dense(units=512, use_bias=use_bias, kernel_initializer='he_normal', name='block_2/dense_1')(x)
    x = custom_dense(x=x, units=512,  name='block_2/dense_1', use_weight_scaling=use_weight_scaling)
    x = LeakyReLU(alpha=leaky_alpha, name=f'block_2/activation_2')(x)
    # x = Dense(units=1, use_bias=use_bias, kernel_initializer='he_normal', activation='linear', name='block_2/dense_2')(x)
    x = custom_dense(x=x, units=1, name='block_2/dense_2', use_weight_scaling=use_weight_scaling)

    return tf.keras.models.Model(inputs=[inputs, alpha], outputs=x, name=name)
