from typing import Optional, Dict, Sequence

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Dense, Reshape


class Generator(tf.keras.Model):
    def __init__(self, start_stage=None, end_stage=None, activation_alpha: float = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # stage variable for dynamic inference
        # self.start_stage = tf.Variable(2, trainable=False, name='start_stage', dtype=tf.int32)
        # self.current_stage = tf.Variable(2, trainable=False, name='current_stage', dtype=tf.int32)
        # self.end_stage = tf.Variable(10, trainable=False, name='end_stage', dtype=tf.int32)

        self.start_stage = None
        self.current_stage = None

        # input feature layer
        self._feature_dense = tf.keras.layers.Dense(units=512 * 4 * 4, use_bias=False, input_shape=(512,))
        self._feature_reshape = tf.keras.layers.Reshape(target_shape=(4, 4, 512), input_shape=(512 * 4 * 4,))

        # upscaling Conv2Ds
        self._conv_8x8 = tf.keras.layers.Conv2DTranspose(512, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, input_shape=(4, 4, 512))
        self._conv_16x16 = tf.keras.layers.Conv2DTranspose(512, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, input_shape=(8, 8, 512))
        self._conv_32x32 = tf.keras.layers.Conv2DTranspose(512, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, input_shape=(16, 16, 512))
        self._conv_64x64 = tf.keras.layers.Conv2DTranspose(256, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, input_shape=(32, 32, 512))
        self._conv_128x128 = tf.keras.layers.Conv2DTranspose(128, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, input_shape=(64, 64, 256))
        self._conv_256x256 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, input_shape=(128, 128, 128))
        self._conv_512x512 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, input_shape=(256, 256, 64))
        self._conv_1024x1024 = tf.keras.layers.Conv2DTranspose(16, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, input_shape=(512, 512, 32))

        # to rgb conv
        self._torgb_4 = None
        self._torgb_5 = None
        self._torgb_6 = None
        self._torgb_7 = None
        self._torgb_8 = None
        self._torgb_9 = None
        self._torgb_10 = None
        self._torgb_11 = None

        # batchnorms
        self._bn_4 = tf.keras.layers.BatchNormalization()
        self._bn_5 = tf.keras.layers.BatchNormalization()
        self._bn_6 = tf.keras.layers.BatchNormalization()
        self._bn_7 = tf.keras.layers.BatchNormalization()
        self._bn_8 = tf.keras.layers.BatchNormalization()
        self._bn_9 = tf.keras.layers.BatchNormalization()
        self._bn_10 = tf.keras.layers.BatchNormalization()
        self._bn_11 = tf.keras.layers.BatchNormalization()

        # leaky relu
        self._relu_4 = tf.keras.layers.LeakyReLU(alpha=activation_alpha)
        self._relu_5 = tf.keras.layers.LeakyReLU(alpha=activation_alpha)
        self._relu_6 = tf.keras.layers.LeakyReLU(alpha=activation_alpha)
        self._relu_7 = tf.keras.layers.LeakyReLU(alpha=activation_alpha)
        self._relu_8 = tf.keras.layers.LeakyReLU(alpha=activation_alpha)
        self._relu_9 = tf.keras.layers.LeakyReLU(alpha=activation_alpha)
        self._relu_10 = tf.keras.layers.LeakyReLU(alpha=activation_alpha)
        self._relu_11 = tf.keras.layers.LeakyReLU(alpha=activation_alpha)

    #def build(self, input_shape, stage):
    #    _input = tf.keras.layers.Input(shape=input_shape)
    #    self._init_graph_network(_input, self.call(_input, stage, full=True))

    def call(self, inputs, stage, full=False, training=None, mask=None):
        x = inputs
        x = self._feature_dense(x, training=training)
        x = self._feature_reshape(x, training=training)

        # create dynamic layers that are not created with build(_input, stage=end_stage)
        if self._torgb_4 is None:
            self._torgb_4 = tf.keras.layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='tanh', use_bias=False, input_shape=(8, 8, 512))
        if self._torgb_5 is None:
            self._torgb_5 = tf.keras.layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='tanh', use_bias=False, input_shape=(16, 16, 512))
        if self._torgb_6 is None:
            self._torgb_6 = tf.keras.layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='tanh', use_bias=False, input_shape=(32, 32, 512))
        if self._torgb_7 is None:
            self._torgb_7 = tf.keras.layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='tanh', use_bias=False, input_shape=(64, 64, 256))
        if self._torgb_8 is None:
            self._torgb_8 = tf.keras.layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='tanh', use_bias=False, input_shape=(128, 128, 128))
        if self._torgb_9 is None:
            self._torgb_9 = tf.keras.layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='tanh', use_bias=False, input_shape=(256, 256, 64))
        if self._torgb_10 is None:
            self._torgb_10 = tf.keras.layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='tanh', use_bias=False, input_shape=(512, 512, 32))
        if self._torgb_11 is None:
            self._torgb_11 = tf.keras.layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='tanh', use_bias=False, input_shape=(1024, 1024, 16))

        if stage > 2:
            x = self._conv_8x8(x, training=training)                # (4, 4, 512) -> (8, 8, 512)
            x = self._bn_4(x, training=training)
            x = self._relu_4(x, training=training)
            if stage == 3 or full:
                out = self._torgb_4(x, training=training)

        if stage > 3:
            x = self._conv_16x16(x, training=training)              # (8, 8, 512) -> (16, 16, 512)
            x = self._bn_5(x, training=training)
            x = self._relu_5(x, training=training)
            if stage == 4 or full:
                out = self._torgb_5(x, training=training)

        if stage > 4:
            x = self._conv_32x32(x, training=training)              # (16, 16, 512) -> (32, 32, 512)
            x = self._bn_6(x, training=training)
            x = self._relu_6(x, training=training)
            if stage == 5 or full:
                out = self._torgb_6(x, training=training)

        if stage > 5:
            x = self._conv_64x64(x, training=training)              # (32, 32, 512) -> (64, 64, 256)
            x = self._bn_7(x, training=training)
            x = self._relu_7(x, training=training)
            if stage == 6 or full:
                out = self._torgb_7(x, training=training)

        if stage > 6:
            x = self._conv_128x128(x, training=training)            # (64, 64, 256) -> (128, 128, 128)
            x = self._bn_8(x, training=training)
            x = self._relu_8(x, training=training)
            if stage == 7 or full:
                out = self._torgb_8(x, training=training)

        if stage > 7:
            x = self._conv_256x256(x, training=training)            # (128, 128, 128) -> (256, 256, 64)
            x = self._bn_9(x, training=training)
            x = self._relu_9(x, training=training)
            if stage == 8 or full:
                out = self._torgb_9(x, training=training)

        if stage > 8:
            x = self._conv_512x512(x, training=training)            # (256, 256, 64) -> (512, 512, 32)
            x = self._bn_10(x, training=training)
            x = self._relu_10(x, training=training)
            if stage == 9 or full:
                out = self._torgb_10(x, training=training)

        if stage > 9:
            x = self._conv_1024x1024(x, training=training)          # (512, 512, 32) -> (1024, 1024, 16)
            x = self._bn_11(x, training=training)
            x = self._relu_11(x, training=training)
            out = self._torgb_11(x, training=training)

        return out


class Discriminator(tf.keras.Model):
    def __init__(self, activation_alpha: float = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # downscaling conv
        self._conv_512x512 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, input_shape=(1024, 1024, 16))
        self._conv_256x256 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, input_shape=(512, 512, 32))
        self._conv_128x128 = tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, input_shape=(256, 256, 64))
        self._conv_64x64 = tf.keras.layers.Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, input_shape=(128, 128, 128))
        self._conv_32x32 = tf.keras.layers.Conv2D(512, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, input_shape=(64, 64, 256))
        self._conv_16x16 = tf.keras.layers.Conv2D(512, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, input_shape=(32, 32, 512))
        self._conv_8x8 = tf.keras.layers.Conv2D(512, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, input_shape=(16, 16, 512))
        self._conv_4x4 = tf.keras.layers.Conv2D(512, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, input_shape=(8, 8, 512))

        # from rgb
        self._fromrgb_3 = None
        self._fromrgb_4 = None
        self._fromrgb_5 = None
        self._fromrgb_6 = None
        self._fromrgb_7 = None
        self._fromrgb_8 = None
        self._fromrgb_9 = None
        self._fromrgb_10 = None

        # batchnorm
        self._bn_3 = tf.keras.layers.BatchNormalization()
        self._bn_4 = tf.keras.layers.BatchNormalization()
        self._bn_5 = tf.keras.layers.BatchNormalization()
        self._bn_6 = tf.keras.layers.BatchNormalization()
        self._bn_7 = tf.keras.layers.BatchNormalization()
        self._bn_8 = tf.keras.layers.BatchNormalization()
        self._bn_9 = tf.keras.layers.BatchNormalization()
        self._bn_10 = tf.keras.layers.BatchNormalization()

        # leaky relu
        self._relu_3 = tf.keras.layers.LeakyReLU(alpha=activation_alpha)
        self._relu_4 = tf.keras.layers.LeakyReLU(alpha=activation_alpha)
        self._relu_5 = tf.keras.layers.LeakyReLU(alpha=activation_alpha)
        self._relu_6 = tf.keras.layers.LeakyReLU(alpha=activation_alpha)
        self._relu_7 = tf.keras.layers.LeakyReLU(alpha=activation_alpha)
        self._relu_8 = tf.keras.layers.LeakyReLU(alpha=activation_alpha)
        self._relu_9 = tf.keras.layers.LeakyReLU(alpha=activation_alpha)
        self._relu_10 = tf.keras.layers.LeakyReLU(alpha=activation_alpha)

        # output layer
        self._flatten = tf.keras.layers.Flatten(input_shape=(4 * 4 * 512,))
        self._output_dense = None

    #def build(self, input_shape, stage):
    #    _input = tf.keras.layers.Input(shape=input_shape)
    #    self._init_graph_network(_input, self.call(_input, stage, full=True))

    def call(self, inputs, stage, full=False, training=None, mask=None):
        x = inputs

        # create dynamic layers that are not created with build(_input, stage=end_stage)
        if self._fromrgb_3 is None:
            self._fromrgb_3 = tf.keras.layers.Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False, input_shape=(8, 8, 3))
        if self._fromrgb_4 is None:
            self._fromrgb_4 = tf.keras.layers.Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False, input_shape=(16, 16, 3))
        if self._fromrgb_5 is None:
            self._fromrgb_5 = tf.keras.layers.Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False, input_shape=(32, 32, 3))
        if self._fromrgb_6 is None:
            self._fromrgb_6 = tf.keras.layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False, input_shape=(64, 64, 3))
        if self._fromrgb_7 is None:
            self._fromrgb_7 = tf.keras.layers.Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False, input_shape=(128, 128, 3))
        if self._fromrgb_8 is None:
            self._fromrgb_8 = tf.keras.layers.Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False, input_shape=(256, 256, 3))
        if self._fromrgb_9 is None:
            self._fromrgb_9 = tf.keras.layers.Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False, input_shape=(512, 512, 3))
        if self._fromrgb_10 is None:
            self._fromrgb_10 = tf.keras.layers.Conv2D(16, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False, input_shape=(1024, 1024, 3))
        if self._output_dense is None:
            self._output_dense = tf.keras.layers.Dense(1, input_shape=(8192,))

        if stage > 9:
            x = self._fromrgb_10(inputs, training=training)          # (1024, 1024, 3) -> (1024, 1024, 16)
            x = self._conv_512x512(x, training=training)        # (1024, 1024, 16) -> (512, 512, 32)
            x = self._bn_10(x, training=training)
            x = self._relu_10(x, training=training)

        if stage > 8:
            if stage == 9 or full:
                x = self._fromrgb_9(inputs, training=training)       # (512, 512, 3) -> (512, 512, 32)
            x = self._conv_256x256(x, training=training)        # (512, 512, 32) -> (256, 256, 64)
            x = self._bn_9(x, training=training)
            x = self._relu_9(x, training=training)

        if stage > 7:
            if stage == 8 or full:
                x = self._fromrgb_8(inputs, training=training)       # (256, 256, 3) -> (256, 256, 64)
            x = self._conv_128x128(x, training=training)        # (256, 256, 64) -> (128, 128, 128)
            x = self._bn_8(x, training=training)
            x = self._relu_8(x, training=training)

        if stage > 6:
            if stage == 7 or full:
                x = self._fromrgb_7(inputs, training=training)       # (128, 128, 3) -> (128, 128, 128)
            x = self._conv_64x64(x, training=training)          # (128, 128, 128) -> (64, 64, 256)
            x = self._bn_7(x, training=training)
            x = self._relu_7(x, training=training)

        if stage > 5:
            if stage == 6 or full:
                x = self._fromrgb_6(inputs, training=training)       # (64, 64, 3) -> (64, 64, 256)
            x = self._conv_32x32(x, training=training)          # (64, 64, 256) -> (32, 32, 512)
            x = self._bn_6(x, training=training)
            x = self._relu_6(x, training=training)

        if stage > 4:
            if stage == 5 or full:
                x = self._fromrgb_5(inputs, training=training)       # (32, 32, 3) -> (32, 32, 512)
            x = self._conv_16x16(x, training=training)          # (32, 32, 512) -> (16, 16, 512)
            x = self._bn_5(x, training=training)
            x = self._relu_5(x, training=training)

        if stage > 3:
            if stage == 4 or full:
                x = self._fromrgb_4(inputs, training=training)       # (16, 16, 3) -> (16, 16, 512)
            x = self._conv_8x8(x, training=training)            # (16, 16, 512) -> (8, 8, 512)
            x = self._bn_4(x, training=training)
            x = self._relu_4(x, training=training)

        if stage > 2:
            if stage == 3 or full:
                x = self._fromrgb_3(inputs, training=training)       # (8, 8, 3) -> (8, 8, 512)
            x = self._conv_4x4(x, training=training)            # (8, 8, 512) -> (4, 4, 512)
            x = self._bn_3(x, training=training)
            x = self._relu_3(x, training=training)

        x = self._flatten(x, training=training)
        x = self._output_dense(x, training=training)
        return x


def celeb_a_generator(
        input_tensor,
        alpha: tf.Variable,
        input_shape: Optional[Sequence] = None,
        noise_dim: int = 512,
        start_stage: int = 2,
        stop_stage: int = 10,
        leaky_alpha: float = 0.2,
        stage_features: Optional[Dict] = None,
        name: str = 'celeb_a_generator',
        *args,
        **kwargs):
    if stage_features is None:
        stage_features = {0: 512, 1: 512, 2: 512, 3: 512, 4: 512, 5: 512, 6: 256, 7: 128, 8: 64, 9: 32, 10: 16}
    input_shape = input_shape or (noise_dim,)

    if input_tensor is None:
        inputs = tf.keras.layers.Input(shape=input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            inputs = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            inputs = input_tensor

    x = inputs

    # project from noise to minimum image
    _target_shape = (2 ** start_stage, 2 ** start_stage,  stage_features[start_stage])
    _units = np.prod(_target_shape)
    x = Dense(units=_units, use_bias=False, input_shape=(noise_dim,), name='block_s/dense_projector')(x)
    x = Reshape(target_shape=_target_shape, input_shape=(_units,), name='block_s/feature_reshape')(x)

    # build blocks
    for stage in range(start_stage + 1, stop_stage + 1):
        x = Conv2DTranspose(
            filters=stage_features[stage],
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            use_bias=False,
            input_shape=_target_shape,
            name=f'block_{stage}/conv2dtranspose')(x)
        x = BatchNormalization(input_shape=_target_shape, name=f'block_{stage}/bn')(x)
        x = LeakyReLU(alpha=leaky_alpha, input_shape=_target_shape, name=f'block_{stage}/activation')(x)
        _target_shape = (2 ** stage, 2 ** stage,  stage_features[stage])

    x = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), activation='tanh', padding='same',
               input_shape=_target_shape, name='toRGB')(x)
    return tf.keras.models.Model(inputs=inputs, outputs=x, name=name)


def celeb_a_discriminator(
        input_tensor,
        alpha: tf.Variable,
        input_shape: Optional[Sequence] = None,
        start_stage: int = 2,
        stop_stage: int = 10,
        leaky_alpha: float = 0.2,
        stage_features: Optional[Dict] = None,
        name: str = 'celeb_a_discriminator',
        *args,
        **kwargs):
    if stage_features is None:
        stage_features = {0: 512, 1: 512, 2: 512, 3: 512, 4: 512, 5: 512, 6: 256, 7: 128, 8: 64, 9: 32, 10: 16}
    input_shape = input_shape or (2 ** stop_stage, 2 ** stop_stage, 3)

    if input_tensor is None:
        inputs = tf.keras.layers.Input(shape=input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            inputs = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            inputs = input_tensor

    x = inputs

    # from RGB
    _target_shape = (2 ** stop_stage, 2 ** stop_stage, 3)
    x = Conv2D(filters=stage_features[stop_stage], kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False,
               input_shape=_target_shape, name='fromRGB')(x)
    _target_shape = (2 ** stop_stage, 2 ** stop_stage, stage_features[stop_stage])

    for stage in reversed(range(start_stage, stop_stage)):
        x = Conv2D(
            filters=stage_features[stage],
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            use_bias=False,
            input_shape=_target_shape,
            name=f'block_{stage}/conv2d')(x)
        x = BatchNormalization(input_shape=_target_shape, name=f'block_{stage}/bn')(x)
        x = LeakyReLU(alpha=leaky_alpha, input_shape=_target_shape, name=f'block_{stage}/activation')(x)
        _target_shape = (2 ** stage, 2 ** stage,  stage_features[stage])

    x = tf.keras.layers.Flatten(name='block_f/flatten')(x)
    x = tf.keras.layers.Dense(units=1, name='block_f/dense')(x)
    return tf.keras.models.Model(inputs=inputs, outputs=x, name=name)
