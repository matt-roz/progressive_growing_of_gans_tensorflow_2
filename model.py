import tensorflow as tf


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
        self._feature_dense = tf.keras.layers.Dense(units=512 * 4 * 4, use_bias=False)
        self._feature_reshape = tf.keras.layers.Reshape(target_shape=(4, 4, 512))

        # upscaling Conv2Ds
        self._conv_8x8 = tf.keras.layers.Conv2DTranspose(512, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_16x16 = tf.keras.layers.Conv2DTranspose(512, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_32x32 = tf.keras.layers.Conv2DTranspose(512, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_64x64 = tf.keras.layers.Conv2DTranspose(256, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_128x128 = tf.keras.layers.Conv2DTranspose(128, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_256x256 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_512x512 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_1024x1024 = tf.keras.layers.Conv2DTranspose(16, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)

        # to rgb conv
        self._torgb_4 = tf.keras.layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='tanh', use_bias=False)
        self._torgb_5 = tf.keras.layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='tanh', use_bias=False)
        self._torgb_6 = tf.keras.layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='tanh', use_bias=False)
        self._torgb_7 = tf.keras.layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='tanh', use_bias=False)
        self._torgb_8 = tf.keras.layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='tanh', use_bias=False)
        self._torgb_9 = tf.keras.layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='tanh', use_bias=False)
        self._torgb_10 = tf.keras.layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='tanh', use_bias=False)
        self._torgb_11 = tf.keras.layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='tanh', use_bias=False)

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

    def build(self, input_shape, stage):
        _input = tf.keras.layers.Input(shape=input_shape)
        self._init_graph_network(_input, self.call(_input, stage))

    def call(self, inputs, stage, training=None, mask=None):
        x = inputs
        x = self._feature_dense(x, training=training)
        x = self._feature_reshape(x, training=training)

        if stage > 2:
            x = self._conv_8x8(x, training=training)                # (4, 4, 512) -> (8, 8, 256)
            x = self._bn_4(x, training=training)
            x = self._relu_4(x, training=training)
            if stage == 3:
                x = self._torgb_4(x, training=training)

        if stage > 3:
            x = self._conv_16x16(x, training=training)              # (8, 8, 256) -> (16, 16, 128)
            x = self._bn_5(x, training=training)
            x = self._relu_5(x, training=training)
            if stage == 4:
                x = self._torgb_5(x, training=training)

        if stage > 4:
            x = self._conv_32x32(x, training=training)              # (16, 16, 128) -> (32, 32, 64)
            x = self._bn_6(x, training=training)
            x = self._relu_6(x, training=training)
            if stage == 5:
                x = self._torgb_6(x, training=training)

        if stage > 5:
            x = self._conv_64x64(x, training=training)              # (32, 32, 64) -> (64, 64, 32)
            x = self._bn_7(x, training=training)
            x = self._relu_7(x, training=training)
            if stage == 6:
                x = self._torgb_7(x, training=training)

        if stage > 6:
            x = self._conv_128x128(x, training=training)            # (64, 64, 32) -> (128, 128, 16)
            x = self._bn_8(x, training=training)
            x = self._relu_8(x, training=training)
            if stage == 7:
                x = self._torgb_8(x, training=training)

        if stage > 7:
            x = self._conv_256x256(x, training=training)            # (128, 128, 16) -> (256, 256, 8)
            x = self._bn_9(x, training=training)
            x = self._relu_9(x, training=training)
            if stage == 8:
                x = self._torgb_9(x, training=training)

        if stage > 8:
            x = self._conv_512x512(x, training=training)            # (256, 256, 8) -> (512, 512, 6)
            x = self._bn_10(x, training=training)
            x = self._relu_10(x, training=training)
            if stage == 9:
                x = self._torgb_10(x, training=training)

        if stage > 9:
            x = self._conv_1024x1024(x, training=training)          # (512, 512, 6) -> (1024, 1024, 4)
            x = self._bn_11(x, training=training)
            x = self._relu_11(x, training=training)
            x = self._torgb_11(x, training=training)

        return x


class Discriminator(tf.keras.Model):
    def __init__(self, activation_alpha: float = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # downscaling conv
        self._conv_512x512 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_256x256 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_128x128 = tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_64x64 = tf.keras.layers.Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_32x32 = tf.keras.layers.Conv2D(512, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_16x16 = tf.keras.layers.Conv2D(512, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_8x8 = tf.keras.layers.Conv2D(512, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_4x4 = tf.keras.layers.Conv2D(512, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)

        # from rgb
        self._fromrgb_3 = tf.keras.layers.Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)
        self._fromrgb_4 = tf.keras.layers.Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)
        self._fromrgb_5 = tf.keras.layers.Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)
        self._fromrgb_6 = tf.keras.layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)
        self._fromrgb_7 = tf.keras.layers.Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)
        self._fromrgb_8 = tf.keras.layers.Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)
        self._fromrgb_9 = tf.keras.layers.Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)
        self._fromrgb_10 = tf.keras.layers.Conv2D(16, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)

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
        self._flatten = tf.keras.layers.Flatten()
        self._output_dense = tf.keras.layers.Dense(1)

    def build(self, input_shape, stage):
        _input = tf.keras.layers.Input(shape=input_shape)
        self._init_graph_network(_input, self.call(_input, stage))

    def call(self, inputs, stage, training=None, mask=None):
        x = inputs

        if stage > 9:
            x = self._fromrgb_10(x, training=training)          # (1024, 1024, 3) -> (1024, 1024, 16)
            x = self._conv_512x512(x, training=training)        # (1024, 1024, 16) -> (512, 512, 32)
            x = self._bn_10(x, training=training)
            x = self._relu_10(x, training=training)

        if stage > 8:
            if stage == 9:
                x = self._fromrgb_9(x, training=training)       # (512, 512, 3) -> (512, 512, 32)
            x = self._conv_256x256(x, training=training)        # (512, 512, 32) -> (256, 256, 64)
            x = self._bn_9(x, training=training)
            x = self._relu_9(x, training=training)

        if stage > 7:
            if stage == 8:
                x = self._fromrgb_8(x, training=training)       # (256, 256, 3) -> (256, 256, 64)
            x = self._conv_128x128(x, training=training)        # (256, 256, 64) -> (128, 128, 128)
            x = self._bn_8(x, training=training)
            x = self._relu_8(x, training=training)

        if stage > 6:
            if stage == 7:
                x = self._fromrgb_7(x, training=training)       # (128, 128, 3) -> (128, 128, 128)
            x = self._conv_64x64(x, training=training)          # (128, 128, 128) -> (64, 64, 256)
            x = self._bn_7(x, training=training)
            x = self._relu_7(x, training=training)

        if stage > 5:
            if stage == 6:
                x = self._fromrgb_6(x, training=training)       # (64, 64, 3) -> (64, 64, 256)
            x = self._conv_32x32(x, training=training)          # (64, 64, 256) -> (32, 32, 512)
            x = self._bn_6(x, training=training)
            x = self._relu_6(x, training=training)

        if stage > 4:
            if stage == 5:
                x = self._fromrgb_5(x, training=training)       # (32, 32, 3) -> (32, 32, 512)
            x = self._conv_16x16(x, training=training)          # (32, 32, 512) -> (16, 16, 512)
            x = self._bn_5(x, training=training)
            x = self._relu_5(x, training=training)

        if stage > 3:
            if stage == 4:
                x = self._fromrgb_4(x, training=training)       # (16, 16, 3) -> (16, 16, 512)
            x = self._conv_8x8(x, training=training)            # (16, 16, 512) -> (8, 8, 512)
            x = self._bn_4(x, training=training)
            x = self._relu_4(x, training=training)

        if stage > 2:
            if stage == 3:
                x = self._fromrgb_3(x, training=training)       # (8, 8, 3) -> (8, 8, 512)
            x = self._conv_4x4(x, training=training)            # (8, 8, 512) -> (4, 4, 512)
            x = self._bn_3(x, training=training)
            x = self._relu_3(x, training=training)

        x = self._flatten(x, training=training)
        x = self._output_dense(x, training=training)
        return x


def celeb_a_generator(input_tensor, input_shape, **kwargs):
    if input_tensor is None:
        inputs = tf.keras.layers.Input(shape=input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            inputs = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            inputs = input_tensor

    x = inputs
    x = tf.keras.layers.Dense(512 * 4 * 4, use_bias=False)(x)
    x = tf.keras.layers.Reshape((4, 4, 512))(x)
    x = tf.keras.layers.Conv2DTranspose(256, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(128, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(16, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.Conv2DTranspose(8, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), activation='tanh', padding='same')(x)
    return tf.keras.models.Model(inputs=inputs, outputs=x, name='celeb_a_generator')


def celeb_a_discriminator(input_tensor, input_shape, noise_stddev: float = 0.25, **kwargs):
    if input_tensor is None:
        inputs = tf.keras.layers.Input(shape=input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            inputs = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            inputs = input_tensor

    x = inputs
    # x = tf.keras.layers.Conv2D(16, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    if noise_stddev != 0.0:
        x = tf.keras.layers.GaussianNoise(stddev=noise_stddev)(x)
        x = tf.clip_by_value(t=x, clip_value_min=0.0, clip_value_max=1.0)
    # x = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)
    return tf.keras.models.Model(inputs=inputs, outputs=x, name='celeb_a_discriminator')
