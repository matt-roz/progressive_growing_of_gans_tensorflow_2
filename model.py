import tensorflow as tf


class Generator(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # input feature layer
        self._feature_dense = tf.keras.layers.Dense(units=512 * 4 * 4, use_bias=False)
        self._feature_reshape = tf.keras.layers.Reshape(target_shape=(4, 4, 512))

        # upscaling Conv2Ds
        # self._conv_1x1 = tf.keras.layers.Conv2DTranspose(2048, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        # self._conv_2x2 = tf.keras.layers.Conv2DTranspose(1024, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        # self._conv_4x4 = tf.keras.layers.Conv2DTranspose(512, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_8x8 = tf.keras.layers.Conv2DTranspose(256, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_16x16 = tf.keras.layers.Conv2DTranspose(128, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_32x32 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_64x64 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_128x128 = tf.keras.layers.Conv2DTranspose(16, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        # self._conv_256x256 = tf.keras.layers.Conv2DTranspose(8, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        # self._conv_512x512 = tf.keras.layers.Conv2DTranspose(6, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        # self._conv_1024x1024 = tf.keras.layers.Conv2DTranspose(4, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)

        # to rgb conv
        self._torgb = tf.keras.layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='tanh', use_bias=False)

        # batchnorms
        self._bn_1 = tf.keras.layers.BatchNormalization()
        self._bn_2 = tf.keras.layers.BatchNormalization()
        self._bn_3 = tf.keras.layers.BatchNormalization()
        self._bn_4 = tf.keras.layers.BatchNormalization()
        self._bn_5 = tf.keras.layers.BatchNormalization()
        self._bn_6 = tf.keras.layers.BatchNormalization()
        self._bn_7 = tf.keras.layers.BatchNormalization()
        self._bn_8 = tf.keras.layers.BatchNormalization()

        # leaky relu
        self._relu_1 = tf.keras.layers.LeakyReLU()
        self._relu_2 = tf.keras.layers.LeakyReLU()
        self._relu_3 = tf.keras.layers.LeakyReLU()
        self._relu_4 = tf.keras.layers.LeakyReLU()
        self._relu_5 = tf.keras.layers.LeakyReLU()
        self._relu_6 = tf.keras.layers.LeakyReLU()
        self._relu_7 = tf.keras.layers.LeakyReLU()
        self._relu_8 = tf.keras.layers.LeakyReLU()

    def build(self, input_shape):
        _input = tf.keras.layers.Input(shape=input_shape)
        self._init_graph_network(_input, self.call(_input))

    def call(self, inputs, training=None, mask=None, stage=None):
        x = inputs
        x = self._feature_dense(x, training=training)
        x = self._feature_reshape(x, training=training)

        x = self._conv_8x8(x, training=training)
        x = self._bn_1(x, training=training)
        x = self._relu_1(x, training=training)

        x = self._conv_16x16(x, training=training)
        x = self._bn_2(x, training=training)
        x = self._relu_2(x, training=training)

        x = self._conv_32x32(x, training=training)
        x = self._bn_3(x, training=training)
        x = self._relu_3(x, training=training)

        x = self._conv_64x64(x, training=training)
        x = self._bn_4(x, training=training)
        x = self._relu_4(x, training=training)

        x = self._conv_128x128(x, training=training)
        x = self._bn_5(x, training=training)
        x = self._relu_5(x, training=training)

        x = self._torgb(x, training=training)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self, noise_stddev: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._noise_stddev = noise_stddev
        self._input_noise = tf.keras.layers.GaussianNoise(stddev=self._noise_stddev)
        self._fromrgb = tf.keras.layers.Conv2D(16, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)

        # downscaling conv
        # self._conv_1024x1024 = tf.keras.layers.Conv2D(4, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        # self._conv_512x512 = tf.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        # self._conv_256x256 = tf.keras.layers.Conv2D(8, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_128x128 = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_64x64 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_32x32 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_16x16 = tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_8x8 = tf.keras.layers.Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self._conv_4x4 = tf.keras.layers.Conv2D(512, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        # self._conv_2x2 = tf.keras.layers.Conv2D(1024, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        # self._conv_1x1 = tf.keras.layers.Conv2D(2048, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)

        # batchnorms
        self._bn_1 = tf.keras.layers.BatchNormalization()
        self._bn_2 = tf.keras.layers.BatchNormalization()
        self._bn_3 = tf.keras.layers.BatchNormalization()
        self._bn_4 = tf.keras.layers.BatchNormalization()
        self._bn_5 = tf.keras.layers.BatchNormalization()
        self._bn_6 = tf.keras.layers.BatchNormalization()
        self._bn_7 = tf.keras.layers.BatchNormalization()
        self._bn_8 = tf.keras.layers.BatchNormalization()

        # leaky relu
        self._relu_1 = tf.keras.layers.LeakyReLU()
        self._relu_2 = tf.keras.layers.LeakyReLU()
        self._relu_3 = tf.keras.layers.LeakyReLU()
        self._relu_4 = tf.keras.layers.LeakyReLU()
        self._relu_5 = tf.keras.layers.LeakyReLU()
        self._relu_6 = tf.keras.layers.LeakyReLU()
        self._relu_7 = tf.keras.layers.LeakyReLU()
        self._relu_8 = tf.keras.layers.LeakyReLU()

        # output layer
        self._flatten = tf.keras.layers.Flatten()
        self._output_dense = tf.keras.layers.Dense(1)

    def build(self, input_shape):
        _input = tf.keras.layers.Input(shape=input_shape)
        self._init_graph_network(_input, self.call(_input))

    def call(self, inputs, training=None, mask=None, stage=None):
        x = inputs
        x = self._fromrgb(x, training=training)

        if self._noise_stddev:
            x = self._input_noise(x, training=training)
            x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0)

        x = self._conv_64x64(x, training=training)
        x = self._bn_1(x, training=training)
        x = self._relu_1(x, training=training)

        x = self._conv_32x32(x, training=training)
        x = self._bn_2(x, training=training)
        x = self._relu_2(x, training=training)

        x = self._conv_16x16(x, training=training)
        x = self._bn_3(x, training=training)
        x = self._relu_3(x, training=training)

        x = self._conv_8x8(x, training=training)
        x = self._bn_4(x, training=training)
        x = self._relu_4(x, training=training)

        x = self._conv_4x4(x, training=training)
        x = self._bn_5(x, training=training)
        x = self._relu_5(x, training=training)

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
    x = (x + 1.0) / 2.0
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
