import tensorflow as tf


class Downsampling2D(tf.keras.layers.Layer):
    def __init__(self, factor: int = 2, data_format: str = 'NHWC', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(factor, int) and factor >= 1
        assert data_format in ['NHWC', 'NCHW']
        self._factor = factor
        self._data_format = data_format
        self._kernel = [1, self._factor, self._factor, 1]

    def call(self, inputs, **kwargs):
        return tf.nn.avg_pool(input=inputs, ksize=self._kernel, strides=self._kernel, padding='VALID',
                              data_format=self._data_format)

    def get_config(self):
        return {'factor': self._factor, 'data_format': self._data_format}


class PixelNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon: float = 1e-8, data_format: str = 'channel_last', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._epsilon = epsilon
        self._data_format = data_format
        self._axis = -1 if self._data_format in ['channel_last', 'NHWC'] else 1

    def call(self, inputs, **kwargs):
        return inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=self._axis, keepdims=True) + self._epsilon)

    def get_config(self):
        return {'epsilon': self._epsilon, 'data_format': self._data_format}


class StandardDeviationLayer(tf.keras.layers.Layer):
    def __init__(self, group_size: int = 4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError()

    def call(self, inputs, **kwargs):
        return inputs
