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
        self._group_size = tf.Variable(initial_value=group_size, trainable=False, dtype=tf.int32)

    def call(self, inputs, **kwargs):
        x = inputs
        group_size = tf.minimum(self._group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NHWC]  Input shape.
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMHWC] Split minibatch into M groups of size G.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMHWC] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MHWC]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MHWC]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)    # [M111]  Take average over fmaps and pixels.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [NHW1]  Replicate over group and pixels.
        return tf.concat([x, y], axis=1)                       # [NHWC]  Append as new fmap.
