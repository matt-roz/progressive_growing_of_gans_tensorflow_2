import tensorflow as tf
import numpy as np

class DownSampling2D(tf.keras.layers.Layer):
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

    def compute_output_shape(self, input_shape):
        channel_last = self._data_format == 'NHWC'
        if channel_last:
            n, h, w, c = input_shape
        else:
            n, c, h, w = input_shape
        _h = tf.math.ceil(h / self._factor)
        _w = tf.math.ceil(w / self._factor)
        return (n, _h, _w, c) if channel_last else (n, c, _h, _w)

    def get_config(self):
        return {'factor': self._factor, 'data_format': self._data_format}


class PixelNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon: float = 1e-8, data_format: str = 'NHWC', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(epsilon, float) and epsilon > 0
        assert data_format in ['NHWC', 'NCHW']
        self._epsilon = epsilon
        self._data_format = data_format
        self._axis = -1 if self._data_format == 'NHWC' else 1

    def call(self, inputs, **kwargs):
        return inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=self._axis, keepdims=True) + self._epsilon)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {'epsilon': self._epsilon, 'data_format': self._data_format}


class StandardDeviationLayer(tf.keras.layers.Layer):
    def __init__(self, epsilon: float = 1e-8, data_format: str = 'NHWC', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(epsilon, float) and epsilon > 0
        assert data_format in ['NHWC', 'NCHW']
        self._epsilon = epsilon
        self._data_format = data_format
        self._channel_axis = -1 if self._data_format == 'NHWC' else 1
        if self._data_format == 'NCHW':
            raise NotImplementedError()  # M. Rozanski: TODO

    def call(self, inputs, **kwargs):
        mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
        mean_square_diff = tf.reduce_mean(tf.math.square(inputs - mean), axis=0, keepdims=True)
        mean_square_diff += self._epsilon
        stddev = tf.sqrt(mean_square_diff)
        mean_stddev = tf.reduce_mean(stddev, keepdims=True)
        input_shape = tf.shape(inputs)
        outputs = tf.tile(mean_stddev, (input_shape[0], input_shape[1], input_shape[2], 1))
        return tf.concat([inputs, outputs], axis=self._channel_axis)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        shape = list(input_shape)
        shape[self._channel_axis] += 1
        return tuple(shape)

    def get_config(self):
        return {'epsilon': self._epsilon, 'data_format': self._data_format}


def he_initializer_scale(shape, slope=1.0):
    fan_in = np.prod(shape[:-1])
    return np.sqrt(2. / ((1. + slope**2) * fan_in))


def _custom_layer_impl(apply_kernel, kernel_shape, bias_shape, activation, name,
                       he_initializer_slope, use_weight_scaling):
    kernel_scale = he_initializer_scale(kernel_shape, he_initializer_slope)
    init_scale, post_scale = kernel_scale, 1.0
    if use_weight_scaling:
        init_scale, post_scale = post_scale, init_scale

    kernel_initializer = tf.random_normal_initializer(stddev=init_scale)

    bias = tf.Variable(np.zeros(shape=bias_shape, dtype=np.float32), dtype=tf.float32, name=f"{name}/bias")

    output = post_scale * apply_kernel(kernel_shape, kernel_initializer) + bias

    if activation is not None:
        output = activation(output)
    return output


def custom_conv2d(x,
                  filters,
                  kernel_size,
                  name,
                  strides=(1, 1),
                  padding='SAME',
                  activation=None,
                  he_initializer_slope=1.0,
                  use_weight_scaling=True):
    if not isinstance(kernel_size, (list, tuple)):
        kernel_size = [kernel_size] * 2
        kernel_size = list(kernel_size)

    def _apply_kernel(kernel_shape, kernel_initializer):
        return tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_shape[0:2],
            strides=strides,
            padding=padding,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            name=name
        )(x)

    return _custom_layer_impl(
        _apply_kernel,
        kernel_shape=kernel_size + [x.shape.as_list()[3], filters],
        bias_shape=(filters,),
        activation=activation,
        name=name,
        he_initializer_slope=he_initializer_slope,
        use_weight_scaling=use_weight_scaling
    )


def custom_dense(x,
                 units,
                 name,
                 activation=None,
                 he_initializer_slope=1.0,
                 use_weight_scaling=True):

    def _apply_kernel(kernel_shape, kernel_initializer):
        return tf.keras.layers.Dense(
            kernel_shape[1],
            use_bias=False,
            kernel_initializer=kernel_initializer,
            name=name
        )(x)

    return _custom_layer_impl(
        _apply_kernel,
        kernel_shape=(x.shape.as_list()[-1], units),
        bias_shape=(units,),
        activation=activation,
        name=name,
        he_initializer_slope=he_initializer_slope,
        use_weight_scaling=use_weight_scaling
    )
