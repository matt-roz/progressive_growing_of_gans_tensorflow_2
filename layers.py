import logging

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import activations
from tensorflow.python.framework import tensor_shape

from utils import he_initializer_scale


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


class CustomDense(tf.keras.layers.Dense):
    def __init__(self, input_shape, units, gain=2.0,  use_weight_scaling=True, use_bias=True, activation=None, **kwargs):
        if 'bias_initializer' in kwargs:
            logging.warning(f"{self.__class__.__name__} ignores bias_initializer={kwargs['bias_initializer']}")
            del kwargs['bias_initializer']
        if 'kernel_initializer' in kwargs:
            logging.warning(f"{self.__class__.__name__} ignores kernel_initializer={kwargs['kernel_initializer']}")
            del kwargs['kernel_initializer']
        super(CustomDense, self).__init__(units=units, use_bias=use_bias, **kwargs)
        self.bias_initializer = tf.zeros_initializer()
        self.gain = gain
        self.use_weight_scaling = use_weight_scaling
        self.wrapper_use_bias = use_bias
        self.wrapper_activation = activations.get(activation)

        # compute kernel initializer
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        kernel_shape = (last_dim, self.units)
        he_scale = he_initializer_scale(shape=kernel_shape, gain=self.gain)
        if self.use_weight_scaling:
            self.conv_op_scale = he_scale
            self.kernel_initializer = tf.random_normal_initializer()
        else:
            self.conv_op_scale = 1.0
            self.kernel_initializer = tf.random_normal_initializer(0, he_scale)

    def build(self, input_shape):
        super(CustomDense, self).build(input_shape)
        self.use_bias = False

    def call(self, inputs):
        outputs = self.conv_op_scale * super(CustomDense, self).call(inputs)

        if self.wrapper_use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.wrapper_activation is not None:
            return self.wrapper_activation(outputs)
        return outputs

    def get_config(self):
        base_config = super(CustomDense, self).get_config()
        base_config['input_shape'] = self.input_shape
        base_config['gain'] = self.gain
        base_config['use_weight_scaling'] = self.use_weight_scaling
        base_config['activation'] = activations.serialize(self.wrapper_activation)  # overrides base config
        base_config['use_bias'] = self.wrapper_use_bias  # overrides base config

        return base_config


class CustomConv2D(tf.keras.layers.Conv2D):
    def __init__(self, input_shape, filters, kernel_size, gain=2.0, use_weight_scaling=True, use_bias=True, activation=None, **kwargs):
        if 'bias_initializer' in kwargs:
            logging.warning(f"{self.__class__.__name__} ignores bias_initializer={kwargs['bias_initializer']}")
            del kwargs['bias_initializer']
        if 'kernel_initializer' in kwargs:
            logging.warning(f"{self.__class__.__name__} ignores kernel_initializer={kwargs['kernel_initializer']}")
            del kwargs['kernel_initializer']
        super(CustomConv2D, self).__init__(filters=filters, use_bias=use_bias, kernel_size=kernel_size, **kwargs)
        self.bias_initializer = tf.zeros_initializer()
        self.gain = gain
        self.use_weight_scaling = use_weight_scaling
        self.wrapper_activation = activations.get(activation)
        self.wrapper_use_bias = use_bias

        # compute kernel initializer
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        kernel_shape = self.kernel_size + (input_channel, self.filters)
        he_scale = he_initializer_scale(shape=kernel_shape, gain=self.gain)
        if self.use_weight_scaling:
            self.conv_op_scale = he_scale
            self.kernel_initializer = tf.random_normal_initializer()
        else:
            self.conv_op_scale = 1.0
            self.kernel_initializer = tf.random_normal_initializer(0, he_scale)

    def build(self, input_shape):
        super(CustomConv2D, self).build(input_shape)
        self.use_bias = False

    def call(self, inputs, **kwargs):
        outputs = self.conv_op_scale * super(CustomConv2D, self).call(inputs)

        if self.wrapper_use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.wrapper_activation is not None:
            return self.wrapper_activation(outputs)
        return outputs

    def get_config(self):
        base_config = super(CustomConv2D, self).get_config()
        base_config['input_shape'] = self.input_shape
        base_config['gain'] = self.gain
        base_config['use_weight_scaling'] = self.use_weight_scaling
        base_config['activation'] = activations.serialize(self.wrapper_activation)  # overrides base config
        base_config['use_bias'] = self.wrapper_use_bias  # overrides base config
        return base_config


class StandardDeviationLayer(tf.keras.layers.Layer):
    def __init__(self, epsilon: float = 1e-8, data_format: str = 'NHWC', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(epsilon, float) and epsilon > 0
        assert data_format in ['NHWC', 'NCHW']
        self._epsilon = epsilon
        self._data_format = data_format
        self._channel_axis = -1 if self._data_format == 'NHWC' else 1
        if self._data_format == 'NCHW':
            raise NotImplementedError()  #TODO(M. Rozanski): channel implementation

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



