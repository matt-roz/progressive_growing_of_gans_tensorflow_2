import logging
import warnings

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import activations
from tensorflow.python.framework import tensor_shape

from utils import he_kernel_initializer, he_initializer_scale

_channel_choices = ['NHWC', 'NCHW', 'channel_last', 'channel_first']


# ----------------------------------------------------------------------------------------------------------------------
# Custom Layers and Wrappers according to their publications
# ----------------------------------------------------------------------------------------------------------------------


class PixelNormalization(tf.keras.layers.Layer):
    """A Layer implementation of PixelNormalization as described in https://arxiv.org/abs/1710.10196.
    original implementation: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120

    Normalizes the feature vector (on channel_axis) in each pixel to unit length. Used in the generator after each Conv.

    Args:
        epsilon: small value for rsqrt to avoid division by zero
        data_format: specifies the channel dimension
        *args: passed down arguments to super().__init__()
        **kwargs: passed down keyword-arguments to super().__init__()

    Attributes:
        epsilon (float): epsilon for sqrt calculation
        data_format (str): specifies the channel dimension
        channel_axis (int): depicts the axis of the channels/features

    Raises:
        TypeError: if epsilon is not of type float
        ValueError: if data_format is invalid
    """
    def __init__(self, epsilon: float = 1e-8, data_format: str = 'channel_last', *args, **kwargs):
        if not isinstance(epsilon, float):
            raise TypeError(f"epsilon must be of type {float} but found {type(epsilon)} instead")
        if data_format not in _channel_choices:
            raise ValueError(f"data_format must be one of {_channel_choices} but found {data_format} instead")
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.data_format = data_format
        self.channel_axis = -1 if self.data_format == 'NHWC' or self.data_format == 'channel_last' else 1

    def call(self, inputs, **kwarg):
        scale = tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=self.channel_axis, keepdims=True) + self.epsilon)
        return inputs * scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(PixelNormalization, self).get_config()
        base_config['epsilon'] = self.epsilon
        base_config['data_format'] = self.data_format
        return base_config


class StandardDeviationLayer(tf.keras.layers.Layer):
    """A layer implementation of StandardDeviationLayer as proposed in https://arxiv.org/abs/1710.10196.
    original implementation: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L127

    A layer concatenating batch statistics to the channel_axis of a 4D Tensor. Used in the last block (stage 2) of the
    discriminator.

    Args:
        epsilon: small value for stability in square root calculation
        data_format: specifies the channel dimension
        **kwargs: passed down keyword-arguments to super().__init__()

    Attributes:
        epsilon (float): epsilon for sqrt calculation
        data_format (str): specifies the channel dimension
        channel_axis (int): depicts the axis of the channels/features

    Raises:
        TypeError: if epsilon is not of type float
        ValueError: if data_format is invalid
    """
    def __init__(self, epsilon: float = 1e-8, data_format: str = 'channel_last', **kwargs):
        if not isinstance(epsilon, float):
            raise TypeError(f"epsilon must be of type {float} but found {type(epsilon)} instead")
        if data_format not in _channel_choices:
            raise ValueError(f"data_format must be one of {_channel_choices} but found {data_format} instead")
        super(StandardDeviationLayer, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.data_format = data_format
        self.channel_axis = -1 if self.data_format == 'NHWC' or self.data_format == 'channel_last' else 1

    def call(self, inputs, **kwargs):
        mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
        mean_square_diff = tf.reduce_mean(tf.math.square(inputs - mean), axis=0, keepdims=True)
        stddev = tf.sqrt(mean_square_diff + self.epsilon)
        mean_stddev = tf.reduce_mean(stddev, keepdims=True)
        input_shape = tf.shape(inputs)
        if self.channel_axis == -1:
            feature_shape = (input_shape[0], input_shape[1], input_shape[2], 1)
        else:
            feature_shape = (input_shape[0], 1, input_shape[1], input_shape[2])
        feature = tf.tile(mean_stddev, feature_shape)
        return tf.concat([inputs, feature], axis=self.channel_axis)

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(f"{self.__class__.__name__} requires a rank=4 tensor but received rank={len(input_shape)}")
        shape = list(input_shape)
        shape[self.channel_axis] += 1
        return tuple(shape)

    def get_config(self):
        base_config = super(StandardDeviationLayer, self).get_config()
        base_config['epsilon'] = self.epsilon
        base_config['data_format'] = self.data_format
        return base_config


class WeightScalingWrapper(tf.keras.layers.Wrapper):
    """A Layer-Wrapper to allow the weight scaling trick as described in https://arxiv.org/abs/1710.10196.
    original implementation: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L22

    Logs warnings if 'bias_initializer' and 'kernel_initializer' are not of appropriate types, since the reference
    implementation requires bias to be initialized with zeros and kernel with random_normal. The weight scaling trick
    applies a scalar 'weight_scale' to the kernel output during the forward pass:

        output = weight_scale * layer_kernel_op(inputs) + layer_bias

    Both add_bias and activation from layer must be delayed after weight_scale has been applied to the kernel output.
    For this reason this implementation deactivates layer.use_bias after layer.build() and wraps 'use_bias' as well as
    'activation' in the call() forward pass after weight_scale has been applied to layer.call().

    Args:
        layer: layer to be wrapped and weight_scaled at runtime
        gain: gain for he_initializer_scale
        **kwargs: passed down keyword-arguments to super().__init__()

    Attributes:
        gain: gain for he_initializer_scale
        has_bias: whether or not wrapped layer has attribute 'bias' or 'use_bias'
        use_bias: wrapped 'use_bias' from layer (False if non-existent)
        has_activation: whether or not wrapped layer has attribute 'activation'
        activation: wrapped 'activation' from layer (None if non-existent)
        has_data_format:  whether or not wrapped layer has attribute 'data_format'
        weight_scale: float scalar to weight_scale wrapped layer with

    Raises:
        ValueError: if layer is not an instance of tf.keras.layers.Layer
    """
    def __init__(self, layer, gain, **kwargs):
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(f"wrapped layer must be type {tf.keras.layers.Layer} but received {type(layer)} instead")
        name = kwargs.pop('name', f"{layer.name}/WeightScaled{layer.__class__.__name__}")
        super(WeightScalingWrapper, self).__init__(layer=layer, name=name, **kwargs)
        self.gain = gain

        # get info from layer that is to be wrapped
        self.has_bias = hasattr(self.layer, 'bias') or hasattr(self.layer, 'use_bias')
        self.use_bias = self.has_bias and self.layer.use_bias
        self.has_activation = hasattr(self.layer, 'activation')
        self.activation = None
        self.has_data_format = hasattr(self.layer, 'data_format')
        self.weight_scale = tf.Variable(0.0, False, dtype=tf.float32, name=f'{self.layer.name}/weight_scale')

        # check if layer has correct initializers set up - log warnings
        if self.use_bias and not isinstance(self.layer.bias_initializer, tf.keras.initializers.Zeros):
            logging.warning(f"bias_initializer of wrapped layer should be instance of {tf.keras.initializers.Zeros} "
                            f"but found {type(self.layer.bias_initializer)} instead.")
        if not isinstance(self.layer.kernel_initializer, tf.keras.initializers.RandomNormal):
            logging.warning(f"kernel_initializer of wrapped layer should be instance of "
                            f"{tf.keras.initializers.RandomNormal} but found {type(self.layer.kernel_initializer)} "
                            f"instead.")

    def build(self, input_shape=None):
        if not self.layer.built:
            self.layer.build(input_shape=input_shape)

        weight_scale = he_initializer_scale(kernel_shape=self.layer.kernel.shape, gain=self.gain)
        self.weight_scale.assign(weight_scale)

        # deactivate layer bias since wrapper takes care of it
        if self.has_bias:
            self.layer.use_bias = False

        # redirect activation output (wrapper takes care of it)
        if self.has_activation:
            self.activation = self.layer.activation
            self.layer.activation = None
        self.built = True

    def call(self, inputs, **kwargs):
        # apply weight scaling trick to kernel output (bias and activation circumvented)
        outputs = self.weight_scale * self.layer.call(inputs)

        # proceed to add bias and use activation based on wrapped configuration
        if self.use_bias:
            if self.has_data_format:
                if self.layer.data_format == 'channels_first':
                    if self.layer.rank == 1:
                        # nn.bias_add does not accept a 1D input tensor.
                        bias = array_ops.reshape(self.layer.bias, (1, self.layer.filters, 1))
                        outputs += bias
                    else:
                        outputs = tf.nn.bias_add(outputs, self.layer.bias, data_format='NCHW')
                else:
                    outputs = tf.nn.bias_add(outputs, self.layer.bias, data_format='NHWC')
            else:
                outputs = tf.nn.bias_add(outputs, self.layer.bias)

        # wrapped activation
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape=input_shape)

    def get_config(self):
        base_config = super(WeightScalingWrapper, self).get_config()
        base_config['gain'] = self.gain
        base_config['layer']['config']['activation'] = activations.serialize(self.activation)  # overrides base config
        base_config['layer']['config']['use_bias'] = self.use_bias                             # overrides base config
        return base_config


# ----------------------------------------------------------------------------------------------------------------------
# deprecated layers (kept for backwards compatibility for trained and serialized h5 models)
# ----------------------------------------------------------------------------------------------------------------------


class CustomDense(tf.keras.layers.Dense):
    """A wrapper around Dense to allow the weight scaling trick as described in https://arxiv.org/abs/1710.10196.
    original implementation: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L34

    Disallow the arguments 'bias_initializer', 'kernel_initializer' to be used, since the implementation requires bias
    to be initialized with zeros and kernel with a custom he_scale init. The weight scaling trick applies op_scale
    to the kernel output during the forward pass:

        output = op_scale * kernel_op(inputs) + bias

    Both add_bias and activation from super() must be delayed after op_scale has been applied to the kernel output. For
    this reason this implementation deactivates use_bias after super().build() and wraps 'use_bias' as well as
    'activation' in the call() forward pass after op_scale has been applied to super().call().

    Args:
        input_shape: shape of input_tensor must be known at layer instantiation since op_scale depends on kernel_shape
        units: number of units for dense layer
        gain: gain for he_initializer_scale
        use_weight_scaling: whether or not to use the weight scale trick
        use_bias: whether or not to use a bias on kernel output
        activation: activation on kernel output + bias
        **kwargs: passed down keyword-arguments to super().__init__()

    Attributes:
        bias_initializer: initializer for bias
        kernel_initializer: initializer for kernel
        gain: gain for he_initializer_scale
        use_weight_scaling: whether or not to use the weight scale trick
        op_scale (float): scalar used to scale the output after kernel_op
        _wrapper_use_bias (bool): wrapper for use_bias
        _wrapper_activation (Callable): wrapper for activation
        _argument_input_shape: placeholder for serialization via get_config
    """
    def __init__(self, input_shape, units, gain=2.0, use_weight_scaling=True, use_bias=True, activation=None, **kwargs):
        warnings.warn(f"CustomDense is deprecated. Use tf.keras.layers.Dense wrapped in WeightScalingWrapper instead.",
                      DeprecationWarning, 2)
        if 'bias_initializer' in kwargs:
            logging.warning(f"{self.__class__.__name__} ignores bias_initializer={kwargs['bias_initializer']}")
            del kwargs['bias_initializer']
        if 'kernel_initializer' in kwargs:
            logging.warning(f"{self.__class__.__name__} ignores kernel_initializer={kwargs['kernel_initializer']}")
            del kwargs['kernel_initializer']
        super(CustomDense, self).__init__(units=units, use_bias=use_bias, **kwargs)
        self.bias_initializer = tf.keras.initializers.zeros()
        self.gain = gain
        self.use_weight_scaling = use_weight_scaling
        self._wrapper_use_bias = use_bias
        self._wrapper_activation = activations.get(activation)
        self._argument_input_shape = input_shape

        # compute kernel shape
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        kernel_shape = (last_dim, self.units)
        self.op_scale, self.kernel_initializer = he_kernel_initializer(kernel_shape, self.gain, self.use_weight_scaling)

    def build(self, input_shape):
        super(CustomDense, self).build(input_shape)  # instantiates kernel and bias
        self.use_bias = False                        # wrapper_use_bias determines bias usage

    def call(self, inputs):
        # apply weight scaling trick to kernel output
        outputs = self.op_scale * super(CustomDense, self).call(inputs)
        # proceed to add bias and use activation based on wrapped vars
        if self._wrapper_use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self._wrapper_activation is not None:
            return self._wrapper_activation(outputs)
        return outputs

    def get_config(self):
        base_config = super(CustomDense, self).get_config()
        base_config['input_shape'] = self._argument_input_shape
        base_config['gain'] = self.gain
        base_config['use_weight_scaling'] = self.use_weight_scaling
        base_config['activation'] = activations.serialize(self._wrapper_activation)  # overrides base config
        base_config['use_bias'] = self._wrapper_use_bias                             # overrides base config
        base_config.pop('bias_initializer', None)
        base_config.pop('kernel_initializer', None)
        return base_config


class CustomConv2D(tf.keras.layers.Conv2D):
    """A wrapper around Conv2D to allow the weight scaling trick as described in https://arxiv.org/abs/1710.10196.
    original implementation: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L44

    Disallow the arguments 'bias_initializer', 'kernel_initializer' to be used, since the implementation requires bias
    to be initialized with zeros and kernel with a custom he_scale init. The weight scaling trick applies op_scale
    to the kernel output during the forward pass:

        output = op_scale * kernel_op(inputs) + bias

    Both add_bias and activation from super() must be delayed after op_scale has been applied to the kernel output. For
    this reason this implementation deactivates use_bias after super().build() and wraps 'use_bias' as well as
    'activation' in the call() forward pass after op_scale has been applied to super().call().

    Args:
        input_shape: shape of input_tensor must be known at layer instantiation since op_scale depends on kernel_shape
        filters: the dimensionality of the output space
        kernel_size: specifying the height and width of the 2D convolution window
        gain: gain for he_initializer_scale
        use_weight_scaling: whether or not to use the weight scale trick
        use_bias: whether or not to use a bias on kernel output
        activation: activation on kernel output + bias
        **kwargs: passed down keyword-arguments to super().__init__()

    Attributes:
        bias_initializer: initializer for bias
        kernel_initializer: initializer for kernel
        gain: gain for he_initializer_scale
        use_weight_scaling: whether or not to use the weight scale trick
        op_scale (float): scalar used to scale the output after kernel_op
        _wrapper_use_bias (bool): wrapper for use_bias
        _wrapper_activation (Callable): wrapper for activation
        _argument_input_shape: placeholder for serialization via get_config
    """
    def __init__(self, input_shape, filters, kernel_size, gain=2.0, use_weight_scaling=True, use_bias=True,
                 activation=None, **kwargs):
        warnings.warn(f"CustomConv2D is deprecated. Use tf.keras.layers.Conv2D wrapped in WeightScalingWrapper "
                      f"instead.", DeprecationWarning, 2)
        if 'bias_initializer' in kwargs:
            logging.warning(f"{self.__class__.__name__} ignores bias_initializer={kwargs['bias_initializer']}")
            del kwargs['bias_initializer']
        if 'kernel_initializer' in kwargs:
            logging.warning(f"{self.__class__.__name__} ignores kernel_initializer={kwargs['kernel_initializer']}")
            del kwargs['kernel_initializer']
        super(CustomConv2D, self).__init__(filters=filters, kernel_size=kernel_size, use_bias=use_bias, **kwargs)
        self.bias_initializer = tf.keras.initializers.zeros()
        self.gain = gain
        self.use_weight_scaling = use_weight_scaling
        self._wrapper_activation = activations.get(activation)
        self._wrapper_use_bias = use_bias
        self._argument_input_shape = input_shape

        # compute kernel initializer
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        kernel_shape = self.kernel_size + (input_channel, self.filters)
        self.op_scale, self.kernel_initializer = he_kernel_initializer(kernel_shape, self.gain, self.use_weight_scaling)

    def build(self, input_shape):
        super(CustomConv2D, self).build(input_shape)  # instantiates kernel and bias
        self.use_bias = False                         # wrapper_use_bias determines bias usage

    def call(self, inputs, **kwargs):
        # apply weight scaling trick to kernel output
        outputs = self.op_scale * super(CustomConv2D, self).call(inputs)
        # proceed to add bias and use activation based on wrapped vars
        if self._wrapper_use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self._wrapper_activation is not None:
            return self._wrapper_activation(outputs)
        return outputs

    def get_config(self):
        base_config = super(CustomConv2D, self).get_config()
        base_config['input_shape'] = self._argument_input_shape
        base_config['gain'] = self.gain
        base_config['use_weight_scaling'] = self.use_weight_scaling
        base_config['activation'] = activations.serialize(self._wrapper_activation)  # overrides base config
        base_config['use_bias'] = self._wrapper_use_bias                             # overrides base config
        base_config.pop('bias_initializer', None)
        base_config.pop('kernel_initializer', None)
        return base_config
