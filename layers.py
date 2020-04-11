import tensorflow as tf


class Upscale2D(tf.keras.layers.Layer):
    def __init__(self, factor: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(factor, int) and factor >= 1
        self._factor = tf.Variable(initial_value=factor, trainable=False, dtype=tf.int32)

    def call(self, inputs, **kwargs):
        s = tf.shape(inputs)
        x = tf.image.resize(inputs, (s[1]*self._factor, s[2]*self._factor), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return x


class Downscale2D(tf.keras.layers.Layer):
    def __init__(self, factor: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(factor, int) and factor >= 1
        self._kernel = [1, factor, factor, 1]

    def call(self, inputs, **kwargs):
        return tf.nn.avg_pool(input=inputs, ksize=self._kernel, strides=self._kernel, padding='VALID')


class PixelNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon: float = 1e-8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._epsilon = epsilon  # tf.Variable(initial_value=epsilon, trainable=False, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        return inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=3, keepdims=True) + self._epsilon)


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


if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    img = Image.open('flower.jpg')
    img.show('original')

    flower = np.array(img)
    flower = flower.astype(dtype=np.float32) / 255.0
    flowers = np.empty(shape=(2, 900, 900, 3), dtype=np.float32)
    flowers[0, :] = flower[:]
    flowers[1, :] = flower[:]

    val = StandardDeviationLayer()

    d = val(flowers)

    print(d.shape)