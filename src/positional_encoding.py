import tensorflow as tf
import keras

class PositionalEncoding(keras.layers.Layer):
    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        pos_encoding = tf.cast(positions[:, tf.newaxis], tf.float32)
        return x + tf.math.sin(pos_encoding)


class PositionalEncoding2(keras.layers.Layer):
    def __init__(self, n=10000, **kwargs):
        super().__init__(**kwargs)
        self.n = float(n)

    def get_config(self):
        config = super().get_config()
        config.update({
            "n": self.n
            })
        return config

    def call(self, x):
        embed_dim = tf.shape(x)[2]
        
        seq_len = tf.shape(x)[1]
        pos = tf.range(start=0, limit=seq_len, delta=1, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(start=0, limit=embed_dim, delta=1, dtype=tf.float32)[tf.newaxis, :]

        angle_rates = 1 / tf.pow(self.n, tf.cast(2 * (i // 2), tf.float32) / tf.cast(embed_dim, tf.float32))
        angle_rads = pos * angle_rates

        # Apply sin to even indices and cos to odd indices
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]  # Shape: (1, seq_len, embed_dim)

        return x + pos_encoding
