import tensorflow as tf
import keras

class PositionalEncoding(keras.layers.Layer):
    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        pos_encoding = tf.cast(positions[:, tf.newaxis], tf.float32)
        return x + tf.math.sin(pos_encoding)