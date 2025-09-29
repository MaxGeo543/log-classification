from keras.layers import LSTM, Dense, Layer
from keras import layers
from keras.saving import register_keras_serializable
import tensorflow as tf


@register_keras_serializable(package="custom")
class LSTMClassifierLayer(Layer):
    """
    A reusable LSTM stack + classifier, implemented as a Keras Layer.
    You can plug this into a Functional model or Sequential.
    """
    def __init__(
        self,
        num_classes: int,
        lstm_layers: int = 1,
        units: int = 50,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = int(num_classes)
        self.lstm_layers = int(lstm_layers)
        self.units = int(units)
        self.dropout = float(dropout)
        self.recurrent_dropout = float(recurrent_dropout)

        # Let Keras know masks can flow through this layer (useful if you ever pass a mask)
        self.supports_masking = True

        # Sub-layers are created here if their config doesn't depend on input shape.
        # (We can also create them in build(); both are fine as long as they're tracked.)
        self._lstm_stack = []
        for i in range(self.lstm_layers):
            return_sequences = i < (self.lstm_layers - 1)
            self._lstm_stack.append(
                LSTM(
                    self.units,
                    return_sequences=return_sequences,
                    dropout=self.dropout,
                    recurrent_dropout=self.recurrent_dropout,
                )
            )
        self._classifier = Dense(self.num_classes, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for lstm in self._lstm_stack:
            x = lstm(x, training=training, mask=mask)
        return self._classifier(x)

    def get_config(self):
        # Enable full serialization (SavedModel / Keras v3 format)
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "lstm_layers": self.lstm_layers,
                "units": self.units,
                "dropout": self.dropout,
                "recurrent_dropout": self.recurrent_dropout,
            }
        )
        return config

    def build(self, input_shape):
        # input_shape: (batch, timesteps, features)
        x_shape = tf.TensorShape(input_shape)

        for lstm in self._lstm_stack:
            # Build current LSTM with the incoming shape
            lstm.build(x_shape)

            # Compute outgoing shape for the next layer
            if lstm.return_sequences:
                # (batch, timesteps, units)
                x_shape = tf.TensorShape([x_shape[0], x_shape[1], lstm.units])
            else:
                # (batch, units)
                x_shape = tf.TensorShape([x_shape[0], lstm.units])

        # Build classifier on final shape
        self._classifier.build(x_shape)

        # Important: mark this layer as built
        super().build(input_shape)
