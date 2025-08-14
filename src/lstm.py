from datetime import datetime
from preprocessor import Preprocessor
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.metrics import Accuracy, Precision, Recall
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from message_encoder import *
from keras.optimizers import Adam, RMSprop, SGD, Nadam
import os

from keras import Model, Input, layers
from keras.saving import register_keras_serializable

@register_keras_serializable(package="custom")
class LSTMClassifier(layers.Layer):
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
                layers.LSTM(
                    self.units,
                    return_sequences=return_sequences,
                    dropout=self.dropout,
                    recurrent_dropout=self.recurrent_dropout,
                )
            )
        self._classifier = layers.Dense(self.num_classes, activation="softmax")

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






























# --- after you have X_train / y_train ---
num_classes = len(np.unique(y_train))
input_shape = pp.data.entry_shape  # (timesteps, features)

inputs = Input(shape=input_shape)
outputs = LSTMClassifier(
    num_classes=num_classes,
    lstm_layers=lstm_layers,
    units=lstm_units_per_layer,
    dropout=dropout,
    recurrent_dropout=recurrent_dropout,
    name="lstm_classifier",
)(inputs)

model = Model(inputs, outputs, name="logs_lstm_model")
model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

early_stopping = EarlyStopping(
    monitor=early_stopping_monitor,
    patience=early_stopping_patience,
    restore_best_weights=early_stopping_restore_best
)
model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split,
    callbacks=[early_stopping]
)
