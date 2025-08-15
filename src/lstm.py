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
from keras.optimizers import Adam, RMSprop, SGD, Nadam
import os

from keras import Model, Input, layers
from keras.saving import register_keras_serializable
from dataset import Dataset
from pseudolabeling import DynamicDataset, PseudoLabelingCallback

@register_keras_serializable(package="custom")
class LSTMClassifierLayer(layers.Layer):
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



class LSTMClassifier:
    def __init__(self, 
                 preprocessor: Preprocessor, 
                 dataset: Dataset,

                 data_split_ratios: tuple[int, int, int] = (4, 1, 1), # Train, validation, test split ratios
                 
                 lstm_layers: int = 1,
                 lstm_units: int = 12,
                 dropout: float = 0.2,
                 recurrent_dropout: float = 0.2,

                 metrics: list[str] = ["accuracy"]
                 ):
        self.pp = preprocessor
        self.dataset = dataset

        
        
        self.training_data, self.validation_data, self.test_data = dataset.stratified_split(data_split_ratios)
        self.sparse = len(self.dataset.data_array_y[0]) == 1
        
        self.training_data: tuple[np.ndarray, np.ndarray] = self.training_data
        self.validation_data: tuple[np.ndarray, np.ndarray] = self.validation_data
        self.test_data: tuple[np.ndarray, np.ndarray] = self.test_data

        inputs = Input(shape=dataset.entry_shape)
        outputs = LSTMClassifierLayer(
            num_classes=len(preprocessor.classes.values),
            lstm_layers=lstm_layers,
            units=lstm_units,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout
        )(inputs)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer = "adam",
            loss = f"{'sparse_' if self.sparse else ''}categorical_crossentropy",
            metrics=metrics
        )


    def train(self, 
              epochs: int, 
              batch_size: int,

              es_monitor: str | None = "val_loss",
              es_patience: int | None = None,
              es_restore_best: bool = True,

              pl_unlabeled_data: np.ndarray | None = None,
              pl_interval: int = 10,
              pl_confidence_threshold: int = 32):
        training_data = DynamicDataset(self.training_data[0], self.training_data[1], batch_size)
        
        callbacks = []
        if es_monitor:
            callbacks.append(
                EarlyStopping(
                    monitor=es_monitor,
                    patience=es_patience or epochs // 10,
                    restore_best_weights=es_restore_best
                )
            )
        
        if pl_unlabeled_data is not None:
            callbacks.append(
                PseudoLabelingCallback(
                    dataset=training_data,
                    X_unlabeled=pl_unlabeled_data,
                    confidence_threshold=pl_confidence_threshold,
                    interval=pl_interval,
                    
                )
            )
        
        self.model.fit(
            training_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data=self.validation_data
        )

    def predict(self, X: np.ndarray):
        return self.model.predict(X)

