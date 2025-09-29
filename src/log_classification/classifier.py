from __future__ import annotations

import sys
import numpy as np
import seaborn as sns
import re
import json
import zipfile
import os
from typing import Tuple, List, overload
from keras.saving import save_model, load_model
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Input, Layer
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

from log_classification.preprocessor import Preprocessor
from log_classification.dataset import Dataset
from log_classification.pseudolabeling import DynamicDataset, PseudoLabelingCallback
from log_classification.lstm import LSTMClassifierLayer
from log_classification.transformer import TransformerBlock, TransformerClassifierLayer


class Classifier:
    """
    Classifier that wraps around a keras model, and let's you train and predict 
    """
    def __init__(self, 
                 preprocessor: Preprocessor, 
                 dataset: Dataset,

                 classifier_layer: Layer,

                 # train, validation, test, pseudolabel(unlabeled) split ratios
                 data_split_ratios: Tuple[int, int, int, int] = (4, 1, 1, 0), 
                 
                 learning_rate: float = 0.001,
                 metrics: List[str] = ["accuracy"],
                 seed: int | None = None
                 ):
        """
        Create a new Classifier object from a preprocessor, dataset and classification layer.
        The classifier_layer currently supports `LSTMClassifierLayer` and `TransformerClassifierLayer` 
        defined in lstm.py and transformer.py of this project

        :param preprocessor: a `Preprocessor` object used for preprocessing raw data if needed
        :param dataset: a `Dataset` object -> already preprocessed and labeled dataset
        :param classifier_layer: `LSTMClassifierLayer` or `TransformerClassifierLayer` as defined in lstm.py and transformer.py of this project
        :param data_split_ratios: a tuple of integers, defines the ratio of train, validation, test, pseudolabel(unlabeled) data in that order
        :param learning_rate: The learning rate used for the optimizer when training the model
        :param metrics: The metrics to be tracked when compiling the model, list of strings
        :param seed: An integer defining a seed for random for reproducability, None for no seed
        """
        self.pp = preprocessor
        self.dataset: Dataset | None = dataset

        self.history = None
        
        self.training_data = self.validation_data = self.test_data = self.pseudo_label_data = None
        if dataset is not None:
            self.training_data, self.validation_data, self.test_data, self.pseudo_label_data = dataset.stratified_split(data_split_ratios, seed)
        # print(f"{len(self.training_data[0]) = }")
        # print(f"{len(self.validation_data[0]) = }")
        # print(f"{len(self.test_data[0]) = }")
        # print(f"{len(self.pseudo_label_data[0]) = }")

        sparse = len(self.dataset.data_array_y[0]) == 1

        self.model = Sequential()
        self.model.add(Input(shape=dataset.entry_shape))
        
        self.model.add(classifier_layer)
        

        self.model.compile(
            optimizer = Adam(learning_rate=learning_rate),
            loss = f"{'sparse_' if sparse else ''}categorical_crossentropy",
            metrics=metrics
        )

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the test data that was split from the dataset

        :return: a tuple test_X, test_y
        """
        return self.test_data

    def train(self, 
              epochs: int, 
              batch_size: int = 32,

              es_monitor: str | None = None,
              es_patience: int | None = None,
              es_min_delta: float = 0.001,
              es_restore_best: bool = True,

              pl_unlabeled_data: np.ndarray | None = None,
              pl_interval: int = 10,
              pl_confidence_threshold: int = 32, 

              data_augmentation_functions: List[Tuple[float, callable]] | None = None):
        """
        Train the model.

        :param epochs: The maximum number of epochs to train the model for
        :param batch_size: Number of samples per gradient update. If unspecified, batch_size will default to 32

        :param es_monitor: EarlyStopping monitor - which Quantity should be monitored, if this is None no EarlyStopping will be applied
        :param es_patience: EarlyStopping patience - how many epochs without change are needed to trigger EarlyStopping, If None is specified it will use epochs/10
        :param es_min_delta: EarlyStopping min_delta - minimum change to be considered an improvement
        :param es_restore_best: EarlyStopping restore_best - whether to restore the best weights of the model after an EarlyStopping was triggered. Defaults to True

        :param pl_unlabeled_data: Additional unlabeled data used for pseudolabeling. Either this has to be defined or a split for pseudolabeling when initializing to enable Pseudolabeling
        :param pl_interval: The epoch interval of trying to add new data from pseudolabeling to the training data. 
        :param pl_confidence_threshold: How much confidence a prediction needs for it to be considered for a pseudolabel

        :param data_augmentation_functions: A list of data augmentation functions and probabilities to trigger them
        """
        training_data = DynamicDataset(self.training_data[0], self.training_data[1], batch_size, data_augmentation_functions)
        
        callbacks = []
        if es_monitor is not None:
            print(f"{es_monitor = }")
            callbacks.append(
                EarlyStopping(
                    monitor=es_monitor,
                    patience=es_patience or epochs // 10,
                    restore_best_weights=es_restore_best,
                    min_delta=es_min_delta,
                )
            )

        if pl_unlabeled_data is not None or len(self.pseudo_label_data[0]) > 0:
            if pl_unlabeled_data is None:
                d = self.pseudo_label_data[0]
            elif len(self.pseudo_label_data[0]) == 0:
                d = pl_unlabeled_data
            else:
                d = np.concatenate((self.pseudo_label_data[0], pl_unlabeled_data))

            callbacks.append(
                PseudoLabelingCallback(
                    dataset=training_data,
                    X_unlabeled=d,
                    confidence_threshold=pl_confidence_threshold,
                    interval=pl_interval,
                    
                )
            )
        
        self.history = self.model.fit(
            training_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data=self.validation_data,
        )

    @overload
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class of a preprocessed input

        :param X: a single or batch input of preprocessed data
        :returns: a predictions for the labels/classes of the input data
        """
        ...

    @overload
    def predict(self, log_file_path: str, lines_start: int, lines_end: int) -> np.ndarray:
        """
        Directly predict labels from a log file. 

        :param log_file_path: Path to the log file
        :param lines_start: First line index
        :param lines_end: Last line index (exclusive)
        :return: Predictions as a numpy array
        """
        ...

    def predict(self, *args, **kwargs) -> np.ndarray:
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            # Case 1: directly predict from numpy array
            X: np.ndarray = args[0]
            return self.model.predict(X)
        
        elif len(args) == 3 and isinstance(args[0], str):
            # Case 2: predict from log file range
            log_file_path, lines_start, lines_end = args
            x, lines = self.pp.preprocess_logfile_range(log_file_path, lines_start, lines_end)
            #print(x)
            return self.model.predict(x), lines
        
        else:
            raise TypeError("Invalid arguments to predict()")


    def evaluate(self, 
                 filename: str, 
                 additional_test_data: Tuple[np.ndarray, np.ndarray] | None = None, 
                 preview_count: int = 80):
        """
        Evaluates the model. To do that it

        - plots the accuracy, validation accuracy, loss and validation loss
        - writes the summary and `preview_count` predictions on the split test data into a text file
        - saves a confusion matrix on the split test data
        - if additional_test_data is passed the last two steps will be repeated for the additional data

        :param filename: the name under which the files will be saved. Depending on the file a suffix will be appended
        :param additional_test_data: optional additional test data, should be disjunct from the internal dataset
        :param preview_count: The amount of preview predictions to do
        """
        import matplotlib.pyplot as plt

        # Plot training & validation accuracy and loss
        plt.figure(figsize=(12, 5))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        #plt.tight_layout()
        plt.savefig(f"{filename}-training_graph.png")
        plt.show()

        with open(f"{filename}-summary.txt", "w") as f:
            ostd = sys.stdout
            sys.stdout = f

            # Summary of the model
            self.model.summary(print_fn = f.write)

            X_test, y_test = self.test_data
            # Evaluate the model on test data
            results = self.model.evaluate(X_test, y_test, verbose=0)
            for name, value in zip(self.model.metrics_names, results):
                print(f"{name}: {value:.4f}")

            # Predictions
            y_test_labels = y_test.squeeze()
            y_test_labels = y_test_labels.argmax(axis=1) if y_test_labels.ndim > 1 else y_test_labels
            predictions = self.model.predict(X_test)
            y_pred = np.argmax(predictions, axis=1)

            # Classification report and F1 score
            print("\nClassification Report:")
            print(classification_report(y_test_labels, y_pred))

            f1 = f1_score(y_test_labels, y_pred, average='weighted')
            print(f"F1 Score (weighted): {f1:.4f}")

            # === Preview Predictions ===
            for i in range(min(preview_count, len(y_pred))):
                print(f"True: {y_test_labels[i]}, Pred: {y_pred[i]}, Prob: {predictions[i]}")

            sys.stdout = ostd
    
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test_labels, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(f"{filename}-test_confusion_matrix.png")
        plt.show()



        if additional_test_data is not None:
            print(f"{additional_test_data = }")

            X_additional, y_additional = additional_test_data
            
            with open(f"{filename}-summary.txt", "a") as f:
                ostd = sys.stdout
                sys.stdout = f
            
                self.model.evaluate(X_additional, y_additional, verbose=0)

                # Get predictions
                y_pred_probs = self.model.predict(X_additional)
                y_pred = y_pred_probs.argmax(axis=1)  # For softmax outputs

                # True labels
                y_true = y_additional.argmax(axis=1) if y_additional.ndim > 1 else y_additional

                # Compute metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='weighted')
                recall = recall_score(y_true, y_pred, average='weighted')
                f1 = f1_score(y_true, y_pred, average='weighted')


                print("test with test data (3 newest files)")
                print(f"Accuracy : {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall   : {recall:.4f}")
                print(f"F1 Score : {f1:.4f}")

                sys.stdout = ostd
            
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues", xticks_rotation=45)
            plt.title("Confusion Matrix")
            plt.savefig(f"{filename}-test_confusion_matrix_additional.png")
            plt.show()
        return
    


    def save(self, path: str = "test.keras"):
        save_model(self.model, path, zipped=True)

        json_data = {
            "preprocessor_key": self.pp.get_key(),
            "preprocessor_origin": self.pp.origin_path or ""
        }

        pp_config_file = "preprocessor_config.json"
        with open(pp_config_file, "w") as f:
            json.dump(json_data, f)

        # Append new file
        with zipfile.ZipFile(path, 'a') as zipf:
            zipf.write(pp_config_file, arcname=pp_config_file)
        
        os.remove(pp_config_file)

    @classmethod
    def load(cls, path: str) -> Classifier:
        obj = cls.__new__(cls)
        obj.dataset = None
        
        pp_config_file = "preprocessor_config.json"
        with zipfile.ZipFile(path, 'r') as zipf:
            with zipf.open(pp_config_file) as json_file:
                pp_config =  json.load(json_file)

        pp_origin = pp_config["preprocessor_origin"]
        pp_key = pp_config["preprocessor_key"]

        # TODO: unpack preprocessor_config.json from path and load pp from the file relative to a project path configured in a config file? 
        obj.pp = Preprocessor.load(pp_origin)

        obj.history = None
        obj.training_data = obj.validation_data = obj.test_data = obj.pseudo_label_data = None
        obj.model = load_model(
            path, {
                "LSTMClassifierLayer": LSTMClassifierLayer, 
                "TransformerBlock": TransformerBlock, 
                "TransformerClassifierLayer": TransformerClassifierLayer
                })

        return obj

if __name__ == "__main__":
    from lstm import LSTMClassifierLayer
    
    pp = Preprocessor.load(r"preprocessors\[chbt2s4V18eW9fsI][20250923_151853]test_pp.json")
    # chbt2s4V18eW9fsI
    dtst = Dataset.load(r"data\datasets\[20250923_161907]dataset.npz")
    clsf = Classifier(pp, dtst, LSTMClassifierLayer(4), (4, 1, 1, 0))

    clsf.save()