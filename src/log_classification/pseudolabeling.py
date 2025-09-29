import numpy as np

from typing import List, Tuple, Callable
from keras.utils import Sequence
from keras.callbacks import Callback



class DynamicDataset(Sequence):
    """
    A dynamic dataset that can change during training for the sake of pseudo labeling or data augmentation.
    """
    def __init__(self, 
                 X: np.ndarray, 
                 y: np.ndarray, 
                 batch_size: int,
                 augmentation_functions: List[Tuple[float, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]] | None = None):
        """
        Create a new DynamicDataset with np.ndarrays of X and y (data and labels), a batch size and an optional list of augmentation functions. 
        Augmentation functions must take four np.ndarrays: the single object and label (x, y) of the data to augment and the whole collection of all data (X, y) 
        
        :params X: np.ndarray of input data
        :params y: np.ndarray of labels
        :params batch_size:
        :params augmentation_functions: List of tuples with probabilities (float) and callables, that are used to augment data
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augmentation_functions = augmentation_functions or []

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def _augment(self, x, y):
        """
        Goes through all augmentation functions and augments the data.
        """
        for prob, func in self.augmentation_functions:
            if np.random.rand() < prob:
                x, y = func(x, y, self.X, self.y)
        return x, y

    def __getitem__(self, idx):
        batch_x = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.augmentation_functions:
            for i in range(batch_x.shape[0]):
                batch_x[i], batch_y[i] = self._augment(batch_x[i], batch_y[i])

        return batch_x, batch_y

    def add_data(self, X_new: np.ndarray, y_new: np.ndarray):
        """
        Add additional data to the dataset
        """
        self.X = np.concatenate([self.X, X_new])
        self.y = np.concatenate([self.y, y_new])


class PseudoLabelingCallback(Callback):
    def __init__(self, 
                 dataset: DynamicDataset, 
                 X_unlabeled: np.ndarray, 
                 confidence_threshold: float = 0.95, 
                 interval: int = 5, 
                 verbose: bool = True):
        """
        keras.Callback that adds data with pseudolabels to a DynamicDataset based on the current state of the model and a given confidence threshold. 
        A pool of unlabeled data must be provided. The confidence threshold is the minimum confidence needed to add the data to the dataset. 
        The interval defines after how many epochs this Callback will be triggered.

        :params dataset: The dynamic dataset containg labeled data and allowing to add new data
        :params X_unlabeled: Collection of unlabeled data to use for pseudolabeling
        :params confidence_threshold: The minimum confidence needed to add new data to the dataset
        :params interval: Interval of epochs for how often this Callback will be triggered
        :params verbose: Whether to print inforamtion to the console.
        """
        self.dataset: DynamicDataset = dataset
        self.X_unlabeled: np.ndarray = X_unlabeled
        self.threshold: float = confidence_threshold
        self.interval: int = interval
        self.verbose: bool = verbose

    def on_epoch_end(self, epoch: int, logs=None):
        if (epoch + 1) % self.interval != 0 or len(self.X_unlabeled) == 0:
            # Nothing to do if epoch is not multiple of interval or no unlabeled data
            return

        if self.verbose: print(f"\n[INFO] Epoch {epoch+1}: Pseudo-labeling unlabeled data...")

        # Predict
        probs = self.model.predict(self.X_unlabeled, verbose=0)
        pseudo_labels = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        if self.verbose: print("best confidence:", max(confidences))

        # Filter by confidence
        confident_idx = np.where(confidences >= self.threshold)[0]
        if len(confident_idx) == 0:
            if self.verbose: print("[INFO] No confident pseudo-labels found.")
            return

        X_confident = self.X_unlabeled[confident_idx]
        y_confident = pseudo_labels[confident_idx]

        if self.verbose: print(f"[INFO] Adding {len(confident_idx)} pseudo-labeled samples to training set.")

        # Add to training data
        self.dataset.add_data(X_confident, y_confident)

        # Remove from unlabeled pool
        self.X_unlabeled = np.delete(self.X_unlabeled, confident_idx, axis=0)