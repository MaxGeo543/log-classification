from keras.utils import Sequence
from keras.callbacks import Callback
import numpy as np


class DynamicDataset(Sequence):
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int):
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

    def add_data(self, X_new: np.ndarray, y_new: np.ndarray):
        self.X = np.concatenate([self.X, X_new])
        self.y = np.concatenate([self.y, y_new])


class PseudoLabelingCallback(Callback):
    def __init__(self, dataset: DynamicDataset, X_unlabeled: np.ndarray, confidence_threshold: float = 0.95, interval: int = 5):
        self.dataset: DynamicDataset = dataset
        self.X_unlabeled: np.ndarray = X_unlabeled
        self.threshold: float = confidence_threshold
        self.interval: int = interval

    def on_epoch_end(self, epoch: int, logs=None):
        if (epoch + 1) % self.interval != 0 or len(self.X_unlabeled) == 0:
            # Nothing to do if epoch is not multiple of interval or no unlabeled data
            return

        print(f"\n[INFO] Epoch {epoch+1}: Pseudo-labeling unlabeled data...")

        # Predict
        probs = self.model.predict(self.X_unlabeled, verbose=0)
        pseudo_labels = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        print("best confidence:", max(confidences))

        # Filter by confidence
        confident_idx = np.where(confidences >= self.threshold)[0]
        if len(confident_idx) == 0:
            print("[INFO] No confident pseudo-labels found.")
            return

        X_confident = self.X_unlabeled[confident_idx]
        y_confident = pseudo_labels[confident_idx]

        print(f"[INFO] Adding {len(confident_idx)} pseudo-labeled samples to training set.")

        # Add to training data
        self.dataset.add_data(X_confident, y_confident)

        # Remove from unlabeled pool
        self.X_unlabeled = np.delete(self.X_unlabeled, confident_idx, axis=0)