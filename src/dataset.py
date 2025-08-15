from __future__ import annotations

from collections import defaultdict
import numpy as np
from hash_list import hash_ndarray
import random
from datetime import datetime
import json
from keras.utils import Sequence

class Dataset(Sequence):
    def __init__(self, entry_shape, preprocessor_key: str | None = None, loaded_logfiles: set[str] | None = None):
        self.data_list = [] # entries are (feature tensor, state) where feature tensor has the shape entr_shape
        
        self.preprocessor_key = preprocessor_key
        self.loaded_logfiles = loaded_logfiles
        
        self.data_array_x = None
        self.data_array_y = None
        
        self._const = False
        
        self.entry_shape = entry_shape
        self.class_counts = defaultdict(int)
    
    def add(self, features: np.ndarray, state):
        if not features.shape == self.entry_shape: 
            raise Exception(f"Shape mismatch: {features.shape} must be the same as {self.entry_shape}.")
        if self._const: raise Exception("Can't edit constant Dataset.")

        self.class_counts[hash_ndarray(state)] += 1
        self.data_list.append((features, state))
    
    def as_xy_arrays(self):
        # try to define data as array if Dataset is not const
        if not self._const and len(self.data_list) != 0:
            x, y = zip(*self.data_list)
            self.data_array_x = np.array(x)
            self.data_array_y = np.array(y)
        
        # raise if no data arrays
        if self.data_array_x is None or self.data_array_y is None:
            raise Exception("Data array is not and could not be defined.")

        # return data arrays
        return self.data_array_x, self.data_array_y
    
    def stratified_split(self, ratios = (4, 1), seed = None):
        # try to define data as array if Dataset is not const
        if not self._const and len(self.data_list) != 0:
            x, y = zip(*self.data_list)
            self.data_array_x = np.array(x)
            self.data_array_y = np.array(y)

        # raise if no data arrays
        if self.data_array_x is None or self.data_array_y is None:
            raise Exception("Data array is not and could not be defined.")
        
        # seed the random module and np.random module
        random.seed(seed)
        np.random.seed(seed)
        class_buckets = defaultdict(list)
        
        # Group samples by class
        for x, y in zip(self.data_array_x, self.data_array_y):
            class_buckets[hash_ndarray(y)].append((x, y))

        # Normalize ratios
        total = sum(ratios)
        normalized_ratios = [r / total for r in ratios]
        num_splits = len(ratios)
        splits = [[] for _ in range(num_splits)]

        # Perform stratified split
        for samples in class_buckets.values():
            random.shuffle(samples)
            n = len(samples)

            split_sizes = [int(n * r) for r in normalized_ratios]
            # Handle rounding by adjusting last split size
            split_sizes[-1] = n - sum(split_sizes[:-1])

            start = 0
            for i, size in enumerate(split_sizes):
                splits[i].extend(samples[start:start + size])
                start += size

        # Shuffle the final splits
        result = []
        for split in splits:
            random.shuffle(split)
            X, y = zip(*split)
            X = np.array(X)
            y = np.array(y)
            result.append((X, y))

        return result
        
    def save(self, file_path: str, save_meta: bool):
        if self._const: raise Exception("Can't save Dataset flagged as constant")
        if not file_path.endswith(".npz"): raise ValueError("file_path must be an .npz file")

        # try to define data as array if Dataset is not const
        if not self._const and len(self.data_list) != 0:
            x, y = zip(*self.data_list)
            self.data_array_x = np.array(x)
            self.data_array_y = np.array(y)

        # raise if no data arrays
        if self.data_array_x is None or self.data_array_y is None:
            raise Exception("Data array is not and could not be defined.")
        
        # format the file path
        dt = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = file_path.format(
            preprocessor_key=self.preprocessor_key,
            timestamp=dt,
            num_files=len(self.loaded_logfiles))
        # save the npz file
        np.savez(file_path, x=self.data_array_x, y=self.data_array_y, preprocessor_key=self.preprocessor_key)
        
        # save metadata
        if save_meta:
            meta = {
                "loaded_logfiles": list(self.loaded_logfiles),
                "preprocessor_key": self.preprocessor_key,
                "saved_at": dt
            }
            json_path = file_path.replace(".npz", ".json")
            with open(json_path, "w") as f:
                json.dump(meta, f, indent=4)

    @staticmethod
    def load(file_path: str, validate_shape: bool = True) -> Dataset:
        if not file_path.endswith(".npz"): raise ValueError("file_path must be an .npz file")
        
        # load npz file
        npz = np.load(file_path)
        data_x, data_y = npz['x'], npz['y']
        preprocessor_key = npz['preprocessor_key'].item()
        npz.close()
        
        # set and validate shape
        shape = data_x[0].shape
        if validate_shape and not all(d.shape == shape for d in data_x):
            raise Exception("feature shapes must be all the same")
        
        # create dataset
        ds = Dataset(shape, preprocessor_key)
        ds.data_array_x = data_x
        ds.data_array_y = data_y
        ds._const = True

        # set states counts
        for state in data_y:
            ds.class_counts[hash_ndarray(state)] += 1

        return ds



        self.X = np.concatenate([self.X, X_new])
        self.y = np.concatenate([self.y, y_new])