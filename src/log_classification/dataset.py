from __future__ import annotations
import numpy as np
import random
import json
from typing import List, Tuple
from collections import defaultdict
from datetime import datetime
from keras.utils import Sequence

from log_classification.util import hash_ndarray

class Dataset(Sequence):
    """
    Dataset containing x, y data as a list and as numpy arrays. Datasets also contain information about the 
    Preprocessor used to preprocess its data and information about the used log files.
    Datasets can be made constant for example after loading it from a file, disabeling adding new data.
    """
    def __init__(self, 
                 entry_shape: Tuple[int,...], 
                 preprocessor_key: str | None = None, 
                 loaded_logfiles: set[str] | None = None):
        """
        Creates a new Dataset. It contains the loaded data as a list of (x, y) tuples. 
        X and y can be accessed as `np.ndarray`s using `as_xy_arrays` and the `stratified_split` methods. 
        The dataset can be saved and loaded from/to files.

        :params entry_shape: the shape of the input data
        :params preprocessor_key: optional key of a preprocessor associated with this dataset
        :loaded_logfiles: a set containing all log files of which this dataset contains data
        """
        self.data_list = [] # entries are (feature tensor, state) where feature tensor has the shape entry_shape
        
        self.preprocessor_key = preprocessor_key
        self.loaded_logfiles = loaded_logfiles
        
        self.data_array_x = None
        self.data_array_y = None
        
        self._const = False
        
        self.entry_shape = entry_shape
        self.class_counts = defaultdict(int)
    
    def add(self, x: np.ndarray, y: np.ndarray):
        """
        Adds (x, y) to the dataset. Validates the shape before adding to the dataset.
        """
        if not x.shape == self.entry_shape: 
            raise Exception(f"Shape mismatch: {x.shape} must be the same as {self.entry_shape}.")
        if self._const: raise Exception("Can't edit constant Dataset.")

        self.class_counts[hash_ndarray(y)] += 1
        self.data_list.append((x, y))
    
    def as_xy_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        get the data in this dataset as two np.ndarrays
        """
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
    
    def stratified_split(self, ratios: Tuple[int,...], seed: int | None = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        splits the data into n parts where n is the length of ratios. Each of the n splits will have the proportion ratios[i]. An optional seed for the randomness can be specified
        
        :params ratios: a tuple of ratios to split the dataset into
        :params seed: an optional seed for the random selection of entries
        :returns: a list of tuples with two np.ndarrays each, the tuples are (x, y)
        """
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

        # --- validate & normalize ratios (zeros allowed) ---
        ratios = np.asarray(ratios, dtype=float)
        if np.any(ratios < 0):
            raise ValueError("Ratios must be non-negative.")
        total = ratios.sum()
        if total == 0:
            raise ValueError("At least one ratio must be > 0.")
        normalized = ratios / total

        num_splits = len(ratios)
        splits = [[] for _ in range(num_splits)]

        # --- stratified allocation with largest remainder; zero ratios stay zero ---
        for samples in class_buckets.values():
            random.shuffle(samples)
            n = len(samples)

            ideal = normalized * n                      # target (floats)
            base = np.floor(ideal).astype(int)         # floor sizes
            remainder = n - int(base.sum())            # leftover to assign

            if remainder > 0:
                frac = ideal - base
                # only splits with positive ratio are eligible for leftovers
                eligible = np.where(ratios > 0)[0]
                order = sorted(eligible, key=lambda i: frac[i], reverse=True)
                for i in order[:remainder]:
                    base[i] += 1

            # assign contiguous chunks according to final sizes
            start = 0
            for i, size in enumerate(base.tolist()):
                if size > 0:
                    splits[i].extend(samples[start:start + size])
                start += size

        # Shuffle and pack results; zero-ratio splits will be empty arrays
        result = []
        for split in splits:
            if split:
                random.shuffle(split)
                X, y = zip(*split)
                result.append((np.array(X), np.array(y)))
            else:
                result.append((np.array([]), np.array([])))

        return result

        
    def save(self, file_path: str, save_meta: bool):
        """
        saves the dataset to a .npz file, if `save_meta` is true, metadata will be saved to a .json file with the same name. 
        Metadata contains loaded_logfiles, preprocessor_key and saved_at. 

        :params file_path: the path to a .npz file to save the Dataset to (must end in .npz)
        :params save_meta: whether Metadata should be saved
        """
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
        
        return file_path

    @staticmethod
    def load(file_path: str, validate_shape: bool = True) -> Dataset:
        """
        Load a .npz file to create a new Dataset object. The Dataset will be constant

        :params file_path:
        :params validate_shape: whether to validate that all features have the same shape
        :returns: the loaded Dataset
        """
        
        if not file_path.endswith(".npz"): raise ValueError("file_path must be an .npz file")
        
        # load npz file
        with np.load(file_path) as npz:
            data_x, data_y = npz['x'], npz['y']
            preprocessor_key = npz['preprocessor_key'].item()
        
        
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