from __future__ import annotations

import re
try:
    get_ipython()  # Only defined in IPython/Jupyter environments
    from tqdm.notebook import tqdm
except (NameError, ImportError):
    from tqdm import tqdm
    
from collections import defaultdict
from states import States as S
from datetime import datetime
from util import *
import random
import numpy as np
import json
import os
import joblib
import zipfile
import tempfile

from encoders.datetime_encoder import DatetimeEncoder
from encoders.datetime_features import DatetimeFeature, DatetimeFeatureBase
from encoders.loglevel_encoder import *
from encoders.message_encoder import *
from encoders.function_encoder import *
from encoders.save_encoder import save_encoder_if_new
from hash_list import hash_list_to_string

DATA_PATH = r"D:\mgeo\projects\log-classification\data"
LOG_PATH = DATA_PATH + "/CCI/CCLog-backup.{n}.log"
LOG_LEVEL_MAP = {'Trace': 0, 'Debug': 1, 'Info': 2, 'Warn': 3, 'Error': 4, 'Fatal': 5}

class Dataset:
    def __init__(self, entry_shape, preprocessor_key: str | None = None, loaded_logfiles: set[str] | None = None):
        self.data_list = [] # entries are (feature tensor, state) where feature tensor has the shape entr_shape
        
        self.preprocessor_key = preprocessor_key
        self.loaded_logfiles = loaded_logfiles
        
        self.data_array_x = None
        self.data_array_y = None
        
        self._const = False
        
        self.entry_shape = entry_shape
        self.states_counts = defaultdict(int)
    
    def add(self, features: np.ndarray, state):
        if not features.shape == self.entry_shape: raise Exception("Shape must be the same as all other entries.")
        if self._const: raise Exception("Can't edit constant Dataset.")

        self.states_counts[state] += 1
        self.data_list.append((features, state))
    
    def as_xy_arrays(self):
        # try to define data as array if Dataset is not const
        if not self._const and len(self.data_list) != 0:
            x, y = zip(*self.data_list)
            x = np.array(x)
            y = np.array(y)
        
        # raise if no data arrays
        if self.data_array_x is None or self.data_array_y is None:
            raise Exception("Data array is not and could not be defined.")

        # return data arrays
        return self.data_array_x, self.data_array_y
    
    def stratified_split(self, ratios = (4, 1), seed = None):
        # try to define data as array if Dataset is not const
        if not self._const and len(self.data_list) != 0:
            x, y = zip(*self.data_list)
            x = np.array(x)
            y = np.array(y)

        # raise if no data arrays
        if self.data_array_x is None or self.data_array_y is None:
            raise Exception("Data array is not and could not be defined.")
        
        # seed the random module and np.random module
        random.seed(seed)
        np.random.seed(seed)
        class_buckets = defaultdict(list)
        
        # Group samples by class
        for x, y in zip(self.data_array_x, self.data_array_y):
            class_buckets[y].append((x, y))

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
        if len(self.data_list) != 0:
            x, y = zip(*self.data_list)
            x = np.array(x)
            y = np.array(y)
        
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
                json.dump(meta)

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
            ds.states_counts[state] += 1

        return ds


class Preprocessor:
    def __init__(self, 
                 message_encoder:  MessageEncoder,
                 function_encoder: FunctionEncoder,
                 datetime_encoder: DatetimeEncoder,
                 log_level_encoder: LogLevelEncoder,

                 window_size: int = 20,
                 name: str | None = None,

                 volatile: bool = False):
        
        self.name = name or "preprocessor"

        # whether to print progress and information while training
        self.volatile = volatile

        # set basic hyperparameters
        self.window_size = window_size
        # encoders
        self.message_encoder = message_encoder
        self.function_encoder = function_encoder
        self.datetime_encoder = datetime_encoder
        self.loglevel_encoder = log_level_encoder

        # initialize set of loaded files
        self.loaded_files = set()
        # load events and catch keyboard interrupts
        self.events = []

        if not self.message_encoder.initialized: self.message_encoder.initialize([ev["log_message"] for ev in self.events])
        if not self.function_encoder.initialized: self.function_encoder.initialize([ev["function"] for ev in self.events])
        if not self.loglevel_encoder.initialized: self.loglevel_encoder.initialize([ev["log_level"] for ev in self.events])

    def initialize(self):
        if not self.message_encoder.initialized: self.message_encoder.initialize([ev["log_message"] for ev in self.events])
        if not self.function_encoder.initialized: self.function_encoder.initialize([ev["function"] for ev in self.events])
        if not self.loglevel_encoder.initialized: self.loglevel_encoder.initialize([ev["log_leve"] for ev in self.events])

    def load_logfiles(self, log_numbers: list[int]):
        try:
            for i in log_numbers: self.load_logfile(LOG_PATH.format(n=i))
        except KeyboardInterrupt:
            pass

    def load_logfile(self, path: str):
        """
        Load and parse a logfile into a list of events. Each event has the following keys: "timestamp", "log_level", "function", "log_message"

        Args:
            path (str): The path to the log file.
        
        Returns:
            a list of event dictionaries
        """
        if path in self.loaded_files:
            return
        
        # open the log file
        with open(path, "r") as f:
            lines = f.readlines()

        # initialize tracker for progress
        filename = os.path.basename(path)
        if self.volatile: progress = tqdm(total=len(lines), desc=f"parsing log file {filename}")

        events = []

        # loop over all lines and parse them into a list of events
        while lines:
            # strip newlines
            line = lines[0].strip("\n")

            # if this is the start of a log entry, create a new event
            parts = line.split("|")
            if len(parts) >= 4 and re.search(r"^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{4}", parts[0]):
                event = {
                    "timestamp": parts[0].strip(),
                    "log_level": parts[1].strip(),
                    "function": parts[2].strip(),
                    "log_message": parts[3].strip()
                }
                events.append(event)

            # otherwise add to the previous log message
            elif events:
                events[-1]["log_message"] += "\n" + line
            
            # pop the first line
            lines.pop(0)

            # update progress
            if self.volatile: progress.update(1)
        
        self.events.extend(events)
        self.loaded_files.add(path)
    
    def preprocess_dataset(self, logs_per_class: int | None = 100, step: int = 1) -> Dataset:
        if len(self.events):
            raise ValueError("events must not be empty")        
        
        data = Dataset(
            (self.window_size, 
             self.datetime_encoder.get_dimension() 
             + self.loglevel_encoder.get_dimension() 
             + self.function_encoder.get_dimension() 
             + self.message_encoder.get_dimension()),
             self.get_key(),
             self.loaded_files)
        
        # initialize tracker for progress
        if self.volatile: progress = tqdm(total=len(self.events)//step, desc="annotating events")

        # let a sliding window go over the events list
        for i in range(self.window_size, len(self.events), step):
            if (logs_per_class is not None) and len(data.states_counts) != 0 and all(v == logs_per_class for v in data.states_counts.values()):
                break
            
            # update progress
            if self.volatile: progress.update(1)
            
            # select the events by sliding window and get the last of the selected events
            window = self.events[(i-self.window_size):i]
            
            vec, label = self.annotate_and_encode_window(window)
            if (logs_per_class is not None) and data.states_counts[label] >= logs_per_class: 
                continue
            else: data.add(vec, label)

            
        
        # log
        if self.volatile:
            print(f"State counts:")
            for k, v in data.states_counts.items(): print(f"  - {k} : {v}")

        if (logs_per_class is not None) and all(v == logs_per_class for v in data.states_counts.values()):
            print(f"All states have the desired log count")

        return data
    
    def annotate_and_encode_window(self, window):
        def event_to_vector(seq):
            sequence_features = []

            for ev in seq:
                dt = datetime.fromisoformat(ev["timestamp"])
                dt_feature = self.datetime_encoder.extract_date_time_features(dt)

                log_level_feature = self.loglevel_encoder.encode(ev['log_level'])

                function_feature = self.function_encoder.encode(ev['function'])

                log_msg_feature = self.message_encoder.encode(ev['log_message'])

                feature_vector = [val for val in dt_feature.values()] + [log_level_feature, function_feature, log_msg_feature]
                flat_feature = np.concatenate([to_flat_array(f) for f in feature_vector])
                sequence_features.append(flat_feature)

            return np.array(sequence_features)
        
        last_event = window[-1]

        ############################
        # Annotation rules
        ############################
        # Rule for UnobservedException
        if last_event["function"] == "C_line_Control_Server.CCServerAppContext.TaskSchedulerUnobservedTaskException":
            return event_to_vector(window), S.UnobservedException
        
        elif last_event["log_level"] == "Error":
            # Rule for DatabaseError
            if "DBProxyMySQL" in last_event["function"] or "DBManager" in last_event["function"]:
                return event_to_vector(window), S.DatabaseError
                
            # Rule for HliSessionError
            elif "SessionFactory.OpenSession" in last_event["function"]:
                return event_to_vector(window), S.HliSessionError
            
            else:
                return event_to_vector(window), S.Normal
                
        # Rule for Normal data
        else:
            return event_to_vector(window), S.Normal


    def preprocess_log_line(self, log_file, line_n):
        timestamp_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{4}")
    
        with open(log_file, 'r') as file:
            lines = file.readlines()

        # Ensure line_number is in range
        line_n = min(line_n, len(lines))
        start = None
        
        for i in range(line_n - 1, -1, -1):  # Go backward from the specified line
            if timestamp_pattern.search(lines[i]):
                start = i
                break
        
        if start is None:
            return None, None, None

        last_line = None
        events = []
        for i in range(start, len(lines)):
            line = lines[i].strip("\n")

            # if this is the start of a log entry, create a new event
            parts = line.split("|")
            if len(parts) >= 4 and re.search(r"^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{4}", parts[0]):
                if len(events) >= self.window_size:
                    last_line = i
                    break
                event = {
                    "timestamp": parts[0].strip(),
                    "log_level": parts[1].strip(),
                    "function": parts[2].strip(),
                    "log_message": parts[3].strip()
                }
                events.append(event)
            
            # otherwise add to the previous log message
            elif events:
                events[-1]["log_message"] += "\n" + line

        if len(events) != self.window_size:
            return None, None, None

        vec, label = self.annotate_and_encode_window(events)
        return vec, label, events, last_line
        
    def get_key(self):
        key = hash_list_to_string([
            self.message_encoder.get_key(),
            self.function_encoder.get_key(),
            self.loglevel_encoder.get_key(),
            self.datetime_encoder.get_key(),

            str(self.window_size)
            ], 16)
        
        return key

    def save(self, path: str):
        # get values for name formatting
        key = self.get_key()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # format preprocessor name and create directories
        preprocessor_path = path + f"/[{key}][{timestamp}]{self.name}.json"
        os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)

        paths = [preprocessor_path]
        # save the encoders and save their paths and keys in obj
        obj = {
            "window_size": self.window_size,
            "name": self.name
            }
        for encoder in [self.message_encoder, self.function_encoder, self.datetime_encoder, self.loglevel_encoder]:
            encoder_t, k, filename = save_encoder_if_new(encoder, path, timestamp)
            obj[f"{encoder_t}_k"] = k
            obj[f"{encoder_t}_f"] = filename
            paths.append(filename)

        # save the preprocessor json
        with open(preprocessor_path, 'w') as f:
            json.dump(obj, f, indent=4)

        return paths

    @staticmethod
    def load(preprocessor_path: str, 
             custom_encoder_directory: str | None = None,
             custom_encoder_paths: dict[str, str] | None = None,
             volatile: bool = True):
        with open(preprocessor_path, 'r') as f:
            json_obj = json.load(f)
        
        name = json_obj["name"]
        window_size = json_obj["window_size"]
        encoders = {}

        for encoder_t in ["datetime", "loglevel", "function", "message"]:
            path = None
            if custom_encoder_paths is not None and encoder_t in custom_encoder_paths:
                path = custom_encoder_paths[encoder_t]
            elif custom_encoder_directory is not None:
                path = os.path.join(custom_encoder_directory, json_obj[f"{encoder_t}_f"])
            else:
                path = os.path.join(os.path.dirname(preprocessor_path), json_obj[f"{encoder_t}_f"])

            # TODO: loading encoders logic
            encoders[encoder_t] = ...

        obj = Preprocessor(
            message_encoder=encoders["message"],
            datetime_encoder=encoders["datetime"],
            loglevel_encoder=encoders["loglevel"],
            function_encoder=encoders["function"],
            window_size=window_size,
            name=name,
            volatile=volatile
        )

        return obj


if __name__ == "__main__":
    if False:
        pp = Preprocessor.load(f"{DATA_PATH}/preprocessors/preprocessor_2files_11lpc_5ws_BERTencx8.zip")
        # pp.preprocess()


        train, test = pp.data.stratified_split((4, 1))
        X_train, y_train = train
        X_test, y_test = test

        print(X_train)
        
        quit()


    # preprocessing
    log_files = [i for i in range(745, 747)]            # list of ints representing the numbers of log files to use
    logs_per_class = 11                                 # How many datapoints per class should be collected if available
    window_size = 5                                     # how many log messages to be considered in a single data point from sliding window
    encoding_output_size = 8                            # size to be passed to the message_encoder, note that this is not neccessairily the shape of the output
    message_encoder = BERTEncoder(encoding_output_size) # the message_encoder to be used. Can be TextVectorizationEncoder (uses keras.layers.TextVectorizer), BERTEncoder (only uses the BERT tokenizer) or BERTEmbeddingEncoder (also uses the BERT model)
    test_ratio = 0.2                                    # percantage of the collected data that should be used for testing rather than training
    extended_datetime_features = False                  # bool, whether the preprocessing should use a multitude of normalized features extracted from the date 

    pp = Preprocessor(log_files, message_encoder, logs_per_class, window_size, extended_datetime_features, True)
    pp.preprocess_dataset()

    train, test = pp.data.stratified_split((4, 1))
    X_train, y_train = train
    X_test, y_test = test

    print(X_train)

