from __future__ import annotations

import re
try:
    get_ipython()  # Only defined in IPython/Jupyter environments
    from tqdm.notebook import tqdm
except (NameError, ImportError):
    from tqdm import tqdm
    
from collections import defaultdict
from states import States as S
import datetime
from sklearn.preprocessing import LabelEncoder
from util import *
import random
import numpy as np
from message_encoder import *
import json
import os
import joblib

LOG_PATH = "C:/Users/Askion/Documents/agmge/log-classification/data/CCI/CCLog-backup.{n}.log"
LOG_LEVEL_MAP = {'Trace': 0, 'Debug': 1, 'Info': 2, 'Warn': 3, 'Error': 4, 'Fatal': 5}

class Dataset:
    def __init__(self, entry_shape):
        self.data_list = [] # entries are (feature tensor, state) where feature tensor has the shape entr_shape
        
        self.data_array_x = None 
        self.data_array_y = None 
        self.const = False
        
        self.entry_shape = entry_shape
        self.states_counts = defaultdict(int)
    
    def add(self, features: np.ndarray, state):
        if not features.shape == self.entry_shape:
            raise Exception("Shape must be the same as all other entries.")
        if self.const:
            raise Exception("Can't edit constant Dataset.")

        self.states_counts[state] += 1
        self.data_list.append((features, state))
    
    def as_xy_arrays(self):
        x, y = zip(*self.data_list)
        x = np.array(x)
        y = np.array(y)

        if not self.const:
            self.data_array_x, self.data_array_y = x, y
        if self.data_array_x is None or self.data_array_y is None:
            raise Exception("Data array is not defined.")

        return self.data_array_x, self.data_array_y
    
    def stratified_split(self, ratios=(4, 1), seed=42):
        x, y = zip(*self.data_list)
        x = np.array(x)
        y = np.array(y)

        if not self.const:
            self.data_array_x, self.data_array_y = x, y
        if self.data_array_x is None or self.data_array_y is None:
            raise Exception("Data array is not defined.")
        
        random.seed(seed)
        np.random.seed(seed)
        class_buckets = defaultdict(list)
        
        # Group samples by class
        for sample in zip(self.data_array_x, self.data_array_y):
            x, y = sample
            class_buckets[y].append(sample)

        # Normalize ratios
        total = sum(ratios)
        normalized_ratios = [r / total for r in ratios]
        num_splits = len(ratios)
        splits = [[] for _ in range(num_splits)]

        # Perform stratified split
        for samples in class_buckets.values():
            samples = list(samples)
            random.shuffle(samples)
            n = len(samples)

            split_sizes = [int(n * r) for r in normalized_ratios]
            # Handle rounding by adjusting last split size
            split_sizes[-1] = n - sum(split_sizes[:-1])

            start = 0
            for i, size in enumerate(split_sizes):
                splits[i].extend(samples[start:start + size])
                start += size

        # Optionally shuffle the final splits
        for split in splits:
            random.shuffle(split)

        return tuple(np.array(split) for split in splits)
        
    def save(self, file_path: str):
        x, y = zip(*self.data_list)
        x = np.array(x)
        y = np.array(y)

        if not self.const:
            self.data_array_x, self.data_array_y = x, y
        if self.data_array_x is None or self.data_array_y is None:
            raise Exception("Data array is not defined.")
        
        np.savez(file_path, x=self.data_array_x, y=self.data_array_y)

    @staticmethod
    def load(file_path: str, validate_shape: bool = True) -> Dataset:
        data = np.load(file_path)
        data_x, data_y = data['x'], data['y']
        shape = data[0][0].shape
        
        if validate_shape and not all(d[0].shape == shape for d in data_x):
            raise Exception("feature shapes must be all the same")
        
        ds = Dataset(shape)
        ds.data_array_x = data_x
        ds.data_array_y = data_y
        ds.const = True

        for state in data_y:
            ds.states_counts[state] += 1

        return ds


class Preprocessor:
    def __init__(self, 
                 log_numbers: list[int], 
                 message_encoder:  MessageEncoder | None,
                 logs_per_class: int = 100,
                 window_size: int = 20,
                 extended_datetime_features: bool = False,
                 volatile: bool = False):
        self.volatile = volatile

        self.extended_datetime_features = extended_datetime_features
        self.logs_per_class = logs_per_class
        self.window_size = window_size
        self.message_encoder = message_encoder

        self.data = Dataset((self.window_size, (11 if self.extended_datetime_features else 2) + 1 + 1 + self.message_encoder.get_result_shape()))
        self.events = []
        for i in log_numbers: self._load_logfile(LOG_PATH.format(n=i))
        self.message_encoder.initialize([ev["log_message"] for ev in self.events])
        self.function_encoder = LabelEncoder()
        self.function_encoder.fit([ev["function"] for ev in self.events])
        

    def _load_logfile(self, path: str):
        """
        Load and parse a logfile into a list of events. Each event has the following keys: "timestamp", "log_level", "function", "log_message"

        Args:
            path (str): The path to the log file.
        
        Returns:
            a list of event dictionaries
        """
        # open the log file
        with open(path, "r") as f:
            lines = f.readlines()

        # initialize tracker for progress
        if self.volatile: progress = tqdm(total=len(lines), desc=f"parsing log file {path}")

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
                self.events.append(event)

            # otherwise add to the previous log message
            elif self.events:
                self.events[-1]["log_message"] += "\n" + line
            
            # pop the first line
            lines.pop(0)

            # update progress
            if self.volatile: progress.update(1)
        
    
    def preprocess(self):
        def event_to_vector(seq):
            sequence_features = []

            for ev in seq:
                dt = datetime.datetime.fromisoformat(ev["timestamp"])
                dt = extract_date_time_features(dt)
                log_level = LOG_LEVEL_MAP.get(ev['log_level'], 0)
                function_id = self.function_encoder.transform([ev['function']])[0]
                log_msg_token = self.message_encoder.encode(ev['log_message'])
                # log_msg_token_id = log_msg_token[0] if log_msg_token else 0

                feature_vector = [val for val in dt.values()] + [log_level, function_id, log_msg_token]
                flat_feature = np.concatenate([to_flat_array(f) for f in feature_vector])
                sequence_features.append(flat_feature)

            return np.array(sequence_features)
        
        def extract_date_time_features(dt: datetime.datetime, normalize: bool = False):
            if not self.extended_datetime_features:
                # date as days since epoch
                date_feature = (dt.date() - datetime.datetime(1970, 1, 1).date()).days
                # Time in seconds since midnight
                time_feature = dt.hour * 3600 + dt.minute * 60 + dt.second

                return {"days_since_epoch": date_feature, "seconds_since_midnight": time_feature}

            features = {}

            # Date-based features
            epoch = datetime.datetime(1970, 1, 1)
            days_since_epoch = (dt.date() - epoch.date()).days
            features['days_since_epoch'] = days_since_epoch

            features['year'] = dt.year
            features['month'] = dt.month        # 1–12
            features['day'] = dt.day            # 1–31
            features['weekday'] = dt.weekday()  # 0 = Monday, 6 = Sunday
            features['is_weekend'] = int(dt.weekday() >= 5)

            # Time-based features
            hour = dt.hour + dt.minute / 60 + dt.second / 3600
            features['hour'] = dt.hour
            features['minute'] = dt.minute
            features['second'] = dt.second

            # Cyclical time features
            features['sin_hour'] = np.sin(2 * np.pi * hour / 24)
            features['cos_hour'] = np.cos(2 * np.pi * hour / 24)

            # Normalize numeric features (rough scaling — adjust as needed)
            max_values = {
                'days_since_epoch': 20000,  # ~55 years from 1970 to 2025
                'year': 2100,
                'month': 12,
                'day': 31,
                'weekday': 6,
                'hour': 23,
                'minute': 59,
                'second': 59
            }

            for key in list(features):
                if key in max_values:
                    features[key] = features[key] / max_values[key]

            return features

        # initialize tracker for progress
        if self.volatile: progress = tqdm(total=len(self.events), desc="annotating events")

        # let a sliding window go over the events list
        for i in range(self.window_size, len(self.events)):
            # update progress
            if self.volatile: progress.update(1)
            
            # select the events by sliding window and get the last of the selected events
            seq = self.events[(i-self.window_size):i]
            last_event = seq[-1]
            
            ############################
            # Annotation rules
            ############################
            # Rule for UnobservedException
            if last_event["function"] == "C_line_Control_Server.CCServerAppContext.TaskSchedulerUnobservedTaskException":
                if self.data.states_counts[S.UnobservedException] >= self.logs_per_class: continue
                
                self.data.add(event_to_vector(seq), S.UnobservedException)
                continue
            
            elif last_event["log_level"] == "Error":
                # Rule for DatabaseError
                if "DBProxyMySQL" in last_event["function"] or "DBManager" in last_event["function"]:
                    if self.data.states_counts[S.DatabaseError] >= self.logs_per_class: continue
                    
                    self.data.add(event_to_vector(seq), S.DatabaseError)
                    continue
                # Rule for HliSessionError
                elif "SessionFactory.OpenSession" in last_event["function"]:
                    if self.data.states_counts[S.HliSessionError] >= self.logs_per_class: continue
                    
                    self.data.add(event_to_vector(seq), S.HliSessionError)
                    continue
            # Rule for Normal data
            else:
                if self.data.states_counts[S.Normal] >= self.logs_per_class: continue

                self.data.add(event_to_vector(seq), S.Normal)
                continue
        
        # log
        if self.volatile:
            print(f"State counts:")
            for k, v in self.data.states_counts.items(): print(f"  - {k} : {v}")

        # break out of the loop once all classes have the required number
        if all(v == logs_per_class for v in self.data.states_counts.values()):
            print(f"All states have the desired log count")
        

    def save(self, path: str):
        base_path, _ = os.path.splitext(path)

        # Paths to encoders (you could customize extensions here if needed)
        message_encoder_path = base_path + "_message_encoder.pkl"
        function_encoder_path = base_path + "_function_encoder.pkl"
        data_path = base_path + "_data.npz"
        
        # Save encoders if they exist
        if hasattr(self, "message_encoder"):
            joblib.dump(self.message_encoder, message_encoder_path)

        if hasattr(self, "function_encoder"):
            joblib.dump(self.function_encoder, function_encoder_path)
        
        if self.data is not None:
            self.data.save(data_path)

        obj = {
            "extended_datetime_features": self.extended_datetime_features,
            "logs_per_class": self.logs_per_class,
            "window_size": self.window_size,
            "message_encoder_path": message_encoder_path,
            "function_encoder_path": function_encoder_path,
            "data_path": data_path,
            "events": []
        }

        for ev in self.events:
            obj["events"].append(ev)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save as JSON
        with open(path, 'w') as f:
            json.dump(obj, f, indent=4)

    @staticmethod
    def load( path: str):
        base_path, _ = os.path.splitext(path)

        # Load JSON metadata
        with open(path, 'r') as f:
            json_obj = json.load(f)

        obj = object.__new__(Preprocessor)

        # Load encoders if paths are specified
        message_encoder_path = json_obj.get("message_encoder_path", base_path + "_message_encoder.pkl")
        if os.path.exists(message_encoder_path):
            obj.message_encoder = joblib.load(message_encoder_path)

        function_encoder_path = json_obj.get("function_encoder_path", base_path + "_function_encoder.pkl")
        if os.path.exists(function_encoder_path):
            obj.function_encoder = joblib.load(function_encoder_path)

        # Load data if it exists
        data_path = json_obj.get("data_path", base_path + "_data.npy")
        if os.path.exists(data_path):
            obj.data = np.load(data_path, allow_pickle=True)
        else:
            obj.data = None

        # Restore other attributes
        obj.extended_datetime_features = json_obj.get("extended_datetime_features", False)
        obj.logs_per_class = json_obj.get("logs_per_class", 0)
        obj.window_size = json_obj.get("window_size", 1)
        obj.events = [tuple(ev) for ev in json_obj.get("events", [])]

        return obj

if __name__ == "__main__":
    # preprocessing
    log_files = [i for i in range(745, 747)]            # list of ints representing the numbers of log files to use
    logs_per_class = 10                                # How many datapoints per class should be collected if available
    window_size = 5                                    # how many log messages to be considered in a single data point from sliding window
    encoding_output_size = 8                           # size to be passed to the message_encoder, note that this is not neccessairily the shape of the output
    message_encoder = BERTEncoder(encoding_output_size) # the message_encoder to be used. Can be TextVectorizationEncoder (uses keras.layers.TextVectorizer), BERTEncoder (only uses the BERT tokenizer) or BERTEmbeddingEncoder (also uses the BERT model)
    test_ratio = 0.2                                    # percantage of the collected data that should be used for testing rather than training
    extended_datetime_features = False                  # bool, whether the preprocessing should use a multitude of normalized features extracted from the date 

    pp = Preprocessor(log_files, message_encoder, logs_per_class, window_size, extended_datetime_features, True)
    pp.preprocess()


    pp.save("./test_pp.json")