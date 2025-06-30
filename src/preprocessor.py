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
import zipfile
import tempfile

DATA_PATH = r"C:\Users\a_gerw500\Documents\agmge\log-classification\data"
LOG_PATH = DATA_PATH + "/CCI/CCLog-backup.{n}.log"
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
        if len(self.data_list) != 0:
            x, y = zip(*self.data_list)
            x = np.array(x)
            y = np.array(y)

            if not self.const:
                self.data_array_x, self.data_array_y = x, y
        if self.data_array_x is None or self.data_array_y is None:
            raise Exception("Data array is not defined.")

        return self.data_array_x, self.data_array_y
    
    def stratified_split(self, ratios=(4, 1), seed=42):
        if len(self.data_list) != 0:
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

        # Optionally shuffle the final splits
        result = []
        for split in splits:
            random.shuffle(split)
            X, y = zip(*split)
            X = np.array(X)
            y = np.array(y)
            result.append((X, y))

        return result
        
    def save(self, file_path: str):
        if len(self.data_list) != 0:
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
        npz = np.load(file_path)
        data_x, data_y = npz['x'], npz['y']
        npz.close()
        
        shape = data_x[0].shape
        
        if validate_shape and not all(d.shape == shape for d in data_x):
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
        self.loaded_files = set()

        self.data = Dataset((self.window_size, (11 if self.extended_datetime_features else 2) + 1 + 1 + self.message_encoder.get_result_shape()))
        self.events = []
        try:
            for i in log_numbers: self.load_logfile(LOG_PATH.format(n=i))
        except KeyboardInterrupt:
            pass
        self.message_encoder.initialize([ev["log_message"] for ev in self.events])
        self.function_encoder = LabelEncoder()
        self.function_encoder.fit([ev["function"] for ev in self.events])
        
    
    def initialize(self):
        self.message_encoder.initialize([ev["log_message"] for ev in self.events])
        self.function_encoder = LabelEncoder()
        self.function_encoder.fit([ev["function"] for ev in self.events])

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
    
    def preprocess(self):
        # initialize tracker for progress
        if self.volatile: progress = tqdm(total=len(self.events), desc="annotating events")

        # let a sliding window go over the events list
        for i in range(self.window_size, len(self.events)):
            # update progress
            if self.volatile: progress.update(1)
            
            # select the events by sliding window and get the last of the selected events
            window = self.events[(i-self.window_size):i]
            
            vec, label = self.annotate_and_encode_window(window)
            if self.data.states_counts[label] >= self.logs_per_class: continue
            else: self.data.add(vec, label)
        
        # log
        if self.volatile:
            print(f"State counts:")
            for k, v in self.data.states_counts.items(): print(f"  - {k} : {v}")

        # break out of the loop once all classes have the required number
        if all(v == self.logs_per_class for v in self.data.states_counts.values()):
            print(f"All states have the desired log count")
    
    def annotate_and_encode_window(self, window):
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
        


    def save(self, path: str | None = None):
        if path is None:
            path = DATA_PATH + f"/preprocessors/preprocessor_{len(self.loaded_files)}files_"
            m = "BERTenc" if isinstance(self.message_encoder, BERTEncoder) else \
                "BERTemb" if isinstance(self.message_encoder, BERTEmbeddingEncoder) else \
                "TextVec" if isinstance(self.message_encoder, TextVectorizationEncoder) else "enc"
            path += f"{self.logs_per_class}lpc_{self.window_size}ws_{m}x{self.message_encoder.get_result_shape() if self.message_encoder is not None else ''}"
            if self.extended_datetime_features:
                path += "_extdt"
            path += ".zip"

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Use a temporary directory to store intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            message_encoder_path = os.path.join(temp_dir, "message_encoder.pkl")
            function_encoder_path = os.path.join(temp_dir, "function_encoder.pkl")
            data_path = os.path.join(temp_dir, "data.npz")
            json_path = os.path.join(temp_dir, "metadata.json")

            # Save encoders
            if hasattr(self, "message_encoder"):
                joblib.dump(self.message_encoder, message_encoder_path)

            if hasattr(self, "function_encoder"):
                joblib.dump(self.function_encoder, function_encoder_path)

            if self.data is not None:
                self.data.save(data_path)

            # Build the JSON metadata
            obj = {
                "extended_datetime_features": self.extended_datetime_features,
                "logs_per_class": self.logs_per_class,
                "window_size": self.window_size,
                "message_encoder_path": "message_encoder.pkl",
                "function_encoder_path": "function_encoder.pkl",
                "data_path": "data.npz",
                "loaded_files": list(self.loaded_files),
                "events": self.events,
            }

            with open(json_path, 'w') as f:
                json.dump(obj, f, indent=4)

            # Zip everything
            with zipfile.ZipFile(path, 'w') as zf:
                zf.write(json_path, "metadata.json")
                if os.path.exists(message_encoder_path):
                    zf.write(message_encoder_path, "message_encoder.pkl")
                if os.path.exists(function_encoder_path):
                    zf.write(function_encoder_path, "function_encoder.pkl")
                if os.path.exists(data_path):
                    zf.write(data_path, "data.npz")

        return path

    @staticmethod
    def load(path: str, volatile: bool = True):
        if not zipfile.is_zipfile(path):
            raise ValueError(f"The provided path is not a zip file: {path}")

        with zipfile.ZipFile(path, 'r') as zip_file:
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_file.extractall(temp_dir)

                # Load JSON metadata
                metadata_path = os.path.join(temp_dir, "metadata.json")
                with open(metadata_path, 'r') as f:
                    json_obj = json.load(f)

                obj = object.__new__(Preprocessor)
                obj.volatile = volatile

                # Load encoders
                message_encoder_file = os.path.join(temp_dir, json_obj.get("message_encoder_path", "message_encoder.pkl"))
                if os.path.exists(message_encoder_file):
                    obj.message_encoder = joblib.load(message_encoder_file)

                function_encoder_file = os.path.join(temp_dir, json_obj.get("function_encoder_path", "function_encoder.pkl"))
                if os.path.exists(function_encoder_file):
                    obj.function_encoder = joblib.load(function_encoder_file)

                # Load data
                data_file = os.path.join(temp_dir, json_obj.get("data_path", "data.npz"))
                if os.path.exists(data_file):
                    obj.data = Dataset.load(data_file)
                else:
                    raise Exception("No data found")

                # Restore attributes
                obj.extended_datetime_features = json_obj.get("extended_datetime_features", False)
                obj.logs_per_class = json_obj.get("logs_per_class", 0)
                obj.window_size = json_obj.get("window_size", 1)
                obj.events = [dict(ev) for ev in json_obj.get("events", [])]
                obj.loaded_files = set(json_obj.get("loaded_files", []))

                obj.initialize()

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
    pp.preprocess()

    train, test = pp.data.stratified_split((4, 1))
    X_train, y_train = train
    X_test, y_test = test

    print(X_train)

