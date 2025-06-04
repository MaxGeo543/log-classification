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
from message_encoder import *
import numpy as np

LOG_PATH = "C:/Users/Askion/Documents/agmge/log-classification/data/CCI/CCLog-backup.{n}.log"
LOG_LEVEL_MAP = {'Trace': 0, 'Debug': 1, 'Info': 2, 'Warn': 3, 'Error': 4, 'Fatal': 5}







class Preprocessor:
    def __init__(self, 
                 log_numbers: list[int], 
                 message_encoder:  MessageEncoder,
                 logs_per_class: int = 100,
                 window_size: int = 20,
                 extended_datetime_features: bool = False,
                 volatile: bool = False):
        self.volatile = volatile

        self.extended_datetime_features = extended_datetime_features

        self.annotated = []
        self.states_counts = defaultdict(int)
        self.logs_per_class = logs_per_class
        self.window_size = window_size
        self.message_encoder = message_encoder

        
        for n in log_numbers:
            file_path = LOG_PATH.format(n=n)

            # log
            if self.volatile: print(f"Current log file: {file_path}")
            
            # load and annotate data
            events = self.load_logfile(file_path)
            self.annotate_data(events)
            
            # log
            if self.volatile:
                print(f"Current State counts:")
                for k, v in self.states_counts.items(): print(f"  - {k} : {v}")

            # break out of the loop once all classes have the required number
            if all(v == logs_per_class for v in self.states_counts.values()):
                print(f"All states have the desired log count")
                break
    
    def load_logfile(self, path: str) -> list[dict]:
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
        if self.volatile: progress = tqdm(total=len(lines), desc="parsing log file")

        # loop over all lines and parse them into a list of events
        events = []
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
        
        # return the list of events
        return events

    def annotate_data(self, events: list[dict]):
        """
        annotates data by adding a "state" key to every event.
        Data is categorized into States defined in states.py based on simple pattern matching.

        Args:
            events (list): list of events as returned by load_logfile.
            progress (tqdm): optional tqdm object for tracking progress
        
        Returns:
            the list of events
        """
        # initialize tracker for progress
        if self.volatile: progress = tqdm(total=len(events), desc="annotating events")

        # let a sliding window go over the events list
        for i in range(self.window_size, len(events)):
            # update progress
            if self.volatile: progress.update(1)
            
            # select the events by sliding window and get the last of the selected events
            seq = events[(i-self.window_size):i]
            last_event = seq[-1]
            
            ############################
            # Annotation rules
            ############################
            # Rule for UnobservedException
            if last_event["function"] == "C_line_Control_Server.CCServerAppContext.TaskSchedulerUnobservedTaskException":
                if self.states_counts[S.UnobservedException] >= self.logs_per_class: continue
                
                self.annotated.append((seq, S.UnobservedException))
                self.states_counts[S.UnobservedException] += 1
                continue
            
            elif last_event["log_level"] == "Error":
                # Rule for DatabaseError
                if "DBProxyMySQL" in last_event["function"] or "DBManager" in last_event["function"]:
                    if self.states_counts[S.DatabaseError] >= self.logs_per_class: continue
                    
                    self.annotated.append((seq, S.DatabaseError))
                    self.states_counts[S.DatabaseError] += 1
                    continue
                # Rule for HliSessionError
                elif "SessionFactory.OpenSession" in last_event["function"]:
                    if self.states_counts[S.HliSessionError] >= self.logs_per_class: continue
                    
                    self.annotated.append((seq, S.HliSessionError))
                    self.states_counts[S.HliSessionError] += 1
                    continue
            # Rule for Normal data
            else:
                if self.states_counts[S.Normal] >= self.logs_per_class: continue
                    
                self.annotated.append((seq, S.Normal))
                self.states_counts[S.Normal] += 1
                continue

    def pre_process(self):
        processed = []
        self.message_encoder.initialize([event['log_message'] for seq in self.annotated for event in seq[0]])
        function_encoder = self.get_function_encoder()

        for ev_seq, state in self.annotated:
            sequence_features = []

            for ev in ev_seq:
                dt = datetime.datetime.fromisoformat(ev["timestamp"])
                dt = self.extract_date_time_features(dt)
                log_level = LOG_LEVEL_MAP.get(ev['log_level'], 0)
                function_id = function_encoder.transform([ev['function']])[0]
                log_msg_token = self.message_encoder.encode([ev['log_message']])
                # log_msg_token_id = log_msg_token[0] if log_msg_token else 0

                feature_vector = [val for val in dt.values()] + [log_level, function_id, log_msg_token]
                flat_feature = np.concatenate([to_flat_array(f) for f in feature_vector])
                sequence_features.append(flat_feature)

            processed.append((np.array(sequence_features), state))

        return processed

    """
    def extract_date_time_features(self, dt: datetime.datetime):
        # date as days since epoch
        date_feature = (dt.date() - datetime.datetime(1970, 1, 1).date()).days
        # Time in seconds since midnight
        time_feature = dt.hour * 3600 + dt.minute * 60 + dt.second

        return date_feature, time_feature
    """

    def extract_date_time_features(self, dt: datetime.datetime, normalize: bool = False):
        """
        Extracts structured features from a datetime object.

        Args:
            dt (datetime): The datetime to process.
            normalize (bool): Whether to normalize the output features.

        Returns:
            dict: A dictionary of date/time features.
        """

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


    def get_function_encoder(self):
        all_functions = [event['function'] for seq in self.annotated for event in seq[0]]

        function_encoder = LabelEncoder()
        function_encoder.fit(all_functions)

        return function_encoder

    def stratified_split(self, test_ratio=0.2, seed=42):
        random.seed(seed)
        class_buckets = defaultdict(list)

        data = self.pre_process()

        # Group samples by class
        for x, y in data:
            class_buckets[y].append((x, y))

        train_data, test_data = [], []

        for class_label, samples in class_buckets.items():
            random.shuffle(samples)
            split_idx = int(len(samples) * (1 - test_ratio))
            train_data.extend(samples[:split_idx])
            test_data.extend(samples[split_idx:])

        # Optionally shuffle the final datasets
        random.shuffle(train_data)
        random.shuffle(test_data)

        return train_data, test_data

    def get_shape(self):
        # datetime_features (2 or 11) + log_level + function_id + log_message
        return self.window_size, (11 if self.extended_datetime_features else 2) + 1 + 1 + self.message_encoder.get_result_shape()
