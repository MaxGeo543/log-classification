from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Callable
from datetime import datetime
import random
import numpy as np
import json
import os
import joblib

from encoders.datetime_encoder import DatetimeEncoder
from encoders.datetime_features import DatetimeFeature, DatetimeFeatureBase
from encoders.loglevel_encoder import *
from encoders.message_encoder import *
from encoders.function_encoder import *
from encoders.classes_encoder import *
from encoders.save_encoder import save_encoder_if_new
from classes import Classes, classes, annotate as cl_annotate
from hash_list import hash_list_to_string, hash_ndarray
from encoders.encoder_type import EncoderType
from util import *


from rich import print
from rich.progress import (
    Progress,
    BarColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    SpinnerColumn,
    TextColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
)



class Dataset:
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
        
        print(self.data_list)

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


class Preprocessor:
    def __init__(self, 
                 message_encoder:  MessageEncoder,
                 function_encoder: FunctionEncoder,
                 datetime_encoder: DatetimeEncoder,
                 log_level_encoder: LogLevelEncoder,
                 classes_encoder: ClassesEncoder,

                 classes: Classes,
                 
                 window_size: int = 20,
                 name: str | None = None,

                 volatile: bool = False):
        
        self.name = name or "preprocessor"

        # whether to print progress and information while training
        self.volatile = volatile

        self.classes = classes

        # set basic hyperparameters
        self.window_size = window_size
        # encoders
        self.message_encoder = message_encoder
        self.function_encoder = function_encoder
        self.datetime_encoder = datetime_encoder
        self.loglevel_encoder = log_level_encoder
        self.classes_encoder = classes_encoder

        # initialize set of loaded files
        self.loaded_files = set()
        # load events and catch keyboard interrupts
        self.events = []

        self.encoders_initialized = False

    def initialize_encoders(self):
        if len(self.events) == 0:
            raise Exception("Events must be loaded before initializing the encoders")
        
        if not self.message_encoder.initialized: self.message_encoder.initialize([ev["log_message"] for ev in self.events])
        if not self.function_encoder.initialized: self.function_encoder.initialize([ev["function"] for ev in self.events])
        if not self.loglevel_encoder.initialized: self.loglevel_encoder.initialize([ev["log_level"] for ev in self.events])
        if not self.classes_encoder.initialized: self.classes_encoder.initialize(self.classes.values)

        self.encoders_initialized = True

    def load_logfiles(self, log_files: list[str]):
        len(log_files)
        try:
            for i, f in enumerate(log_files): 
                self.load_logfile(f, num=i+1, max_n=len(log_files))
        except KeyboardInterrupt:
            pass

    def load_logfile(self, path: str, num=0, max_n=0):
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
        events = []

        # helper to process a single line
        def process_line(line: str):
            nonlocal events
            line = line.rstrip("\n")
            parts = line.split("|")
            if len(parts) >= 4 and re.search(r"^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{4}", parts[0]):
                events.append({
                    "timestamp": parts[0].strip(),
                    "log_level": parts[1].strip(),
                    "function": parts[2].strip(),
                    "log_message": parts[3].strip()
                })
            elif events:
                events[-1]["log_message"] += "\n" + line


        if self.volatile:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40, complete_style="green", finished_style="green"),
                TaskProgressColumn(),  # shows percent like '23%'
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task(f"[cyan]{num}/{max_n} parsing log file {filename}...[/cyan]", total=len(lines))
                for line in lines:
                    process_line(line)
                    progress.update(task, advance=1)
        else:
            for line in lines:
                process_line(line)


        self.events.extend(events)
        self.loaded_files.add(path)
    
    def get_shape(self):
        return (
            self.window_size, 
            
            self.datetime_encoder.get_dimension() 
            + self.loglevel_encoder.get_dimension() 
            + self.function_encoder.get_dimension() 
            + self.message_encoder.get_dimension()
            )

    def preprocess_dataset_old(self, logs_per_class: int | None = 100, force_logs_per_class: bool = True, shuffle: bool = False, step: int = 1, max_events: int | None = None) -> Dataset:
        if len(self.events) == 0:
            raise ValueError("events must not be empty")        
        
        # create copy of events and shuffle
        _events = self.events.copy()
        if shuffle: random.shuffle(_events)

        # instanciate dataset
        data = Dataset(
            self.get_shape(),
             self.get_key(),
             self.loaded_files
        )
        
        for cl in self.classes.values: data.class_counts[hash_ndarray(self.classes_encoder.encode(cl))] = 0

        n = len(_events)
        end = min(max_events, n) if max_events is not None else n


        # initialize tracker for progress
        if self.volatile: 
            total_iters = max(0, (end - self.window_size + (step - 1)) // step)
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40, complete_style="green", finished_style="green"),
                TaskProgressColumn(),  # shows percent like '23%'
                MofNCompleteColumn(),  # shows progress like '23/100'
            )
            progress.start()
            task = progress.add_task("[cyan]Preparing Dataset...[/cyan]", total=total_iters)
            if logs_per_class is not None:
                cl_tasks = {
                    hash_ndarray(self.classes_encoder.encode(cl)): progress.add_task(f"[cyan]{cl}[/cyan]", total=logs_per_class) for cl in self.classes.values
                }

        # let a sliding window go over the events list
        try:
            for i in range(self.window_size, min(max_events, len(_events)) if max_events else len(_events), step):
                if (logs_per_class is not None) and all(v >= logs_per_class for v in data.class_counts.values()):
                    break
                
                # select the events by sliding window and get the last of the selected events
                window = _events[(i-self.window_size):i]
                
                vec, label = self.encode_window(window), self.annotate_window(window)

                # update progress
                if self.volatile: 
                    # progress.update(1)
                    progress.update(task, advance=1)

                if (logs_per_class is not None) and data.class_counts[hash_ndarray(label)] >= logs_per_class: 
                    continue
                else: 
                    data.add(vec, label)
                    if self.volatile and logs_per_class is not None:
                            progress.update(cl_tasks[hash_ndarray(label)], advance=1)
        except KeyboardInterrupt:
            print("[yellow]Preprocessing dataset interrupted...[/yellow]")

        # log
        if self.volatile:
            print(f"State counts:")
            for k, v in data.class_counts.items(): print(f"  - {k} : {v}")

            progress.stop()

        # Downsample classes
        if force_logs_per_class and logs_per_class is not None:
            print("[yellow]Forcing logs per class by downsampling...[/yellow]")

            class_counts = data.class_counts
            n = min(class_counts.values())
            
            data.data_list
            seen = defaultdict(int)
            out = []
            for t in data.data_list:
                label = hash_ndarray(t[1])
                if seen[label] < n:
                    out.append(t)
                    seen[label] += 1
            
            data.data_list = out

        if (logs_per_class is not None) and all(v == logs_per_class for v in data.class_counts.values()):
            print(f"All states have the desired log count")

        return data
    
    def preprocess_dataset(self, logs_per_class: int | None = 100, force_logs_per_class: bool = True, shuffle: bool = False, step: int = 1, max_events: int | None = None) -> Dataset:
        if len(self.events) == 0:
            raise ValueError("events must not be empty")        
        
        # create copy of events and shuffle
        _events = self.events.copy()
        if shuffle: random.shuffle(_events)

        # instanciate dataset
        data = Dataset(
            (self.window_size, 
             self.datetime_encoder.get_dimension() 
             + self.loglevel_encoder.get_dimension() 
             + self.function_encoder.get_dimension() 
             + self.message_encoder.get_dimension()),
             self.get_key(),
             self.loaded_files
        )
        
        for cl in self.classes.values: data.class_counts[hash_ndarray(self.classes_encoder.encode(cl))] = 0

        window_size = self.window_size
        num_events = len(_events)
        end = min(max_events, num_events) if max_events is not None else num_events

        # return if not enough events
        if end <= window_size: return  # nothing to do

        # initialize tracker for progress
        if self.volatile: 
            total_iters = max(0, (end - window_size + (step - 1)) // step)
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40, complete_style="green", finished_style="green"),
                TaskProgressColumn(),  # shows percent like '23%'
                MofNCompleteColumn(),  # shows progress like '23/100'
            )
            progress.start()
            task = progress.add_task("[cyan]Preparing Dataset...[/cyan]", total=total_iters)
            if logs_per_class is not None:
                cl_tasks = {
                    hash_ndarray(self.classes_encoder.encode(cl)): progress.add_task(f"[cyan]{cl}[/cyan]", total=logs_per_class) for cl in self.classes.values
                }

        # speed up attribute / global lookups inside the loop
        encode_window = self.encode_window
        annotate_window = self.annotate_window
        add = data.add
        class_counts = data.class_counts
        hash_label = hash_ndarray

        # track which classes have already hit their cap so we don't scan the whole dict every time
        done_classes = set()

        # rolling window: prefill the first w items once
        win = deque(_events[:window_size], maxlen=window_size)

        try:
            # iterate i at window end positions
            for i in range(window_size, end, step):
                # early-stop if every known class is done
                if logs_per_class is not None and len(done_classes) == len(class_counts):
                    break

                ###############################################

                # slide the window if it not the first iteration
                if i > window_size:
                    win.extend(_events[i - step:i])

                # compute vec/label from the current window
                vec = encode_window(win)
                label = annotate_window(win)

                # update progress
                if self.volatile and progress is not None and task is not None: progress.update(task, advance=1)

                # if there is no maximum logs per class, add the log and continue
                if logs_per_class is None:
                    add(vec, label)
                    continue


                # per-class capping with single hash and no double work
                hashed_label = hash_label(label)
                previous_cnt = class_counts[hashed_label]
                # continue if class already reached its log limit
                if previous_cnt >= logs_per_class: continue
                
                # add the window
                add(vec, label)

                # update the done classes set
                new_cnt = previous_cnt + 1
                if new_cnt >= logs_per_class:
                    done_classes.add(hashed_label)

                # update progresses
                if self.volatile and cl_tasks is not None and hashed_label in cl_tasks:
                    # advance one (or mark complete) for this class
                    if new_cnt >= logs_per_class:
                        progress.update(cl_tasks[hashed_label], completed=logs_per_class)
                    else:
                        progress.update(cl_tasks[hashed_label], advance=1)
        except KeyboardInterrupt:
            print("[yellow]Preprocessing dataset interrupted...[/yellow]")

        # print info to console and stop progress
        if self.volatile:
            print(f"State counts:")
            for k, v in data.class_counts.items(): print(f"  - {k} : {v}")

            progress.stop()

        # Downsample classes
        if force_logs_per_class and logs_per_class is not None:
            print("[yellow]Forcing logs per class by downsampling...[/yellow]")

            class_counts = data.class_counts
            num_events = min(class_counts.values())
            
            data.data_list
            seen = defaultdict(int)
            out = []
            for t in data.data_list:
                label = hash_ndarray(t[1])
                if seen[label] < num_events:
                    out.append(t)
                    seen[label] += 1
            
            data.data_list = out

        if (logs_per_class is not None) and all(v == logs_per_class for v in data.class_counts.values()):
            print(f"All states have the desired log count")

        return data
    


    def annotate_window(self, window):
        label = self.classes.annotate(window)
        return self.classes_encoder.encode(label)

    def encode_window(self, window):
        sequence_features = []

        for ev in window:
            dt = datetime.fromisoformat(ev["timestamp"])
            dt_feature = self.datetime_encoder.extract_date_time_features(dt)

            log_level_feature = self.loglevel_encoder.encode(ev['log_level'])

            function_feature = self.function_encoder.encode(ev['function'])

            log_msg_feature = self.message_encoder.encode(ev['log_message'])

            feature_vector = [val for val in dt_feature.values()] + [log_level_feature, function_feature, log_msg_feature]
            flat_feature = np.concatenate([to_flat_array(f) for f in feature_vector])
            sequence_features.append(flat_feature)

        return np.array(sequence_features)

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

        vec, label = self.encode_window(events), self.annotate_window(events)
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
            "name": self.name,
            "classes": self.classes.values
            }
        for encoder in [self.message_encoder, 
                        self.function_encoder, 
                        self.datetime_encoder, 
                        self.loglevel_encoder, 
                        self.classes_encoder]:
            encoder_t, k, filename = save_encoder_if_new(encoder, path, timestamp)
            obj[f"{encoder_t}_k"] = k
            obj[f"{encoder_t}_f"] = filename
            # print(filename, type(filename))
            paths.append(filename)

        # save the preprocessor json
        with open(preprocessor_path, 'w') as f:
            json.dump(obj, f, indent=4)

        return paths

    @staticmethod
    def load(preprocessor_path: str, 
             custom_encoder_directory: str | None = None,
             custom_encoder_paths: dict[str, str] | None = None,
             annotate: Callable[[list[dict[str,str]]], Any] | None = None,
             volatile: bool = True):
        with open(preprocessor_path, 'r') as f:
            json_obj = json.load(f)
        
        name = json_obj["name"]
        window_size = json_obj["window_size"]
        obj_classes = json_obj["classes"]
        encoders = {}

        for encoder_t in EncoderType.types():
            path = None
            if custom_encoder_paths is not None and encoder_t in custom_encoder_paths:
                path = custom_encoder_paths[encoder_t]
            elif custom_encoder_directory is not None:
                path = os.path.join(custom_encoder_directory, json_obj[f"{encoder_t}_f"])
            else:
                path = os.path.join(os.path.dirname(preprocessor_path), json_obj[f"{encoder_t}_f"])

            encoders[encoder_t] = joblib.load(path)

        obj = Preprocessor(
            message_encoder=encoders[EncoderType.message],
            datetime_encoder=encoders[EncoderType.datetime],
            log_level_encoder=encoders[EncoderType.loglevel],
            function_encoder=encoders[EncoderType.function],
            classes_encoder=encoders[EncoderType.classes],
            classes=Classes(obj_classes, annotate=annotate),
            window_size=window_size,
            name=name,
            volatile=volatile
        )

        return obj


if __name__ == "__main__":
    # preprocessing
    load = True
    if load:
        pp = Preprocessor.load("[Hv_-zDu9v7jr4Sam][20250811_105705]preprocessor_test.json", annotate=cl_annotate)
    else:
        pp = Preprocessor(
            message_encoder=MessageTextVectorizationEncoder(max_tokens=1000, output_mode="count"),
            function_encoder=FunctionOneHotEncoder(min_frequency=3),
            datetime_encoder=DatetimeEncoder([
                DatetimeFeature.second.since_midnight,
                DatetimeFeature.day.since_epoch
            ]),
            log_level_encoder=LogLevelOneHotEncoder(),
            classes_encoder=ClassesLabelBinarizer(),
            classes=classes,
            window_size=20,
            name="preprocessor_test",
            volatile=True
        )

    pp.load_logfiles([
        rf"D:\mgeo\projects\log-classification\data\CCI\CCLog-backup.{i}.log"
        for i in [749]
    ])

    pp.initialize_encoders()

    if not load: pp.save(".")


    ds_load = True
    if ds_load:
        dataset = Dataset.load("dataset.npz", validate_shape=True)
    else:
        try:
            dataset = pp.preprocess_dataset(logs_per_class=100, shuffle=True, step=100)
        except KeyboardInterrupt:
            pass

        dataset.save("dataset.npz", True)

    train, test = dataset.stratified_split((4, 1))
    X_train, y_train = train
    X_test, y_test = test

