from __future__ import annotations

import random
import numpy as np
import json
import os
import joblib
import re
from collections import defaultdict, deque
from typing import Callable, Any, List, Dict, Tuple
from datetime import datetime
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

from dataset import Dataset
from encoders.datetime_encoder import DatetimeEncoder
from encoders.datetime_features import DatetimeFeature
from encoders.loglevel_encoder import *
from encoders.message_encoder import *
from encoders.function_encoder import *
from encoders.classes_encoder import *
from encoders.save_encoder import save_encoder_if_new
from classes import Classes, classes, annotate as cl_annotate
from util import hash_list_to_string, hash_ndarray
from encoders.encoder_type import EncoderType
from util import *


class Preprocessor:
    """
    Preprocessor class to annotate and preprocess Log files into a readable format for Machine Learning
    """
    def __init__(self, 
                 message_encoder:  MessageEncoder,
                 function_encoder: FunctionEncoder,
                 datetime_encoder: DatetimeEncoder,
                 log_level_encoder: LogLevelEncoder,

                 classes_encoder: ClassesEncoder,

                 classes: Classes,
                 
                 window_size: int = 20,
                 name: str | None = None,

                 verbose: bool = False):
        """
        Initializes a new Preprocessor object. 
        Pass different kinds of encoder objects to encode features (message, function, datetime, loglevel) and labels (classes). 
        
        :params message_encoder: Encoder used to encode log messages
        :params function_encoder: Encoder used to encode function
        :params datetime_encoder: Encoder used to encode datetime features
        :params log_level_encoder: Encoder used to encode log levels

        :params classes_encoder: Encoder used to encode labels/classes

        :params classes: Classes object defining which labels exist and how to annotate them
        :params window_size: How many logs should be combined into a window
        :params name: Name of the preprocessor, `preprocessor` by default
        :params verbose: Display various additional information on the console
        """
        self.name = name or "preprocessor"

        # whether to print progress and information while training
        self.verbose = verbose

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
        """
        Initialize all encoders if they haven't been initialized already. Events must be loaded before initializing the encoders.
        """
        if not self.encoders_initialized:
            if len(self.events) == 0:
                raise Exception("Events must be loaded before initializing the encoders")
            
            if not self.message_encoder.initialized: self.message_encoder.initialize([ev["log_message"] for ev in self.events])
            if not self.function_encoder.initialized: self.function_encoder.initialize([ev["function"] for ev in self.events])
            if not self.loglevel_encoder.initialized: self.loglevel_encoder.initialize([ev["log_level"] for ev in self.events])
            if not self.classes_encoder.initialized: self.classes_encoder.initialize(self.classes.values)

            self.encoders_initialized = True

    def load_logfiles(self, log_files: list[str]):
        """
        Load all events from log files in a list into the preprocessor, this process can be interrupted with a KeyboardInterrupt

        :log_files: A list of paths to log files to be loaded
        """
        try:
            for i, f in enumerate(log_files): 
                events = self.load_logfile(f, num=i+1, max_n=len(log_files))
                
                self.events.extend(events)
                self.loaded_files.add(f)
        except KeyboardInterrupt:
            pass

    def load_logfile(self, path: str, num: int = 0, max_n: int = 0) -> List[Dict[str, str]]:
        """
        Load and parse a logfile into a list of events. 
        Each event has the following keys: "timestamp", "log_level", "function", "log_message"
        Displays a progress bar if self.verbose is True

        :param path: The path to the log file.
        :param num: Number of currently loading log file - Used for progress tracking
        :param max_n: Maximum value of num - Used for progress tracking
        :returns: a list of event dictionaries
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


        if self.verbose:
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

        return events
    
    def get_shape(self) -> Tuple[int, int]:
        """
        Get the input (X) shape of features preprocessed with this preprocessor. => (window_size, features_size)
        """
        return (
            self.window_size, 
            
            self.datetime_encoder.get_dimension() 
            + self.loglevel_encoder.get_dimension() 
            + self.function_encoder.get_dimension() 
            + self.message_encoder.get_dimension()
            )

    def preprocess_dataset(self, 
                           logs_per_class: int | None = 100, 
                           force_same_logs_per_class: bool = True, 
                           step: int = 1, 
                           max_events: int | None = None) -> Dataset:
        """
        Preprocess the previously loaded log files into a Dataset

        :params logs_per_class: how many events should at least be added to the dataset for each class. If this is None the number of objects per class is not limited
        :params force_same_logs_per_class: If this True, the class counts will be downsampled to the class with the fewest events
        :params step: In what increment the window indexes increase
        :params max_events: Only consider that many events
        :returns: a `Dataset` object containing preprocessed data
        """
        
        if len(self.events) == 0:
            raise ValueError("events must not be empty")        
        
        # create copy of events and shuffle
        _events = self.events.copy()

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
        if self.verbose: 
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

                #################
                # update progress
                if self.verbose and progress is not None and task is not None: progress.update(task, advance=1)

                # slide the window if it not the first iteration
                if i > window_size:
                    win.extend(_events[i - step:i])

                label = annotate_window(win)
                # per-class capping with single hash and no double work
                hashed_label = hash_label(label)
                previous_cnt = class_counts[hashed_label]
                # continue if class already reached its log limit
                if previous_cnt >= logs_per_class: continue

                # compute vec/label from the current window
                vec = encode_window(win)

                # if there is no maximum logs per class, add the log and continue
                if logs_per_class is None:
                    add(vec, label)
                    continue
                
                # add the window
                add(vec, label)

                # update the done classes set
                new_cnt = previous_cnt + 1
                if new_cnt >= logs_per_class:
                    done_classes.add(hashed_label)

                # update progresses
                if self.verbose and cl_tasks is not None and hashed_label in cl_tasks:
                    # advance one (or mark complete) for this class
                    if new_cnt >= logs_per_class:
                        progress.update(cl_tasks[hashed_label], completed=logs_per_class)
                    else:
                        progress.update(cl_tasks[hashed_label], advance=1)
        except KeyboardInterrupt:
            print("[yellow]Preprocessing dataset interrupted...[/yellow]")

        # print info to console and stop progress
        if self.verbose:
            print(f"State counts:")
            for k, v in data.class_counts.items(): print(f"  - {k} : {v}")

            progress.stop()

        # Downsample classes to the class with the fewest events
        if force_same_logs_per_class and logs_per_class is not None:
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
    
    def annotate_window(self, window: List[Dict[str,str]]) -> np.ndarray:
        """
        assigns a class/label to a window

        :params window: The window of objects to annotate
        """
        label = self.classes.annotate(window)
        return self.classes_encoder.encode(label)

    def encode_window(self, window: List[Dict[str,str]]) -> np.ndarray:
        """
        Encodes a window to make it readable for Machine Learning Models

        :params window: The window of objects to encode
        """
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

    # TODO: rework this
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
        
    def get_key(self) -> str:
        """
        get a key that is unique to the preprocessor with these encoders and parameters
        """

        key = hash_list_to_string([
            self.message_encoder.get_key(),
            self.function_encoder.get_key(),
            self.loglevel_encoder.get_key(),
            self.datetime_encoder.get_key(),

            str(self.window_size)
            ], 16)
        
        return key

    def save(self, path: str) -> List[str]:
        """
        saves the preprocessor to several files. The files will be placed at path.

        The files are: 
        - ./[\<key>][\<timestamp>]\<name>.json
        - ./encoders/\<encoder-type>/[\<key>][\<timestamp>].pkl

        For a total of 6 files (1 preprocessor json + 5 encoder .pkl files)

        If a file with the key already exists, that file will be skipped, and not saved again.
        
        :params path: A path to a directory to save the files in
        :returns: A list of strings with the paths to the files
        """
        
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
            encoder_t, k, filename = save_encoder_if_new(encoder, path, timestamp, verbose=self.verbose)
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
             custom_encoder_paths: Dict[str, str] | None = None,
             annotate: Callable[[List[Dict[str,str]]], Any] | None = None,
             verbose: bool = True):
        """
        Load a preprocessor and encoders from files. 
        By default encoder files are expected at ./encoders/\<encodertype>/\<encoder_name>.pkl relative to the encoders location. 
        Custom directory or custom file paths can be specified as arguments.

        :params preprocessor_path: The path to the json file defining the preprocessor
        :params custom_encoder_directory: A custom path to search for encoders. Will then look for encoders at \<custom_path>/\<encodertype>/\<encoder_name>.pkl
        :params custom_encoder_paths: A dictionary containing completely custom paths to encoders. The keys are names defined in `EncoderType.types()`
        :params annotate: A callable to annotate data with - the callable must take a dictionary of strings and return the label
        :params verbose: Whether to print console information - also makes the preprocessor itself verbose
        """
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
            verbose=verbose
        )

        return obj

