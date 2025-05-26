import re
from tqdm import tqdm
from collections import defaultdict
from states import States as S
import datetime
from keras.layers import TextVectorization
from sklearn.preprocessing import LabelEncoder

LOG_PATH = "./data/CCI/CCLog-backup.{n}.log"
LOG_LEVEL_MAP = {'Trace': 0, 'Debug': 1, 'Info': 2, 'Warn': 3, 'Error': 4, 'Fatal': 5}







class Preprocessor:
    def __init__(self, 
                 log_numbers: list[int], 
                 logs_per_class: int = 100,
                 window_size: int = 20, 
                 volatile: bool = False):
        self.volatile = volatile

        self.annotated = []
        self.states_counts = defaultdict(int)
        self.logs_per_class = logs_per_class
        self.window_size = window_size

        
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
            progress (tqdm): optional tqdm object for tracking progress
        
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
        tokenizer = self.get_log_message_encoder()
        function_encoder = self.get_function_encoder()
        for ev, state in self.annotated:
            last_event = ev[-1]
            
            date, time = self.extract_date_time_features(datetime.datetime.fromisoformat(last_event["timestamp"]))
            log_level = LOG_LEVEL_MAP[last_event['log_level']]
            function_id = function_encoder.transform([last_event['function']])[0]
            log_msg_token = tokenizer([last_event['log_message']])[0]
            msg_token_id = log_msg_token[0] if log_msg_token else 0
            processed.append([date, time, log_level, function_id, msg_token_id, state])
        return processed
    
    def _p(self):
        processed = []
        tokenizer = self.get_log_message_encoder()
        function_encoder = self.get_function_encoder()

        for ev_seq, state in self.annotated:
            sequence_features = []

            for ev in ev_seq:
                dt = datetime.datetime.fromisoformat(ev["timestamp"])
                date, time = self.extract_date_time_features(dt)
                log_level = LOG_LEVEL_MAP.get(ev['log_level'], 0)
                function_id = function_encoder.transform([ev['function']])[0]
                log_msg_token = tokenizer([ev['log_message']])
                log_msg_token_id = log_msg_token[0] if log_msg_token else 0

                sequence_features.append([date, time, log_level, function_id, log_msg_token_id])

            processed.append((sequence_features, state))

        return processed

    def extract_date_time_features(self, dt: datetime.datetime):
        # Normalize date as days since epoch
        date_feature = (dt.date() - datetime.datetime(1970, 1, 1).date()).days

        # Time in seconds since midnight
        time_feature = dt.hour * 3600 + dt.minute * 60 + dt.second
        # print(date_feature, time_feature)

        return date_feature, time_feature

    def get_log_message_encoder(self):
        text_vectorizer = TextVectorization(
            max_tokens=10000,
            output_mode='int',
            pad_to_max_tokens=True,
            output_sequence_length=1  # We use 1 token per message
        )
        
        all_log_messages = [event['log_message'] for seq in self.annotated for event in seq[0]]
        text_vectorizer.adapt(all_log_messages)

        return text_vectorizer

    def get_function_encoder(self):
        all_functions = [event['function'] for seq in self.annotated for event in seq[0]]

        function_encoder = LabelEncoder()
        function_encoder.fit(all_functions)

        return function_encoder



if __name__ == "__main__":
    pp = Preprocessor([i for i in range(745, 760)], volatile=True)
    data = pp._p()

    print(data[0])
