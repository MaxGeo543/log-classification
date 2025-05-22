import re
from tqdm import tqdm
from collections import defaultdict
from states import States as S
import datetime
from keras.layers import TextVectorization

def load_logfile(path: str, track_progress: bool = False) -> list[dict]:
    """
    Load and parse a logfile into a list of events. Each event has the following keys: "timestamp", "log_level", "function", "log_message"

    Args:
        path (str): The path to the log file.
        progress (tqdm): optional tqdm object for tracking progress
    
    Returns:
        a list of event dictionaries
    """
    with open(path, "r") as f:
        lines = f.readlines()

    # initialize tracker for progress
    if track_progress: progress = tqdm(total=len(lines), desc="parsing log file")

    events = []
    while lines:
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
        if track_progress: progress.update(1)
    
    # return the list of events
    return events

def annotate_data(events: list[dict], window_size: int = 20, max_logs_per_class: int = 100, annotated: tuple[list,defaultdict]|None = None, track_progress: bool = False) -> tuple[list,defaultdict]:
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
    if track_progress: progress = tqdm(total=len(events), desc="annotating events")

    event_seq = [] if annotated is None else annotated[0]
    states_counts = defaultdict(int) if annotated is None else annotated[1]


    for i in range(window_size, len(events)):
        if track_progress: progress.update(1)
        seq = events[(i-window_size):i]
        last_event = seq[-1]

        s = None
        if last_event["function"] == "C_line_Control_Server.CCServerAppContext.TaskSchedulerUnobservedTaskException":
            if states_counts[S.UnobservedException] >= max_logs_per_class: continue
            
            event_seq.append((seq, S.UnobservedException))
            states_counts[S.UnobservedException] += 1
            continue
        
        elif last_event["log_level"] == "Error":
            if "DBProxyMySQL" in last_event["function"] or "DBManager" in last_event["function"]:
                if states_counts[S.DatabaseError] >= max_logs_per_class: continue
                
                event_seq.append((seq, S.DatabaseError))
                states_counts[S.DatabaseError] += 1
                continue

            elif "OnCDILogin" in last_event["function"]:
                if states_counts[S.CDILoginError] >= max_logs_per_class: continue
                
                event_seq.append((seq, S.CDILoginError))
                states_counts[S.CDILoginError] += 1
                continue
            
            elif "SessionFactory.OpenSession" in last_event["function"]:
                if states_counts[S.HliSessionError] >= max_logs_per_class: continue
                
                event_seq.append((seq, S.HliSessionError))
                states_counts[S.HliSessionError] += 1
                continue

        else:
            if states_counts[S.Normal] >= max_logs_per_class: continue
                
            event_seq.append((seq, S.Normal))
            states_counts[S.Normal] += 1
            continue
    
    return events, states_counts





from keras.layers import TextVectorization

# Flatten all log messages from all sequences
all_log_messages = [event['log_message'] for seq in data for event in seq]

# Create and adapt the vectorizer
text_vectorizer = TextVectorization(
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=1  # We use 1 token per message
)
text_vectorizer.adapt(all_log_messages)




LOG_LEVEL_MAP = {'Trace': 0, 'Debug': 1, 'Info': 2, 'Warn': 3, 'Error': 4, 'Fatal': 5}

def extract_date_time_features(dt: datetime):
    # Normalize date as days since epoch
    date_feature = (dt.date() - datetime(1970, 1, 1).date()).days
    # Time in seconds since midnight
    time_feature = dt.hour * 3600 + dt.minute * 60 + dt.second
    return date_feature, time_feature

def pre_process(seq, function_encoder, tokenizer):
    processed = []
    for ev, state in seq:
        date, time = extract_date_time_features(datetime.datetime.fromisoformat(ev["timestamp"]))
        log_level = LOG_LEVEL_MAP[ev['log_level']]
        function_id = function_encoder.transform([ev['function']])[0]
        log_msg_token = tokenizer.texts_to_sequences([ev['log_message']])[0]
        msg_token_id = log_msg_token[0] if log_msg_token else 0
        processed.append([time_delta, log_level, function_id, msg_token_id])
    return processed

if __name__ == "__main__":
    annotated, states_counts = [], defaultdict(int)
    logs_per_class = 100
    window_size = 20

    log_file_numbers =  list(range(745, 760))

    for n in log_file_numbers:
        file_path = f"./data/CCI/CCLog-backup.{n}.log"

        print(f"Current log file: {file_path}")
        events = load_logfile(file_path, True)
        annotated, states_counts = annotate_data(events, window_size, logs_per_class, (annotated, states_counts), True)
        print(f"Current State counts:")
        for k, v in states_counts.items(): print(f"  - {k} : {v}")

        if all(v == logs_per_class for v in states_counts.values()):
            print(f"All states have the desired log count")
            break
    

    preprocessed_data = pre_process(annotated, function_encoder)
