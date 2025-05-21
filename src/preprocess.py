import re
from tqdm import tqdm
from collections import defaultdict

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


def annotate_data(events: list[dict], prev_logs: int = 3, next_logs: int = 3, max_logs_per_class: int = 5, track_progress: bool = False):
    """
    annotates data by adding a "state" key to every event.
    Currently data is categorized into 4 categories: 
        - normal
        - fatal
        - database
        - connection
    based on simple pattern matching.

    Args:
        events (list): list of events as returned by load_logfile.
        progress (tqdm): optional tqdm object for tracking progress
    
    Returns:
        the list of events
    """
    # initialize tracker for progress
    if track_progress: progress = tqdm(total=len(events), desc="annotating events")

    event_seq = []

    states_counts = defaultdict(int)
    for i, event in enumerate(events):
        if track_progress: progress.update(1)

        lower = i-prev_logs
        upper = i+next_logs
        if lower < 0 or upper >= len(events): continue
        seq = events[lower:upper]

        if states_counts["fatal"] < max_logs_per_class and  event["log_level"] == "Fatal":
            event_seq.append((seq, "fatal"))
            states_counts["fatal"] += 1
            continue
        
        elif event["log_level"] == "Error":
            if "DBProxyMySQL" in event["function"] or "DBManager" in event["function"]:
                event["state"] = "database"
                states["database"] += 1
                continue
            elif "OnCDILogin" in event["function"]:
                event["state"] = "connection"
                states["connection"] += 1
                continue
            else:
                event["state"] = "normal"
                states["normal"] += 1
                continue
        else:
            event["state"] = "normal"
            states["normal"] += 1
    
    return events, states

def reduce_data(events: list[dict]):
    pass


if __name__ == "__main__":
    for n in range(745, 760):
        file_path = f"./data/CCI/CCLog-backup.{n}.log"

        events = load_logfile(file_path, True)
        events, states = annotate_data(events, True)

        fatal = states['fatal']
        normal = states['normal']
        database = states['database']
        connection = states['connection']
        print(f"file: {file_path}")
        print(f"normal: {normal} ({100*normal/len(events)}%)")
        print(f"database: {database} ({100*database/len(events)}%)")
        print(f"connection: {connection} ({100*connection/len(events)}%)")
        print(f"fatal: {fatal} ({100*fatal/len(events)}%)")
        print()