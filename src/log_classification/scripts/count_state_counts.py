import re
import os
from collections import defaultdict
from tqdm import tqdm

import json

def get_sorted_log_numbers_by_size(directory):
    pattern = re.compile(r'^CCLog-backup\.(\d+)\.log$')
    log_files = []

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            n = int(match.group(1))
            full_path = os.path.join(directory, filename)
            size = os.path.getsize(full_path)
            log_files.append((n, size))

    # Sort by file size
    log_files.sort(key=lambda x: x[1])

    # Return only the numbers n
    return [n for n, _ in log_files]


STATE_NORMAL = "state_normal"
STATE_UNOBSERVED = "state_unobserved"
STATE_DBERR = "state_database_error"
STATE_HLIERR = "state_hli_session_error"
CONT = "continuation"
LINES = "lines"
CHARS = "chars"


def get_counts(files, save_json=False):
    counts = defaultdict(int)
    function_counts = defaultdict(lambda: defaultdict(int))

    for file_num in tqdm(files, desc="Processing log files"):

        with open(f"./data/CCI/CCLog-backup.{file_num}.log", "r") as f:
            lines = f.readlines()

            

            # Optional: Inner progress bar (can slow down large files)
            for line in tqdm(lines, desc=f"Processing log {file_num}"):
                counts[CHARS] += len(line)
                m = re.search(r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{4}) \| (?P<log_level>\w+) \| (?P<function>.+) \|", line)
                if m:
                    counts[LINES] += 1

                    log_level = m.group("log_level")
                    function = m.group("function")

                    function_counts[function][log_level] += 1

                    if function == "C_line_Control_Server.CCServerAppContext.TaskSchedulerUnobservedTaskException":
                        counts[STATE_UNOBSERVED] += 1

                    elif log_level == "Error":
                        if "DBProxyMySQL" in function or "DBManager" in function:
                            counts[STATE_DBERR] += 1
                        elif "SessionFactory.OpenSession" in function:
                            counts[STATE_HLIERR] += 1
                        else:
                            counts[STATE_NORMAL] += 1
                    else:
                        counts[STATE_NORMAL] += 1
                else:
                    counts[CONT] += 1
    
    if save_json:
        # Convert defaultdicts to standard dicts
        def to_dict(obj):
            if isinstance(obj, defaultdict):
                return {k: to_dict(v) for k, v in obj.items()}
            return obj

        # Save counts to JSON
        with open("counts.json", "w", encoding="utf-8") as f:
            json.dump(to_dict(counts), f, indent=4, ensure_ascii=False)

        # Save function-level counts to JSON
        with open("function_counts.json", "w", encoding="utf-8") as f:
            json.dump(to_dict(function_counts), f, indent=4, ensure_ascii=False)

        print("\n✅ Saved counts to counts.json and function_counts.json")

    return counts, function_counts


counts, function_counts = get_counts([770,771,772])
[
    585,
    588, 589, 590, 591,
    595, 596, 597, 598, 599, 600,
    602, 603,
    610, 611,
    749, 750, 751, 752,
    768
]


total_lines = counts[LINES] or 1  # avoid division by zero

def percent(part):
    return f"{(part / total_lines * 100):6.4f}%"


# --- Summary Output ---
print("\nLog Summary:")
print(f"Total lines processed:       {total_lines}")
print(f"Normal log entries:          {counts[STATE_NORMAL]:>8} ({percent(counts[STATE_NORMAL])})")
print(f"Unobserved task exceptions:  {counts[STATE_UNOBSERVED]:>8} ({percent(counts[STATE_UNOBSERVED])})")
print(f"Database errors:             {counts[STATE_DBERR]:>8} ({percent(counts[STATE_DBERR])})")
print(f"HLI session errors:          {counts[STATE_HLIERR]:>8} ({percent(counts[STATE_HLIERR])})")


print("continued lines", counts[CONT])
print("total lines:", counts[CONT] + total_lines)
print("total chars:", counts[CHARS])

log_levels_of_interest = ["Debug", "Info", "Warn", "Error", "Trace", "Fatal"]

# Initialize total counts
log_level_totals = defaultdict(int)

# Sum the levels
for func, levels in function_counts.items():
    for level, count in levels.items():
        if level in log_levels_of_interest:
            log_level_totals[level] += count

# Display result
print("\nLog Level Totals:")
for level in log_levels_of_interest:
    print(f"{level:<7}: {log_level_totals[level]:>8} ({percent(log_level_totals[level])})")




