import re
from collections import defaultdict

LOG_PATH = "C:/Users/Askion/Documents/agmge/log-classification/data/selected"

def parameterize_known_patterns(line: str):
    # Start -> Timeout: 2500 ms
    # Exception (2509 ms): TimedOut 
    # End -> Bytes read: 0
    # HliSession::Receive(): 10.239.134.85:40001 - 'HS200_03028' -> 2500
    # HliSession::Dispose(): 10.239.134.85:40001 - 'HS200_03028' -> True
    # HliSession::Close(): 10.239.134.85:40001 - 'HS200_03028'
    # HliSession::Send(): 10.239.134.85:40001 - 'HS200_03028' -> <ASKION_C-LINE_HLI><VERSION V="3.50" /><MESSAGE MES="DET" NUM="13382" CID="GERS229" /></ASKION_C-LINE_HLI>
    ...

def parse_logfile(path: str):
    with open(path, "r") as f:
        lines = f.readlines()

    events = []
    while lines:
        line = lines[0].strip("\n ")

        # if this is the start of an event, create a new event
        if re.search(r"^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{4}", line):
            parts = line.split(" | ")
            event = {
                "timestamp": parts[0],
                "log_level": parts[1],
                "class": parts[2],
                "log_message": parts[3]
            }
            events.append(event)

        # otherwise add to the previous log message
        elif events:
            events[-1]["log_message"] += "\n" + line

        lines.pop(0)
    
    return events

def extract_messages(events, out_file: str):
    with open(out_file, "w") as f:
        for e in events:
            f.write(e["log_message"].replace("\n", "__\\n__")+"\n")

# ChatGPT generated, fixed myself...
def process_file(input_path, output_path, pattern_replacements):
    """
    Processes a file line by line applying regex transformations.

    Parameters:
    - input_path (str): Path to the input .txt file.
    - output_path (str): Path to save the processed file.
    - pattern_replacements (dict): A dictionary where keys are regex patterns
      and values are the replacement strings for the first match.
    """
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        replaced = set()
        out_lines = []
        for line in infile:
            
            needs_to_print = True
            for pattern, replacement in pattern_replacements.items():
                # Find all matches
                if re.match(pattern, line):
                    if not pattern in replaced:
                        out_lines.insert(0, replacement)
                        replaced.add(pattern)
                    needs_to_print = False
                    break
            
            if needs_to_print:
                out_lines.append(line)

        outfile.writelines(out_lines)


patterns = {
    r"Start -> Timeout: \d+ ms\s*": "Start -> Timeout: <x1> ms\n",
    r"Exception \(\d+ ms\): TimedOut\s*": "Exception (<x1> ms): TimedOut\n",
    r"End -> Bytes read: \d+\s*": "End -> Bytes read: <x1>\n",
    r"HliSession::Receive\(\): \d+\.\d+\.\d+\.\d+:\d+ - '.*' -> \d+\s*": "HliSession::Receive(): <x1>:<x2> - 'x3' -> <x4>\n",
    r"HliSession::Dispose\(\): \d+\.\d+\.\d+\.\d+:\d+ - '.*' -> .+\s*": "HliSession::Dispose(): <x1>:<x2> - 'x3' -> <x4>\n",
    r"HliSession::Close\(\): \d+\.\d+\.\d+\.\d+:\d+ - '.*'\s*": "HliSession::Close(): <x1>:<x2> - '<x3>'\n",
    r"HliSession::Send\(\): \d+\.\d+\.\d+\.\d+:\d+ - '.*' -> .+\s*": "HliSession::Send(): <x1>:<x2> - 'x3' -> <x4>\n",
}

process_file('messages.txt', 'messages1.txt', patterns)
