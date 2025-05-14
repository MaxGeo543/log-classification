import re
from collections import defaultdict
from tqdm.notebook import tqdm

LOG_PATH = "C:/Users/Askion/Documents/agmge/log-classification/data/selected"
PATTERNS = [
    r"Start -> Timeout: (\d+) ms\s*",
    r"Exception \((\d+) ms\): TimedOut\s*",
    r"End -> Bytes read: (\d+)\s*",
    r"HliSession::Receive\(\): (\d+\.\d+\.\d+\.\d+):(\d+) - '(.*)' -> (\d*)\s*",
    r"HliSession::Dispose\(\): (\d+\.\d+\.\d+\.\d+):(\d)+ - '(.*)' ->\s?(.*)\s*",
    r"HliSession::Close\(\): (\d+\.\d+\.\d+\.\d+):(\d+) - '(.*)'\s*",
    r"HliSession::Send\(\): (\d+\.\d+\.\d+\.\d+):(\d+) - '.*' ->\s?(.*)\s*",
    r"HliSession::CloseInternal\(\): Close (\d+\.\d+\.\d+\.\d+):(\d+) - '(.*)' ->\s*",
    r"HliSession::CloseInternal\(\): (\d+\.\d+\.\d+\.\d+):(\d+) - '(.*)' ->\s?(.*)\s*",
    r"(\d+\.\d+\.\d+\.\d+):(\d+) - '(.*)'\s*",
    r"End\s*",
    r"Read (\d+) bytes in \((\d{2}:\d{2}:\d{2}.\d{7})\)\s*",
    r"HliSession::OnSessionClosed\(\): (\d+\.\d+\.\d+\.\d+):(\d+) - '(.*)'\s*",
    r"(\d+\.\d+\.\d+\.\d+):(\d+) - '(.*)' ->\s?(.*)\s*",
    r"Start: buffer.Length: (\d+), timeout: (\d+)\s*",
    r"(False|True) = (False|True)\s*",
    r"SessionFactory::OpenSession\(\): (\d+\.\d+\.\d+\.\d+):(\d+) - '(.*)' ->\s?(.*)\s*",
    r"SessionFactory::OpenSession\(\): End: (\d+\.\d+\.\d+\.\d+):(\d+) - '(.*)' ->\s?(.*)\s*",
    r"SessionFactory::OpenSession\(\): (\d+\.\d+\.\d+\.\d+):(\d+) - '(.*)'\s*",
    r"SELECT (.*) FROM (.*) WHERE (.*)\s*",
    r"SELECT (.*) FROM (.*) where (.*)\s*",
    r"SELECT (.*) from (.*) WHERE (.*)\s*",
    r"SELECT (.*) from (.*) where (.*)\s*",
    r"Select (.*) from (.*) where (.*)\s*",
    r"Select (.*) from (.*)\s*",
    r"SELECT (.*) from (.*)\s*",
    r"SELECT (.*) FROM (.*)\s*",
    r"SELECT (.*)\s*",
    r"Select (.*)\s*",
    r"(True|False)\s*",
    r"(.+):(\d+) -> (.+) - '(.*)' ///\s?(.*)\s*",
    r"(.+):(\d+) -> (.+) - '(.*)'\s*",
    r"(.+):(\d+)\s*",
    r"called\s*",
    r"path not found\s*",
    r"(\d+\.\d+\.\d+\.\d+\.\d+\.\d+)-->State: (True|False)\s*",
    r"buffer\.Length: (\d+)\s*",
    r"TaskStatus: (.*)\s*",
    r"C_line_Control_Server\.CDIClientManager\s*",
    r"prc_trackinghistory_device_new\s*",
    r"prc_system_message_new\s*",
    r"System\.Net\.Sockets\.TcpClient\s*",
    r"Return (.*)\s*",
    r"MessageID: (\d+) MessageType: (.{3})\s*",
    r"(\d+)\s*",
    r"System\.AggregateException: One or more errors occurred. --->\s?(.*)\s*",
    r"send Dewar EMS for: (.*)\s*",
    r"Exception \((\d+) ms\): ConnectionReset\s*",
    r"\[is not a valid DBCommand object!\]\s*",
    r"Start Deleting EMSMessages\s*",
    r"prc_md_ems_message_del\s*",
    r"cache pending: HLIAcks\[id: (\d*), HliClient: (.*), HliMessageID: (\d*), CdiMessageID: (\d*), SampleID: (\d*), StorageID: (\d*), UserID: (\d*), HliMessageLength: (\d*)]\s*",
    r"catched SocketException -> SendQueueMessage\(\) get cancelled; SocketErrorCode: ConnectionRefused\s*",
    r"C_line_Control_Server\.Services\.PushService\s*",
    r"prc_md_cachependingpush\s*",
    r"prc_trackingerror_new\s*",
    r"C_line_Control_Server\.Services\.PushService\s*",

    r"CLineCommon_CCIMessage\.(\w+)\s*",
    r"CLineCommon_CCIMessage\.(\w+) -> Askion\.CLine\.Common\.Security\.AuthenticateAttribute\s*",
    r"ClinePrincipal: Identity \[ClineIdentity: UserId \[(\d+)\], Name \[(.*)\], AuthenticationType \[(.*)\]\], HSLevel \[(\d+)\] -> CLineCommon_CCIMessage\.(\w+)\s*",
    r"ClinePrincipal: Identity \[ClineIdentity: UserId \[(\d+)\], Name \[(.*)\], AuthenticationType \[(.*)\]\], HSLevel \[(\d+)\] -> CLineCommon_CCIMessage\.(\w+) \((\d+)\)\s*",
    r"MessageID: (\d+) MessageType: (.*)\s*",
    r"Calling DBRequestFinished\s*",
    r"entered\s*",
    r"\?<\?xml version=\"1.0\" encoding=\"utf-8\"\?>(.*)\s*",
    r"Got Client: (\d+\.\d+\.\d+\.\d+):(\d+) for MsgId: (\d+)\s*",
    r"stream\.Write MessageLength: (\d+)\s*",
    r"stream\.Written\s*",
    r"DBRequestFinished finished\s*",

    r"(\d+\.\d+\.\d+\.\d+):(\d+) -> RanToCompletion\s*",
    r"\((\d+\.\d+\.\d+\.\d+):(\d+)\) read (\d+) bytes in \((\d{2}:\d{2}:\d{2}.\d{7})\)\s*",
    r"\((\d+\.\d+\.\d+\.\d+):(\d+)\) read complete (\d+) bytes in \((\d{2}:\d{2}:\d{2}.\d{7})\)\s*",

    r"called --> Sender (\d+\.\d+\.\d+\.\d+\.\d+\.\d+)\s*",
    r"Read Message TITemperatureEvent \[Date: (.*), Tempertures: \[CryoVessel_1: (-?\d+\.\d+), CryoVessel_2: (-?\d+\.\d+), CryoVessel_3: (-?\d+\.\d+), RackRoom: (-?\d+), WorkingRoom: (-?\d+\.\d+)\], SamplePosition: \[RackRing: (\d+), Rack: (\d+), Tray: (\d+), Column: (\d+), Row: (\d+), SbsRackColumn: (\d+), SbsRackRow: (\d+), InsertColumn: (\d+), InsertRow: (\d+)\], MessageSize: (\d+)\s*",
    r"Read Message \[Date: (.*), Temperatures:[WorkingRoomCold: (-?\d+\.\d+), WorkingRoomWarm: (-?\d+\.\d+), Interim: (-?\d+\.\d+)], MessageSize:(\d+)]\s*",
    r"(.+):(\d+): (\d+\.\d+\.\d+\.\d+):(\d+) - 'HS200_03028'\s*",
]

"""
r"\s*",


called: HS200_03028

called CheckCoolingScheduleAfterReconnect

GERS229:40004 -> HS200_03028 /// <ASKION_C-LINE_HLI><VERSION V="3.50" /><MESSAGE MES="TSY" NUM="12619" CID="GERS229" REC="HS200_03028" TIM="2024-11-27T00:08:31" /></ASKION_C-LINE_HLI>

GERS229:40004 -> HS200_03028 /// <ASKION_C-LINE_HLI><VERSION V="3.50" /><MESSAGE MES="ACK" NUM="12619" CID="HS200_03028" RES="EOK" OMT="TSY" /></ASKION_C-LINE_HLI> /// <ASKION_C-LINE_HLI><VERSION V="3.50" /><MESSAGE MES="TSY" NUM="12619" CID="GERS229" REC="HS200_03028" TIM="2024-11-27T00:08:31" /></ASKION_C-LINE_HLI>

GERS229:40004: HS200_03028 -> <ASKION_C-LINE_HLI><VERSION V="3.50" /><MESSAGE MES="ACK" NUM="12619" CID="HS200_03028" RES="EOK" OMT="TSY" /></ASKION_C-LINE_HLI> /// <ASKION_C-LINE_HLI><VERSION V="3.50" /><MESSAGE MES="TSY" NUM="12619" CID="GERS229" REC="HS200_03028" TIM="2024-11-27T00:08:31" /></ASKION_C-LINE_HLI>

GERS229:40004 -> HS200_03028 /// <ASKION_C-LINE_HLI><VERSION V="3.50" /><MESSAGE MES="PUL" NUM="12620" CID="GERS229" REC="HS200_03028" /><USERS><USER NAM="ASKION" PWD="BSXN" SIL="99" /><USER NAM="ADMIN" PWD="@ELHO" SIL="99" /><USER NAM="PK" PWD="QJ" SIL="99" /><USER NAM="A" PWD="@" SIL="99" /><USER NAM="B" PWD="C" SIL="99" /><USER NAM="Z" PWD="[" SIL="99" /><USER NAM="Ako" PWD="`jn" SIL="49" /><USER NAM="T" PWD="U" SIL="49" /><USER NAM="C" PWD="B" SIL="99" /><USER NAM="Y" PWD="X" SIL="99" /></USERS></ASKION_C-LINE_HLI>

GERS229:40004 -> HS200_03028 /// <ASKION_C-LINE_HLI><VERSION V="3.50" /><MESSAGE MES="ACK" NUM="12620" CID="HS200_03028" RES="EOK" OMT="PUL" /></ASKION_C-LINE_HLI> /// <ASKION_C-LINE_HLI><VERSION V="3.50" /><MESSAGE MES="PUL" NUM="12620" CID="GERS229" REC="HS200_03028" /><USERS><USER NAM="ASKION" PWD="BSXN" SIL="99" /><USER NAM="ADMIN" PWD="@ELHO" SIL="99" /><USER NAM="PK" PWD="QJ" SIL="99" /><USER NAM="A" PWD="@" SIL="99" /><USER NAM="B" PWD="C" SIL="99" /><USER NAM="Z" PWD="[" SIL="99" /><USER NAM="Ako" PWD="`jn" SIL="49" /><USER NAM="T" PWD="U" SIL="49" /><USER NAM="C" PWD="B" SIL="99" /><USER NAM="Y" PWD="X" SIL="99" /></USERS></ASKION_C-LINE_HLI>

"""


def parameterize_events(events, patterns):
    """
    parameterizes a list of events. transforms a list of events as returned by parse_logfile and for every event
    replaces "log_message" with "log_key" and "parameters"
    """
    new_events = []

    for e in events:
        parameterized_message = e["log_message"]
        parameters = []
        # try every pattern until one is successful then
        for pattern in patterns:
            m = parameterize_message(pattern, e["log_message"])
            if m is not None:
                parameterized_message = m[0]
                parameters = m[1]
                break
        
        # try to convert parameters that are ints, floats or bools
        for param in parameters:
            if param == "True":
                param = True
                continue
            if param == "False":
                param = False
                continue
            try:
                if "." in param:
                    param = float(param)
                else:
                    param = int(param)
                continue
            except (ValueError, TypeError):
                pass
        
        new_event = {
            "timestamp": e["timestamp"],
            "log_level": e["log_level"],
            "class": e["class"],
            "log_key": parameterized_message,
            "parameters": parameters
        }
        
        new_events.append(new_event)
    
    return new_events

# Generated by ChatGPT, adjusted by me
def parameterize_message(pattern, text) -> None|tuple[str|list[str]]:
    """
    parameterizes a text based on a regex pattern. If there is no match between them None is returned. 
    Otherwise the text will be returned with all parameters replaced by place holders 
    as well as a list of the corresponding parameters. 

    Args:
        pattern (str): the pattern to parameterize by. Groups will be treated as parameters.
        text (str): The text to be parameterized
    
    Returns:
        input text with parameters replaced by place holders
        list of parameters
    """
    match = re.search(pattern, text)
    if not match:
        return None

    if match.span() != (0, len(text)):
        return None
    
    groups = list(match.groups())
    
    # Replace the match in the original string
    replaced = text

    for i, group in enumerate(groups, start=1):
        if group is not None:
            # Use re.escape to safely replace the exact match
            replaced = replaced.replace(group, f"<x{i}>", 1)

    # Reconstruct the final string
    return replaced, groups

def parse_logfile(path: str):
    """
    Parse a logfile into a list of events. Each event has the following keys: "timestamp", "log_level", "class", "log_message"

    Args:
        path (str): The path to the log file.
    
    Returns:
        a list of event dictionaries
    """
    with open(path, "r") as f:
        lines = f.readlines()

    progress = tqdm(total =len(lines), desc="parsing log file")

    events = []
    while lines:
        line = lines[0].strip("\n ")

        # if this is the start of an event, create a new event
        parts = line.split(" | ")
        if len(parts) >= 4 and re.search(r"^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{4}", line):
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
        progress.update(1)
    
    progress.close()
    return events

def extract_messages(events, out_file: str):
    """
    extract the log messages from a list of log events as returned by parse_logfile() and save them to a file.

    Args:
        events (list): a list of events as returned by parse_logfile()
        out_file (str): The path to the file the messages should be saved to
    """
    
    with open(out_file, "w") as f:
        for i, e in tqdm(enumerate(events), total=len(events), desc="extracting messages"):
            f.write(e["log_message"].replace("\n", " ")+"\n")

# ChatGPT generated, fixed myself...
def extract_log_keys(input_path: str, output_path: str, patterns):
    """
    Extract log keys from a file with extracted messages into a new file. Lines that couldn't find a fitting pattern will be kept the same.
    Unprocessed lines and log keys will be seperated by an empty line. 

    Args:
        input_path (str): Path to the input .txt file.
        output_path (str): Path to save the processed file.
        patterns (list): A list of replacement patterns to look for
    """
    with open(input_path, 'r') as infile:
        line_cnt = sum(1 for _ in infile)

    # copy for better performance
    pattern_list = patterns[:]
    
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        replaced = set()
        out_lines = [""]
        for i, line in tqdm(enumerate(infile), total=line_cnt, desc="extracting log keys"):
            # print(line)
            needs_to_print = True
            for idx, pattern in enumerate(pattern_list):
                m = parameterize_message(pattern, line)

                if m is not None:
                    # Move matched pattern to the front
                    if idx != 0:
                        pattern_list.insert(0, pattern_list.pop(idx))

                    if pattern not in replaced:
                        out_lines.insert(0, m[0])
                        replaced.add(pattern)
                    needs_to_print = False
                    break
            if needs_to_print:
                print(line)
                out_lines.append(line)

        outfile.writelines(out_lines)

