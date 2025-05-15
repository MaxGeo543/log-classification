import re
from collections import defaultdict
from tqdm.notebook import tqdm

LOG_PATH = "C:/Users/Askion/Documents/agmge/log-classification/data/selected"
PATTERNS = [
    # HliSession
    r"HliSession::Receive\(\): (\d+\.\d+\.\d+\.\d+):(\d+) - '(.*)' -> (\d*)\s*",
    r"HliSession::Dispose\(\): (\d+\.\d+\.\d+\.\d+):(\d)+ - '(.*)' ->\s?(.*)\s*",
    r"HliSession::Close\(\): (\d+\.\d+\.\d+\.\d+):(\d+) - '(.*)'\s*",
    r"HliSession::Send\(\): (\d+\.\d+\.\d+\.\d+):(\d+) - '.*' ->\s?(.*)\s*",
    r"HliSession::CloseInternal\(\): Close (\d+\.\d+\.\d+\.\d+):(\d+) - '(.*)' ->\s*",
    r"HliSession::CloseInternal\(\): Close (\d+\.\d+\.\d+\.\d+):(\d+) - '(.*)' ->\s?(.*)\s*",
    r"HliSession::CloseInternal\(\): (\d+\.\d+\.\d+\.\d+):(\d+) - '(.*)' ->\s?(.*)\s*",
    r"HliSession::OnSessionClosed\(\): (\d+\.\d+\.\d+\.\d+):(\d+) - '(.*)'\s*",
    # HliServer
    r"HliServer::Dispose\(\): (.+):(\d+) -> (.*)\s*",
    # SessionFactory
    r"SessionFactory::OpenSession\(\): (\d+\.\d+\.\d+\.\d+):(\d+) - '(.*)' ->\s?(.*)\s*",
    r"SessionFactory::OpenSession\(\): End: (\d+\.\d+\.\d+\.\d+):(\d+) - '(.*)' ->\s?(.*)\s*",
    r"SessionFactory::OpenSession\(\): (\d+\.\d+\.\d+\.\d+):(\d+) - '(.*)'\s*",

    # fixed messages (no parameters)
    r"End\s*",
    r"called\s*",
    r"path not found\s*",
    r"Start Deleting EMSMessages\s*",
    r"\[is not a valid DBCommand object!\]\s*",
    r"Calling DBRequestFinished\s*",
    r"entered\s*",
    r"stream\.Written\s*",
    r"DBRequestFinished finished\s*",
    r"Message set to Response\s*",
    r"HS200L_REFERENZ\s*",
    r"ExKrA_Ref\s*",
    r"Stop\s*",
    r"Start\s*",
    r"LV7005072003\s*",
    r"ReportBatch: called\s*",
    r"Connected to DB successfully\.\s*",
    r"CCServerAppContext::Process_CustomSearchSamples\(\): called\s*",
    # prc
    r"prc_md_cachependingpush\s*",
    r"prc_trackingerror_new\s*",
    r"prc_md_ems_message_del\s*",
    r"prc_user_login\s*",
    r"prc_update_dewar_timer\s*",
    r"prc_md_wizardsettings\s*",
    r"prc_md_cachependingcdi_new\s*",
    r"prc_v_device\s*",
    r"prc_dewar_use_new\s*",
    r"prc_trackinghistory_device_new\s*",
    r"prc_system_message_new\s*",
    r"prc_md_cachependingcdi_del\s*",
    r"prc_dewar_use_del\s*",
    r"prc_batch_finished\s*",
    r"prc_getLastSampleAction\s*",
    r"prc_reset_storage\s*",
    r"prc_batch_del\s*",
    
    # SQL
    r"\s*SELECT (.*) FROM (.*) WHERE (.*)\s*",
    r"\s*SELECT (.*) FROM (.*) where (.*)\s*",
    r"\s*SELECT (.*) from (.*) WHERE (.*)\s*",
    r"\s*SELECT (.*) from (.*) where (.*)\s*",
    r"\s*Select (.*) from (.*) where (.*)\s*",
    r"\s*Select (.*) from (.*)\s*",
    r"\s*SELECT (.*) from (.*)\s*",
    r"\s*SELECT (.*) FROM (.*)\s*",
    r"\s*SELECT (.*)\s*",
    r"\s*Select (.*)\s*",

    # IP Adresses
    r"(.+):(\d+) -> (.+) - '(.*)'\s*",
    r"(.+):(\d+) -> (.+) - '(.*)' ///\s?(.*)\s*",
    r"(.+):(\d+) -> (.+)\s*",
    r"(.+):(\d+) -> (.+) ///\s?(.*)\s*",
    r"(.+):(\d+)\s*",
    r"(.+):(\d+) ->\s*",
    r"(.+):(\d+) - '(.*)' ->\s?(.*)\s*",
    r"(.+):(\d+) - '(.*)'\s*",
    r"(\d+\.\d+\.\d+\.\d+):(\d+) -> (.*)\s*",
    r"\((\d+\.\d+\.\d+\.\d+):(\d+)\) read (\d+) bytes in \((\d{2}:\d{2}:\d{2}.\d{7})\)\s*",
    r"\((\d+\.\d+\.\d+\.\d+):(\d+)\) read complete (\d+) bytes in \((\d{2}:\d{2}:\d{2}.\d{7})\)\s*",
    #
    r"(.+):(\d+): (\d+\.\d+\.\d+\.\d+):(\d+) - 'HS200_03028'\s*",
    r"(.+):(\d+): (.+) -> (.+)\s*",
    
    # Reads
    r"Read (\d+) bytes in \((\d{2}:\d{2}:\d{2}.\d{7})\)\s*",
    r"Start: buffer.Length: (\d+), timeout: (\d+)\s*",
    r"buffer\.Length: (\d+)\s*",
    r"Start -> Timeout: (\d+) ms\s*",
    r"End -> Bytes read: (\d+)\s*",
    # Status
    r"TaskStatus: (.*)\s*",
    r"(\d+\.\d+\.\d+\.\d+\.\d+\.\d+)-->State: (True|False)\s*",
    r"(False|True) = (False|True)\s*",
    r"(True|False)\s*",
    r"LV7005072003 -->StorageState: (.*)\s*",

    # Classes
    r"System\.Net\.Sockets\.TcpClient\s*",
    r"CLineCommon_CCIMessage\.(\w+)\s*",
    r"CLineCommon_CCIMessage\.(\w+) -> Askion\.CLine\.Common\.Security\.AuthenticateAttribute\s*",
    r"Askion\.CLine\.(.+)\s*",
    r"C_line_Common\.(.+)\s*",
    # C_line_Control_Server
    r"C_line_Control_Server\.CDIClientManager\s*",
    r"C_line_Control_Server\.Services\.PushService\s*",
    r"C_line_Control_Server\.Services\.PushService\s*",
    #
    r"C_line_Control_Server\.(.+) -> <\?xml.*\?>(.*)\s*",
    r"C_line_Control_Server\.(.+) ->\s?(.*)\s*",
    
    # Messages
    r"Read Message TITemperatureEvent \[Date: (.*), Tempertures: \[CryoVessel_1: (-?\d+\.?\d*), CryoVessel_2: (-?\d+\.?\d*), CryoVessel_3: (-?\d+\.?\d*), RackRoom: (-?\d+\.?\d*), WorkingRoom: (-?\d+\.?\d*)\], SamplePosition: \[RackRing: (\d+), Rack: (\d+), Tray: (\d+), Column: (\d+), Row: (\d+), SbsRackColumn: (\d+), SbsRackRow: (\d+), InsertColumn: (\d+), InsertRow: (\d+)\], MessageSize: (\d+)\s*",
    r"Read Message \[Date: (.*), Temperatures:\[WorkingRoomCold: (-?\d+\.?\d*), WorkingRoomWarm: (-?\d+\.?\d*), Interim: (-?\d+\.?\d*)\], MessageSize:(\d+)\]\s*",
    #
    r"Read Message SystemMessage \[Date: (.*), MessageID: (.*), Message: (.*), ResetMessage:\d Button1: , Button2: , Button3: , Category: (.*), Size: (\d+)\s*",
    r"SystemMessage \[Date: (.*), MessageID: (.*),  Size: (\d+)\s*",
    #
    r"Device: (.*) Message: \[Rack1: (.*); Rack2: (.*), SampleFormat: (.*), Operator: (.*); TRGBarcode: (.*); Storagemode: (.*) MessageSize: (\d+)\s*",
    #
    r"Message ->\[Rack1: (.*); Rack2: (.*), SampleFormat: (.*), Operator: (.*); TRGBarcode: (.*); Storagemode: (.*) MessageSize: (\d+)\s*",
    r"Message ->\[Date: (.*), MessageID: (.*), PailID: (.*), ExKrADestinations: \[\[DestinationID: (.*), DestinationType: (.*), EstimatedDuration: (.*), Sequence: (.*)]; \[DestinationID: (.*), DestinationType: (.*), EstimatedDuration: (.*), Sequence: (.*)\]\] MessageSize: (.*)\]\s*",
    #
    r"\[DewarName: (.*), Position: (.*), State: (.*), MessageSize: (\d+)\]\s*",
    #
    r"MessageID: (\d+) MessageType: (.+)\s*",

    # calls and updates
    r"update (.*) with (.*)\s*",
    r"called:\s?(.+)\s*",
    r"called (.+)\s*",
    r"called, path:(.*) delemiter:(.*), decimalseparator:(.*)\s*",
    r"called --> Sender (\d+\.\d+\.\d+\.\d+\.\d+\.\d+)\s*",
    
    # xml
    r"(Send|Receive)\s?:\s?\?+<\?xml.*\?>(.*)\s*",
    r"\?+<\?xml.*\?>(.*)\s*",

    # Exceptions
    r"Exception \((\d+) ms\): ConnectionReset\s*",
    r"Exception \((\d+) ms\): TimedOut\s*",
    r"catched SocketException -> SendQueueMessage\(\) get cancelled; SocketErrorCode: ConnectionRefused\s*",
    r"System\.AggregateException: One or more errors occurred\. --->\s?(.*)\s*",
    r"Exception: System\.AggregateException: One or more errors occurred\. --->\s?(.*)\s*",
    r"(.+):(\d+) Exception: System\.AggregateException: One or more errors occurred. --->\s?(.*)\s*",
    r"(.+):(\d+) System\.AggregateException: One or more errors occurred. --->\s?(.*)\s*",
    r"System.ObjectDisposedException: .*\s*",

    # misc
    r"(\d+)\s*", # number
    r"{(\d+)}\s*", # number in curly brackets
    r"(\d) -> (\d)\s*", # digit -> digit
    r"(\w):\\(.*)\s*", # path
    r"Return (.*)\s*",
    r"timer removed: (.*)\s*",

    # dewar
    r"Stop --- finished (.+) ---\s*",
    r"differnceHours: (-?\d+\.?\d*) for dewarTimer: (.+)\s*",
    r"UpdateDewarTimer for dewar: --> (.*)\s*"
    r"send Dewar EMS for: (.*)\s*",

    
    
    
    r"C-line Control Server, Version=(.+), Culture=(.+), PublicKeyToken=(.+) started \[(\w):\\(.*)C-line Control Server\.exe\]\s*",
    r"ClinePrincipal: Identity \[ClineIdentity: UserId \[(\d+)\], Name \[(.*)\], AuthenticationType \[(.*)\]\], HSLevel \[(\d+)\] -> CLineCommon_CCIMessage\.(\w+)\s*",
    r"ClinePrincipal: Identity \[ClineIdentity: UserId \[(\d+)\], Name \[(.*)\], AuthenticationType \[(.*)\]\], HSLevel \[(\d+)\] -> CLineCommon_CCIMessage\.(\w+) \((\d+)\)\s*",
    r"HS Level : \(ushort\)principal.HermeticStorageLevel\s*",
    # Client
    r"Got Client: (\d+\.\d+\.\d+\.\d+):(\d+) for MsgId: (\d+)\s*",
    r"Feiled to get client for MsgId: (\d+)\s*",
    r"Clients: Push (.*):(.*)\s*",
    r"\[Date: (.*), UserName: (.+)\]\s*",
    
    r"cache pending: HLIAcks\[id: (\d*), HliClient: (.*), HliMessageID: (\d*), CdiMessageID: (\d*), SampleID: (\d*), StorageID: (\d*), UserID: (\d*), HliMessageLength: (\d*)]\s*",
    r"stream\.Write MessageLength: (\d+)\s*",
    r"check ExkraId: (\d+)\s*"
    r"Send -> CDIGetCachingSBS To -> (.*):(\d+) Response -> \[Rack1: (.*); Rack2: (.*), SampleFormat: (.*), Operator: (.*); TRGBarcode: (.*); Storagemode: (.*) MessageSize: (\d+)\s*",
    r"Send -> NewJobRequest To -> (.*):(\d+) Response -> \[Date: (.*), OrderID: (.*), ExKraError: (.*) MessageSize: (\d+)\]\s*",
    r"operation '(.*)' \((.*)\) -> user 'A' \(Create, Edit, Depleted, Delete, Details, Store, Retrieve, Freeze, Search, Reporting\)\s*",
]





"""
r"\s*",
MessageID: 27050 MessageType: DBSystemMessageList

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
        line = lines[0].strip("\n")

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

