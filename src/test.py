import sys
import os
sys.path.append(os.path.abspath("../src"))
from parse import *
n=606
extract_log_keys(f'./data/messages{n}.txt', f'./data/messages_types{n}.txt', PATTERNS)