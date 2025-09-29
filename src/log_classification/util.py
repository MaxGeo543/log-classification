from typing import Any, List

import numpy as np
import tensorflow as tf
import torch
import os
import re
import hashlib
import base64
import numpy as np

def to_flat_array(x: Any) -> np.ndarray:
    """
    Converts scalar or array-like into 1D NumPy array
    """
    if isinstance(x, tf.Tensor):
        return x.numpy().reshape(-1)
    elif isinstance(x, torch.Tensor):
        return x.numpy().reshape(-1)
    elif isinstance(x, np.ndarray):
        return x.reshape(-1)
    elif isinstance(x, np.generic):
        return np.array([x.item()])
    elif isinstance(x, (int, float)):
        return np.array([x])
    elif isinstance(x, (list, tuple)):
        return np.array(x)
    else:
        raise ValueError(f"Unsupported type: {type(x)}")

def get_sorted_log_numbers_by_size(directory: str) -> List[int]:
    """
    For a directory containing log files formatted like "CCLog-backups.<n>.log" ectract all n in a list of integers sorted by the size of the logs
    """
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

def hash_list_to_string(str_list: List[str], length: int) -> str:
    """
    Hashes a list of strings into a fixed‐length string.
    
    :param str_list: List of input strings.
    :param length: Desired length of output string.
    :return: A string of exactly `length` characters.
    """
    # 1) Create hash and feed in each string
    h = hashlib.sha256()
    for s in str_list:
        h.update(s.encode('utf-8'))
    
    # 2) Base64‐encode the raw digest, URL‐safe, strip padding
    b64 = base64.urlsafe_b64encode(h.digest()).decode('ascii').rstrip('=')
    
    # 3) If that’s already long enough, just truncate
    if len(b64) >= length:
        return b64[:length]
    
    # 4) Otherwise, extend by re-hashing with a salt
    #    until we have enough characters
    extra = b64
    counter = 0
    while len(extra) < length:
        counter += 1
        h2 = hashlib.sha256()
        # use the original digest + a counter as “salt”
        h2.update(h.digest() + counter.to_bytes(4, 'big'))
        extra += base64.urlsafe_b64encode(h2.digest()).decode('ascii').rstrip('=')
    
    return extra[:length]

def hash_ndarray(arr: np.ndarray) -> int:
    """
    Hashes an np.ndarray. 
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a NumPy ndarray.")
    if arr.ndim != 1:
        raise ValueError("Input must be a 1D array.")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("Array must contain integers.")

    return hash(tuple(arr))