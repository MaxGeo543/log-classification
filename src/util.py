import numpy as np
import tensorflow as tf
import torch
import os
import re

def to_flat_array(x):
    # Converts scalar or array-like into 1D NumPy array
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
    else:
        raise ValueError(f"Unsupported type: {type(x)}")

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
