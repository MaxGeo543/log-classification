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
    elif isinstance(x, (list, tuple)):
        return np.array(x)
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


    """
    Decide whether labels are sparse (class ids) or one-hot.
    Returns: dict(mode, num_classes, loss, metric), and possibly
    transformed y (squeezed for sparse).
    """
    y = np.asarray(y)

    # Column vector of class ids -> squeeze to (N,)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)

    # One-hot: 2D, >1 columns, values in {0,1}, rows sum to 1
    if y.ndim == 2 and y.shape[1] > 1:
        is_binary = np.isin(y, (0, 1)).all()
        row_sums_one = np.allclose(y.sum(axis=1), 1.0)
        if is_binary and row_sums_one:
            return {
                "mode": "one_hot",
                "num_classes": y.shape[1],
                "loss": "categorical_crossentropy",
                "metric": "accuracy",
                "y": y,
            }

    # Sparse: class ids (ints) in 1D
    if y.ndim == 1:
        # (allow float labels that are whole numbers, but cast to int)
        if not np.allclose(y, np.round(y)):
            raise ValueError("Labels look dense/continuous; expected class ids or one-hot.")
        y = y.astype("int32")
        return {
            "mode": "sparse",
            "num_classes": int(np.max(y)) + 1,
            "loss": "sparse_categorical_crossentropy",
            "metric": "sparse_categorical_accuracy",
            "y": y,
        }

    raise ValueError(
        f"Could not infer label format from shape {y.shape}. "
        "Expected (N, C) one-hot or (N,) / (N,1) class ids."
    )