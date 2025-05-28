import numpy as np
import tensorflow as tf

def to_flat_array(x):
    # Converts scalar or array-like into 1D NumPy array
    if isinstance(x, tf.Tensor):
        return x.numpy().reshape(-1)
    elif isinstance(x, np.ndarray):
        return x.reshape(-1)
    elif isinstance(x, np.generic):
        return np.array([x.item()])
    elif isinstance(x, (int, float)):
        return np.array([x])
    else:
        raise ValueError(f"Unsupported type: {type(x)}")