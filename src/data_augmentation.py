from typing import Tuple
import numpy as np

def remove_and_pad(x: np.ndarray, y: np.ndarray, min_rem: int = None, max_rem: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    removes randomly chosen entries from a window of data and then pads the window with zeros

    :params x: A single window from X data
    :params y: The label of the X data
    :params min_rem: the minimum number of entries to remove from the window
    :params max_rem: the maximum number of entries to remove from the window
    :returns: The same x but with entries removed and padded, and the unchanged y 
    """
    if x.ndim != 2:
        raise ValueError("x must be a 2D array of shape (n, f).")
    n, f = x.shape

    # Handle trivial cases early
    if n == 0:
        return x

    # Defaults based on n
    if min_rem is None:
        min_rem = 1
    if max_rem is None:
        max_rem = n // 2

    # Clamp to valid range
    min_rem = max(0, min_rem)
    max_rem = max(0, max_rem)
    max_rem = min(max_rem, n)  # can't remove more rows than exist

    # Ensure min_rem <= max_rem; if not, make them equal
    if min_rem > max_rem:
        min_rem = max_rem

    # If nothing to remove, just return x unchanged
    if max_rem == 0:
        return x, y

    rng = np.random.default_rng()
    # randint-like with inclusive upper bound: use integers(low, high+1)
    r = int(rng.integers(min_rem, max_rem + 1))

    if r == 0:
        return x, y

    # Choose r distinct row indices to remove
    remove_idx = rng.choice(n, size=r, replace=False)

    # Keep the complement
    mask = np.ones(n, dtype=bool)
    mask[remove_idx] = False
    x_kept = x[mask]

    # Pad with r zero-rows at the end
    pad = np.zeros((r, f), dtype=x.dtype)
    x_out = np.vstack((x_kept, pad))
    return x_out, y
