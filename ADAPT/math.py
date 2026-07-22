"""
For mathematical operations
"""

import torch

def quadratic_interp(x_query, x_known, y_known):
    n = len(x_known)

    # Flatten y_known to 1D for safe indexing
    y_known = y_known.flatten()

    # Find insertion position
    idx = torch.searchsorted(x_known, x_query, right=True)

    # Clamp idx so we can take a neighbor on each side
    idx = torch.clamp(idx, 1, n - 2)

    # Left, mid, right points for quadratic fit
    idx0 = idx - 1
    idx1 = idx
    idx2 = idx + 1

    # Gather values
    x0, x1, x2 = x_known[idx0], x_known[idx1], x_known[idx2]
    y0, y1, y2 = y_known[idx0], y_known[idx1], y_known[idx2]

    # Lagrange basis
    L0 = ((x_query - x1) * (x_query - x2)) / ((x0 - x1) * (x0 - x2))
    L1 = ((x_query - x0) * (x_query - x2)) / ((x1 - x0) * (x1 - x2))
    L2 = ((x_query - x0) * (x_query - x1)) / ((x2 - x0) * (x2 - x1))

    return y0 * L0 + y1 * L1 + y2 * L2