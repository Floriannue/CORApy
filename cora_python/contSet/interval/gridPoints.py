import numpy as np
import itertools
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def gridPoints(I, segments):
    """
    Computes uniformly partitioned grid points of an interval.
    """
    if I.is_empty():
        return np.zeros((I.dim(), 0))

    n_orig, m_orig = I.inf.shape
    int_trans = False
    int_mat = False

    # Ensure all working arrays are flat
    inf_flat = I.inf.flatten()
    sup_flat = I.sup.flatten()
    segments_flat = segments

    if n_orig > 1 and m_orig > 1:
        int_mat = True
        if not np.isscalar(segments):
            segments_flat = np.array(segments).flatten()
    elif m_orig > 1:
        int_trans = True
        # Re-flatten transposed data
        inf_flat = I.inf.T.flatten()
        sup_flat = I.sup.T.flatten()
        segments_flat = np.array(segments).T.flatten()

    n = len(inf_flat)
    if np.isscalar(segments_flat):
        segments_flat = np.full(n, segments_flat)

    if not np.all(segments_flat > 0):
        raise CORAerror('CORA:wrongValue', 'second', "must be larger than 0")
    if len(segments_flat) != n:
        raise CORAerror('CORA:wrongValue', 'second', "must be a scalar or fit the dimension of the interval")

    r = (sup_flat - inf_flat) / 2
    c = inf_flat + r

    if not np.any(r):
        p = c
        if int_mat:
            return {p.reshape(n_orig, m_orig)}
        if int_trans:
            return p.reshape(1, -1)
        return p.reshape(-1, 1)

    seg_eq_1 = (segments_flat == 1) | (r == 0)
    
    # Initialize with zeros to avoid division by zero issues
    seg_length_vec = np.zeros_like(inf_flat, dtype=float)
    
    # Calculate segment length only for valid segments
    valid_segments = segments_flat > 1
    seg_length_vec[valid_segments] = (sup_flat[valid_segments] - inf_flat[valid_segments]) / (segments_flat[valid_segments] - 1)

    starting_point = inf_flat.copy()
    starting_point[seg_eq_1] = c[seg_eq_1]
    segments_flat[seg_eq_1] = 1

    # Use itertools.product to get permutations
    ranges = [range(int(s)) for s in segments_flat]
    perms = np.array(list(itertools.product(*ranges))).T

    p = starting_point[:, np.newaxis] + perms * seg_length_vec[:, np.newaxis]

    if int_trans:
        p = p.T
    
    if int_mat:
        return [p[:, i].reshape(n_orig, m_orig) for i in range(p.shape[1])]

    return p 