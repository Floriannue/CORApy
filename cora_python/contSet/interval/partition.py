import numpy as np
import itertools
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def partition(I, splits):
    """
    Partitions a multidimensional interval into subintervals.
    """
    if not (isinstance(splits, (int, float)) and splits > 0):
        raise CORAerror('CORA:wrongValue', 'second', "must be a positive scalar")

    incr = (I.sup - I.inf) / splits
    inf = I.inf

    # Get all combinations of indices
    indices = itertools.product(range(splits), repeat=I.dim())

    dzNew = []
    for current_indices in indices:
        current_indices = np.array(current_indices).reshape(-1, 1)
        lower_bound = inf + current_indices * incr
        upper_bound = inf + (current_indices + 1) * incr
        dzNew.append(Interval(lower_bound, upper_bound))

    return dzNew 