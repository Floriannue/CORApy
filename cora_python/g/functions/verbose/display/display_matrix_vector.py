import numpy as np
from cora_python.g.macros import DISPLAYDIM_MAX
from cora_python.contSet.interval import Interval
# from cora_python.matrixSet.intervalMatrix import IntervalMatrix # Assuming this exists
# from cora_python.matrixSet.matZonotope import MatZonotope # Assuming this exists

def display_matrix_vector(x, varname):
    """
    Displays a matrix or vector on the command window
    up to a certain maximum size and abbreviated when all-zero.
    """
    res = []
    
    # TODO: Placeholder for intervalMatrix and matZonotope when they are translated
    # For now, we handle numpy arrays and Intervals
    
    is_interval = isinstance(x, Interval)
    is_numpy = isinstance(x, np.ndarray)

    if not is_interval and not is_numpy:
        # Fallback for untranslated types
        return f"{varname} = [unsupported type: {type(x).__name__}]"

    if is_numpy and x.size == 0:
        return f"\n{varname} = []\n"
    elif is_interval and x.inf.size == 0:
        return f"\n{varname} = []\n"

    x_size = x.shape
    text = "matrix" if len(x_size) > 1 and x_size[1] > 1 else "vector"
    if is_interval:
        text = "interval " + text

    # All-zero check
    all_zero = False
    if is_numpy:
        all_zero = not np.any(x)
    elif is_interval:
        all_zero = not np.any(x.rad)

    # Identity check
    is_identity = False
    if is_numpy and len(x_size) == 2 and x_size[0] == x_size[1]:
        is_identity = np.all(x == np.eye(x_size[0]))

    if all_zero:
        res.append(f"{varname} = all-zero {x_size[0]}-by-{x_size[1]} {text}")
    elif is_identity:
        res.append(f"{varname} = {x_size[0]}-by-{x_size[1]} identity matrix")
    elif all(s <= DISPLAYDIM_MAX for s in x_size):
        res.append(f"{varname} = \n{x}")
    else:
        res.append(f"{varname} = {x_size[0]}-by-{x_size[1]} {text}")

    return "\n" + "\n".join(res) + "\n" 