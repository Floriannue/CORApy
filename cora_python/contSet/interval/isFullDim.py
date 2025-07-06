import numpy as np

def isFullDim(I):
    """
    Checks if the dimension of the affine hull of an interval is
    equal to the dimension of its ambient space.
    """
    if I.is_empty():
        return False
    else:
        # An interval is full-dimensional if no radius is close to zero.
        return not np.any(np.isclose(I.rad(), 0)) 