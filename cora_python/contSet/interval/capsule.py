import numpy as np
from cora_python.contSet.capsule.capsule import Capsule
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def capsule(I):
    """
    Encloses an interval with a capsule
    """

    n = I.dim()
    if isinstance(n, (list, np.ndarray)) and len(n) > 1:
        raise CORAerror('wrongValue', 'first', 'Interval must not be an n-d array with n > 1.')

    # Handle empty interval case
    if I.isemptyobject():
        # Return empty capsule with appropriate dimensions
        c = np.zeros((n, 1))
        g = np.zeros((n, 1))
        r = 0
        return Capsule(c, g, r)

    width = I.rad()
    ind = np.argsort(width)[::-1]

    g = np.zeros((n, 1))
    g[ind[0]] = width[ind[0]]
    
    int_ = I.project(ind[1:])
    r = int_.radius()

    C = Capsule(I.center(), g, r)
    
    return C 