import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def polyZonotope(I):
    """
    Convert an interval object to a polynomial zonotope.
    """
    from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope
    # Check if not a matrix set
    if len(I.inf.shape) > 1 and I.inf.shape[1] > 1:
        raise CORAerror('CORA:wrongValue', 'first', 'Interval must not be an n-d array with n > 1.')

    # Get center, radii, and dimension
    c = I.center()
    G = np.diag(I.rad().flatten())
    n = I.dim()
    E = np.eye(n)
    
    # An interval is a special case of a zonotope, which is a special case of a polyZonotope.
    # The conversion results in a polyZonotope with only linear generators.
    # PolyZonotope constructor needs: c, G, GI, E, id (5 arguments)
    GI = np.array([])  # independent generators (empty for interval)
    id = np.array([])  # identifier vector (empty, will be auto-generated)
    return PolyZonotope(c, G, GI, E, id) 