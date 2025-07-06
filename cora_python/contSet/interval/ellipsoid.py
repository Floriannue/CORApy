import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def ellipsoid(I, mode='outer'):
    """
    Converts an interval object into an ellipsoid object.
    
    Syntax:
        E = ellipsoid(I)
        E = ellipsoid(I,mode)
    
    Inputs:
        I - interval object
        mode - str, 'outer' or 'inner'
    
    Outputs:
        E - ellipsoid object
    """

    # Check if not a matrix set
    n = I.dim()
    if hasattr(n, '__len__') and len(n) > 1:
        raise CORAerror('CORA:wrongValue', 'first', 'Interval must not be an n-d array with n > 1.')

    if mode.startswith('outer'):
        # outer approximation
        
        # avoid numerical issues, enlarge very small non-degenerate dimensions
        rad_min = 5e-3
        rad_cur = I.rad()

        factor = np.ones_like(rad_cur.flatten())
        mask = rad_cur.flatten() < rad_min
        factor[mask] = rad_min / rad_cur.flatten()[mask]
        factor[np.isinf(factor)] = 1
        
        I_enlarged = I.enlarge(factor)
        
        # convert interval to zonotope (exact computation very efficient because
        # generator matrix is square)
        Z = I_enlarged.zonotope()
        E = Z.ellipsoid('outer:exact')

    elif mode.startswith('inner'):
        # inner approximation
        
        # convert interval to zonotope (exact computation very efficient because
        # generator matrix is square)
        Z = I.zonotope()
        E = Z.ellipsoid('inner:exact')
    
    else:
        raise CORAerror('CORA:wrongValue', 'second', f"Unknown mode '{mode}'")

    return E 