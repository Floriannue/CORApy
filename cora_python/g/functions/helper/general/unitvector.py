import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def unitvector(idx, dim):
    """
    unitvector - creates a standard basis vector

    Syntax:
       v = unitvector(idx, dim)

    Inputs:
       idx - index (1-based) of the component that is 1
       dim - dimension of the vector

    Outputs:
       v - numpy array (column vector) representing the unit vector

    Example:
       # v = unitvector(1, 3) would produce np.array([[1], [0], [0]])

    Authors:       (Assumed MATLAB built-in functionality)
                   Automatic python translation: Florian Nüssel BA 2025
    Written:       --- 
    Last update:   --- 
    Last revision: --- 
    """

    # Validate dim first
    if not isinstance(dim, int) or dim < 1:
        raise CORAerror('CORA:unitvector:invalidDimension', 'Dimension must be a positive integer.')

    # Validate idx against dim
    if not isinstance(idx, int) or idx < 1 or idx > dim:
        raise CORAerror('CORA:unitvector:invalidIndex', 'Index must be a positive integer within the dimension.')

    v = np.zeros((dim, 1))
    v[idx - 1] = 1.0
    return v 