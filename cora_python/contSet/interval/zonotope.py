import numpy as np

def zonotope(I):
    """
    Converts an interval object into a zonotope object.
    
    Syntax:
        Z = zonotope(I)
    
    Inputs:
        I - interval object
    
    Outputs:
        Z - zonotope object
    
    Example:
        I = interval([1;-1], [2; 1]);
        Z = zonotope(I);
    
    Other m-files required: none
    Subfunctions: none
    MAT-files required: none
    
    See also: interval, polytope
    """
    from cora_python.contSet.zonotope.zonotope import Zonotope
    from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
    
    # check if not a matrix set
    n = I.dim()
    # MATLAB: if numel(n) > 1, raise error (interval matrices not supported)
    # In Python, dim() can return int (for vectors) or list (for matrices)
    # Only raise error if n is an array/list with multiple elements
    if isinstance(n, (list, tuple)) and len(n) > 1:
        raise CORAerror('CORA:wrongValue', 'first', 'Interval must not be an n-d array with n > 1.')
    # Also check for numpy array case
    if isinstance(n, np.ndarray) and n.size > 1:
        raise CORAerror('CORA:wrongValue', 'first', 'Interval must not be an n-d array with n > 1.')
    
    # obtain center
    c = I.center()
    
    # construct generator matrix G
    r = I.rad()
    G = np.diag(r.flatten())
    
    # Remove columns where radius is zero
    non_zero_indices = r.flatten() != 0
    G_filtered = G[:, non_zero_indices]
    
    # instantiate zonotope: [center, generators]
    if G_filtered.shape[1] > 0:
        # Reshape center to column vector to match G_filtered dimensions
        c_col = c.reshape(-1, 1)
        Z = Zonotope(np.hstack([c_col, G_filtered]))
    else:
        # If no generators, just use center
        Z = Zonotope(c.reshape(-1, 1))
    
    return Z 