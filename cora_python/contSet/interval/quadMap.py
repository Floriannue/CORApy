from .interval import Interval

def quadMap(I, *varargin):
    """
    Computes the quadratic map of an interval by converting it to a zonotope,
    applying the zonotope's quadMap, and converting back to an interval.
    
    Syntax:
        res = quadMap(I1, Q)
        res = quadMap(I1, I2, Q)
    
    Inputs:
        I - interval object
        I1 - interval object
        I2 - interval object (optional)
        Q - quadratic coefficients as a list of matrices
    
    Outputs:
        res - interval object
    
    Example:
        Z = zonotope([0, 1, 1; 0, 1, 0])
        I = interval(Z)
        Q = [[[0.5, 0.5], [0, -0.5]], [[-1, 0], [1, 1]]]
        res = quadMap(I, Q)
    """
    nargin = len(varargin) + 1  # +1 for I

    if nargin == 2:
        Q = varargin[0]
        # Convert interval to zonotope, then call quadMap
        Z = I.zonotope()
        res_zono = Z.quadMap(Q)
    elif nargin == 3:
        I2, Q = varargin
        # Convert intervals to zonotopes, then call quadMap
        Z1 = I.zonotope()
        Z2 = I2.zonotope()
        res_zono = Z1.quadMap(Z2, Q)
    else:
        raise ValueError("Invalid number of input arguments")

    # Enclose the resulting zonotope with an interval
    return Interval(res_zono) 