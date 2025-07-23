"""
interval - overapproximates a zonotope by an interval

Syntax:
   I = interval(Z)

Inputs:
   Z - zonotope object

Outputs:
   I - interval object

Example: 
   Z = zonotope([-1;1],[3 2 -1; 2 1 -1]);
   I = interval(Z);

   figure; hold on;
   plot(Z,[1,2],'b');
   plot(I,[1,2],'r');

Other m-files required: interval (constructor)
Subfunctions: none
MAT-files required: none

See also: ---

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 14-September-2006 (MATLAB)
Python translation: 2025
"""

import numpy as np


def interval(Z):
    """
    Overapproximates a zonotope by an interval
    
    Args:
        Z: zonotope object
        
    Returns:
        Interval object
    """
    # Import here to avoid circular imports
    from cora_python.contSet.interval import Interval
    
    # Handle empty zonotope: return truly empty interval of correct dimension
    if hasattr(Z, 'isemptyobject') and Z.isemptyobject():
        from cora_python.contSet.interval.empty import empty as interval_empty
        from cora_python.contSet.zonotope.dim import dim as zonotope_dim
        n = zonotope_dim(Z)
        return interval_empty(n)

    # extract center
    c = np.asarray(Z.c).flatten()  # ensure 1D as in MATLAB
    
    # determine lower and upper bounds in each dimension
    # sum(abs(Z.G),2) in MATLAB sums along columns (axis=1 in Python)
    delta = np.sum(np.abs(Z.G), axis=1)  # 1D array, shape (n,)
    leftLimit = c - delta
    rightLimit = c + delta
    
    # instantiate interval (keep original shape)
    I = Interval(leftLimit, rightLimit)
    
    return I 