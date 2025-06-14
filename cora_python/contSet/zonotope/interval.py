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
    
    # extract center
    c = Z.c
    
    # determine lower and upper bounds in each dimension
    # sum(abs(Z.G),2) in MATLAB sums along columns (axis=1 in Python)
    delta = np.sum(np.abs(Z.G), axis=1, keepdims=True)
    leftLimit = c - delta
    rightLimit = c + delta
    
    # Flatten to 1D arrays for interval constructor
    leftLimit = leftLimit.flatten()
    rightLimit = rightLimit.flatten()
    
    # instantiate interval
    I = Interval(leftLimit, rightLimit)
    
    return I 