"""
priv_box_V - computes the halfspace representation of the box enclosure
    given a vertex representation

Syntax:
    A, b, empty, fullDim, bounded = priv_box_V(V, n)

Inputs:
    V - vertex representation
    n - dimension of polytope

Outputs:
    A - inequality constraint matrix
    b - inequality constraint offset
    empty - true/false whether result is the empty set
    fullDim - true/false on degeneracy
    bounded - true/false on boundedness

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 03-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Tuple
from cora_python.g.functions.matlab.validate.check import withinTol


def priv_box_V(V: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Computes the halfspace representation of the box enclosure
    given a vertex representation
    
    Args:
        V: Vertex representation (n x nrVertices matrix)
        n: Dimension of polytope
        
    Returns:
        tuple: (A, b, empty) where:
            A - inequality constraint matrix
            b - inequality constraint offset  
            empty - true/false whether result is the empty set
    """
    
    # check for emptiness
    empty = V.size == 0
    if empty:
        A = np.array([]).reshape(0, n)
        b = np.array([]).reshape(0, 1)
        return A, b, empty
    
    # compute lower and upper bound
    ub = np.max(V, axis=1, keepdims=True)  # Keep as column vector
    lb = np.min(V, axis=1, keepdims=True)  # Keep as column vector
    
    # construct constraint matrix and offset
    A = np.vstack([np.eye(n), -np.eye(n)])
    b = np.vstack([ub, -lb])
    
    return A, b, empty 