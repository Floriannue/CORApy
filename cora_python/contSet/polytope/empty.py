# This file is part of the CORA project.
# Copyright (c) 2024, System Control and Robotics Group, TU Graz.
# All rights reserved.

import numpy as np
from cora_python.contSet.polytope import Polytope

def empty(n: int = 0) -> Polytope:
    """
    Instantiates an empty n-dimensional polytope.

    An empty polytope can be represented by an infeasible constraint,
    such as 0*x <= -1.

    Args:
        n: The dimension of the polytope.

    Returns:
        An empty n-dimensional Polytope object.
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError("Dimension n must be a non-negative integer.")

    # the polytope 0*x <= -1 is empty (following MATLAB implementation)
    nrRows = min([n, 1])
    A = np.zeros((nrRows, n))
    b = -np.ones((nrRows, 1))
    
    # Create polytope with infeasible constraint
    P_out = Polytope(A, b)
    
    # Set properties explicitly like MATLAB does
    P_out._emptySet = True
    P_out._bounded = True
    P_out._fullDim = False
    P_out._minHRep = True
    P_out._minVRep = True
    P_out._V = np.zeros((n, 0))
    P_out._isVRep = True
    
    return P_out 