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
    
    # Create polytope with infeasible constraint, explicitly passing dimension
    P_out = Polytope(A, b, dim=n)
    
    # Set properties explicitly like MATLAB does (lines 40-46)
    P_out._emptySet_val = True        # P_out.emptySet.val = true;
    P_out._bounded_val = True         # P_out.bounded.val = true;
    P_out._fullDim_val = False        # P_out.fullDim.val = false;
    P_out._minHRep_val = True         # P_out.minHRep.val = true;
    P_out._minVRep_val = True         # P_out.minVRep.val = true;
    P_out.isVRep = True              # P_out.isVRep.val = true;
    P_out._V = np.zeros((n, 0))       # P_out.V_.val = zeros(n,0);
    # P_out.isHRep will be true because of the constructor Polytope(A,b)
    
    return P_out 