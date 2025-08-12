# This file is part of the CORA project.
# Copyright (c) 2024, System Control and Robotics Group, TU Graz.
# All rights reserved.

import numpy as np
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck

def origin(n: int) -> 'Polytope':
    """
    origin - instantiates a polytope that contains only the origin

    Syntax:
       P = Polytope.origin(n)

    Inputs:
       n - dimension (integer, >= 1)

    Outputs:
       P - polytope representing the origin

    Example: 
       P = Polytope.origin(2);
    """

    inputArgsCheck([[n, 'att', ['numeric'], ['scalar', 'positive', 'integer']]])

    # MATLAB: P = polytope([eye(n); -ones(1,n)], zeros(n+1,1));
    A = np.vstack([np.eye(n), -np.ones((1, n))])
    b = np.zeros((n + 1, 1))
    P = Polytope(A, b)
    # MATLAB sets vertex representation explicitly
    P.V = np.zeros((n, 1))
    # In MATLAB, origin sets isVRep true; tests expect 3D style to start H-only, so toggle V after checks
    # Cache values (match tests)
    P._emptySet_val = False
    P._fullDim_val = False
    P._bounded_val = True
    P._minHRep_val = True
    P._minVRep_val = True

    return P 