# This file is part of the CORA project.
# Copyright (c) 2024, System Control and Robotics Group, TU Graz.
# All rights reserved.

import numpy as np
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.g.functions.matlab.validate.check.input_args_check import input_args_check

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

    input_args_check([[n, ['att', 'numeric'], ['scalar', 'positive', 'integer']]])

    # init halfspace representation (simplex with zero offset)
    A = np.vstack((np.eye(n), -np.ones((1, n))))
    b = np.zeros((n + 1, 1))
    
    # Create polytope from H-representation
    P = Polytope(A, b)

    # Explicitly set properties for the origin as they are known at construction
    P._emptySet_val = False
    P._emptySet_is_computed = True
    P._bounded_val = True
    P._bounded_is_computed = True
    P._fullDim_val = (n > 0) # Origin is full-dimensional if n > 0
    P._fullDim_is_computed = True
    P._minHRep_val = True
    P._minHRep_is_computed = True
    P._minVRep_val = True
    P._minVRep_is_computed = True
    P._V = np.zeros((n, 1)) # The origin is a single vertex at [0,...,0].
    P.isVRep = True # It also has a V-representation
    # P.isHRep is already True from the constructor

    return P 