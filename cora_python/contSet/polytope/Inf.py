# This file is part of the CORA project.
# Copyright (c) 2024, System Control and Robotics Group, TU Graz.
# All rights reserved.

import numpy as np
from itertools import product
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.g.functions.matlab.validate.check.input_args_check import input_args_check

def Inf(n: int = 0) -> Polytope:
    """
    Inf - instantiates an n-dimensional polytope that is equivalent to R^n

    Syntax:
       P_out = Polytope.Inf(n)

    Inputs:
       n - dimension

    Outputs:
       P_out - polytope representing R^n

    Examples:
       P = Polytope.Inf(2);
    """

    # parse input
    input_args_check([[n, 'att', 'numeric', {'scalar', 'nonnegative'}]])

    # the polytope 0*x <= 1 is R^n
    # For R^n, A and b should be empty, representing no constraints.
    # The MATLAB code creates zeros(0,n) and ones(0,1), which effectively are empty constraints.
    P_out = Polytope(np.zeros((0, n)), np.ones((0, 1)), dim=n)

    # Explicitly set properties for R^n as they are known at construction
    P_out._emptySet_val = False
    P_out._emptySet_is_computed = True
    P_out._bounded_val = False
    P_out._bounded_is_computed = True
    P_out._fullDim_val = True
    P_out._fullDim_is_computed = True
    P_out._minHRep_val = True
    P_out._minHRep_is_computed = True
    P_out._minVRep_val = False # R^n has no minimal V-representation
    P_out._minVRep_is_computed = True
    P_out._V = np.zeros((n, 0)) # Explicitly set V to empty for R^n
    P_out.isVRep = False # Explicitly set VRep flag

    return P_out 