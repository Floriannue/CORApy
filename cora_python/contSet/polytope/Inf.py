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

    # Set properties like MATLAB does (lines 38-42)
    P_out._emptySet_val = False   # P_out.emptySet.val = false;
    P_out._bounded_val = False    # P_out.bounded.val = false;
    P_out._fullDim_val = True     # P_out.fullDim.val = true;
    P_out._minHRep_val = True     # P_out.minHRep.val = true;
    
    # Handle 0-dimensional case specially
    if n == 0:
        # 0-dimensional polytope represents a single point (the origin)
        # For 0D, we need to create a representation that indicates it's not empty
        # Since 0D has no coordinates, we'll create a special case
        P_out._V = np.array([]).reshape(0, 0)  # Empty array with shape (0, 0)
        P_out._minVRep_val = True
        P_out.isVRep = True
        # For 0D, we need to override the emptySet property since it represents a point
        P_out._emptySet_val = False
    # Only store vertices for low-dimensional polytopes (MATLAB: if n <= 8)
    elif n <= 8:
        # Compute all possible combinations of lower/upper bounds like MATLAB
        from itertools import product
        # MATLAB: fac = logical(combinator(2,n,'p','r')-1);
        # This creates all combinations of 0/1 for n dimensions
        fac = list(product([False, True], repeat=n))
        nrComb = len(fac)
        
        # Init all points with -Inf (MATLAB: V = -Inf(n,nrComb);)
        V = np.full((n, nrComb), -np.inf)
        
        # Loop over all factors (MATLAB: for i=1:nrComb, V(fac(i,:)',i) = Inf; end)
        for i, factor in enumerate(fac):
            for j, is_upper in enumerate(factor):
                if is_upper:
                    V[j, i] = np.inf
        
        P_out._minVRep_val = True     # P_out.minVRep.val = true;
        P_out._V = V                  # P_out.V_.val = V;
        P_out.isVRep = True           # P_out.isVRep.val = true;
    else:
        # For high dimensions, don't set V-representation (like MATLAB)
        # Don't set _V at all - let vertices_() handle it when called
        P_out.isVRep = False          # Not V-rep for high dimensions

    return P_out 