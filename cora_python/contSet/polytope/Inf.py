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
    P_out = Polytope(np.zeros((0, n)), np.ones((0, 1)))

    # Note: In Python, we don't set properties directly. Instead, we rely on
    # functions like isBounded(), representsa_('emptySet'), etc. to compute
    # these properties when needed. This avoids the complexity of maintaining
    # cached state and follows better Python design principles.

    return P_out 