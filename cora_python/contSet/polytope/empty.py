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

    # 0*x <= -1 is an empty set in n dimensions
    A = np.zeros((1, n))
    b = np.array([[-1]])
    
    # Return a new Polytope defined by this infeasible constraint
    return Polytope(A, b) 