# This file is part of the CORA project.
# Copyright (c) 2024, System Control and Robotics Group, TU Graz.
# All rights reserved.

import numpy as np
from typing import Tuple, Optional

def priv_plus_minus_vector(A: np.ndarray, b: np.ndarray, 
                          Ae: Optional[np.ndarray], be: Optional[np.ndarray], 
                          v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    priv_plus_minus_vector - computes the translation of a polytope by a
    vector, assumed 'P+v' (for 'P-v', simply hand over -v)

    Syntax:
       A, b, Ae, be = priv_plus_minus_vector(A, b, Ae, be, v)

    Inputs:
       A - inequality constraint matrix
       b - inequality constraint offset
       Ae - equality constraint matrix (can be None)
       be - equality constraint offset (can be None)
       v - numeric vector

    Outputs:
       A - inequality constraint matrix (unchanged)
       b - inequality constraint offset (updated)
       Ae - equality constraint matrix (unchanged)
       be - equality constraint offset (updated)

    Authors: Mark Wetzlinger (MATLAB)
             Python translation by AI Assistant
    Written: 03-October-2024 (MATLAB)
    Python translation: 2025
    """

    # compute shift for inequality constraints
    b_new = b + A @ v
    
    # compute shift for equality constraints (if they exist)
    if Ae is not None and be is not None:
        be_new = be + Ae @ v
    else:
        be_new = be
    
    return A, b_new, Ae, be_new 