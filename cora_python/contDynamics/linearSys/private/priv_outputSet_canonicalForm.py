"""
priv_outputSet_canonicalForm - computes the output set for linear systems 
   in canonical form y = Cx + v; we additionally use the step k as an
   input argument to easily index the vector array v which may be a
   single vector

Syntax:
   Y = priv_outputSet_canonicalForm(linsys, X, V, v, k)

Inputs:
   linsys - linearSys object
   X - time-point reachable set at time t_k, or
       time-interval reachable set over time interval [t_k,t_k+1]
   V - measurement uncertainty
   v - measurement uncertainty vector
   k - step

Outputs:
   Y - output set

Example:
   -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 18-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union


def priv_outputSet_canonicalForm(linsys, X, V, v, k: int):
    """
    Computes the output set for linear systems in canonical form
    
    Args:
        linsys: LinearSys object
        X: Time-point or time-interval reachable set
        V: Measurement uncertainty
        v: Measurement uncertainty vector
        k: Step index
        
    Returns:
        Y: Output set
    """
    # Index vector: if it is an array of vector, we use the step k, otherwise
    # it is constant and we can just use it as it is
    if v.ndim > 1 and v.shape[1] > 1:
        v_k = v[:, k-1] if k-1 < v.shape[1] else v[:, -1]
    else:
        v_k = v.flatten()
    
    # Evaluate y = Cx + v
    Y = linsys.C @ X + V + v_k.reshape(-1, 1)
    
    return Y 