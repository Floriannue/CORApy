"""
canonicalForm - rewrite inhomogeneity to canonical forms:
   Ax + Bu + c + w  ->  Ax + u', where u' ∈ U_ + u_
   Cx + Du + k + v  ->  Cx + v', where v' ∈ V_ + v_

Syntax:
    [linsys_, U_, u_, V_, v_] = canonicalForm(linsys, U, uVec, W, V, vVec)

Inputs:
    linsys - linearSys object
    U - input set (time-varying)
    uVec - input vector (piecewise constant)
    W - disturbance set (time-varying)
    V - noise set (time-varying)
    vVec - noise vector (piecewise constant)

Outputs:
    sys_ - linearSys object in canonical form
    U_ - input set, interpreted as time-varying over t
    u_ - piecewise constant input trajectory
    V_ - output uncertainty
    v_ - piecewise constant offset on output

Example:
    linsys = LinearSys([-1, -4; 4, -1], [1; -1])
    U = Zonotope(1, 0.02)
    uVec = [2, 4, -1, 0, 3]
    W = Zonotope(0.5, 0.05)
    V = Zonotope(-0.2, 0.1)
    vVec = [0, -1, 0, 1, 2, 3]

    [linsys_, U_, u_, V_, v_] = canonicalForm(linsys, U, uVec, W, V, vVec)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: linearSys/linearSys

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 05-April-2024 (MATLAB)
Last update: 17-October-2024 (MW, integrate disturbance/noise matrix) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Tuple, Union
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval


def canonicalForm(linsys, U, uVec, W, V, vVec) -> Tuple:
    """
    Rewrite inhomogeneity to canonical forms
    
    Args:
        linsys: LinearSys object
        U: Input set (time-varying)
        uVec: Input vector (piecewise constant)
        W: Disturbance set (time-varying)
        V: Noise set (time-varying)
        vVec: Noise vector (piecewise constant)
        
    Returns:
        Tuple of (linsys_, U_, u_, V_, v_) in canonical form
    """
    # Given sets:
    # - U: set, not necessarily centered at zero
    # - uVec: vector or list of vectors
    # - W: set, not necessarily centered at zero
    # - V: set, not necessarily centered at zero
    # - vVec: vector or list of vectors
    
    # Read out all centers
    centerU = _center(U)
    centerW = _center(W)
    centerV = _center(V)
    
    # Shift all sets so that they are centered at the origin
    U = U + (-centerU)
    W = W + (-centerW)
    V = V + (-centerV)
    
    # Offset vector the output: if it is not constant, we require an additional
    # column in U because there are steps+1 time points (on which we evaluate
    # the output set) but only steps time intervals (for which we need uVec)
    if uVec.shape[1] == 1:
        # Handle F matrix multiplication: if F is a column vector, use element-wise multiplication
        if linsys.F.shape[1] == 1 and (centerV + vVec).shape == linsys.F.shape:
            F_term = linsys.F * (centerV + vVec)
        else:
            F_term = linsys.F @ (centerV + vVec)
        v_ = linsys.D @ uVec + linsys.k + F_term
    elif not np.any(linsys.D) and not np.any(linsys.k) and not np.any(linsys.F):
        v_ = np.zeros((linsys.nr_of_outputs, 1))
    else:
        # Only compute if a non-zero result is to be expected
        uVec_extended = np.hstack([uVec, np.zeros((linsys.nr_of_inputs, 1))])
        # Handle F matrix multiplication: if F is a column vector, use element-wise multiplication
        if linsys.F.shape[1] == 1 and (centerV + vVec).shape == linsys.F.shape:
            F_term = linsys.F * (centerV + vVec)
        else:
            F_term = linsys.F @ (centerV + vVec)
        v_ = linsys.D @ uVec_extended + linsys.k + F_term
    
    # Simplify representation if result is all-zero
    if not np.any(v_):
        v_ = np.zeros((linsys.nr_of_outputs, 1))
    
    # Time-varying uncertainty on the output
    # Handle F matrix multiplication: if F is a column vector, use element-wise multiplication
    if linsys.F.shape[1] == 1 and V.dim() == linsys.F.shape[0]:
        # F is a column vector and V has compatible dimension
        # In MATLAB, this would be element-wise multiplication
        # For zonotopes, we need to create a diagonal matrix from F
        F_diag = np.diag(linsys.F.flatten())
        V_ = linsys.D @ (U + centerU) + F_diag @ V
    else:
        V_ = linsys.D @ (U + centerU) + linsys.F @ V
    
    # Offset vector for state
    u_ = linsys.B @ uVec + linsys.B @ centerU + linsys.c + linsys.E @ centerW
    
    # Time-varying uncertainty for state
    U_ = linsys.B @ U + linsys.E @ W
    
    # Update system dynamics
    n = linsys.nr_of_dims
    r = linsys.nr_of_outputs
    
    # Create new system in canonical form: x' = Ax + u, y = Cx + v
    from .linearSys import LinearSys
    linsys_ = LinearSys(
        A=linsys.A,
        B=np.eye(n),
        c=np.zeros((n, 1)),
        C=linsys.C,
        D=np.zeros((r, n)),
        k=np.zeros((r, 1)),
        E=np.zeros((n, n)),
        F=np.eye(r)
    )
    
    # Copy helper property
    if hasattr(linsys, 'taylor'):
        linsys_.taylor = linsys.taylor
    
    return linsys_, U_, u_, V_, v_


def _center(set_obj):
    """Get center of a set object"""
    if hasattr(set_obj, 'center'):
        center = set_obj.center()
    elif isinstance(set_obj, np.ndarray):
        center = set_obj
    elif hasattr(set_obj, 'c'):
        center = set_obj.c
    else:
        # Assume it's a numeric value
        center = np.array(set_obj)
    
    # Ensure center is a column vector
    if isinstance(center, np.ndarray):
        if center.ndim == 1:
            center = center.reshape(-1, 1)
        elif center.ndim == 2 and center.shape[1] != 1:
            # If it's a row vector, transpose it
            if center.shape[0] == 1:
                center = center.T
    
    return center 