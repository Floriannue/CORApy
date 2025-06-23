"""
oneStep - computes the reachable continuous set for one time step 

Syntax:
    [Rtp, Rti] = oneStep(linsys, X, U, u, timeStep, truncationOrder)
    [Rtp, Rti, Htp, Hti, PU, Pu] = oneStep(linsys, X, U, u, timeStep, truncationOrder)
    [Rtp, Rti, Htp, Hti, PU, Pu, C_state, C_input] = oneStep(linsys, X, U, u, timeStep, truncationOrder)
    [Rtp, Rti, Htp, Hti, PU, Pu, C_state, C_input] = oneStep(linsys, X, U, u, timeStep, truncationOrder, blocks)

Inputs:
    linsys - linearSys object
    X - start set at t = 0
    U - input set centered at the origin, time varying over [0, Delta t]
    u - constant input over [0, Delta t]
    timeStep - time step size Delta t
    truncationOrder - truncation order for power series
    blocks - bx2 array with b blocks for decomposition algorithm

Outputs:
    Rtp - reachable set at t = Delta t
    Rti - reachable set over [0, Delta t]
    Htp - affine solution at t = Delta t
    Hti - affine solution over [0, Delta t] without C_input (necessary so
          that we can propagate Hti)
    PU - particular solution due to time-varying inputs at t = Delta t
    Pu - constant input solution at t = Delta t
    C_state - curvature error for the state
    C_input - curvature error for the input

Example:
    linsys = LinearSys([-1, -4; 4, -1])
    X = Zonotope(ones(2,1), diag([0.2; 0.5]))
    U = Zonotope(zeros(2,1), 0.01*eye(2))
    u = [1; 0]
    timeStep = 0.05
    truncationOrder = 6
    
    [Rtp, Rti] = oneStep(linsys, X, U, u, timeStep, truncationOrder)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 07-May-2007 (MATLAB)
Last update: 03-January-2008 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Tuple, Optional, Union
from cora_python.contSet.zonotope import Zonotope


def oneStep(linsys, X, U, u, timeStep: float, truncationOrder: int, 
           blocks: Optional[np.ndarray] = None) -> Tuple:
    """
    Computes the reachable continuous set for one time step
    
    Args:
        linsys: LinearSys object
        X: Start set at t = 0
        U: Input set centered at the origin, time varying over [0, Delta t]
        u: Constant input over [0, Delta t]
        timeStep: Time step size Delta t
        truncationOrder: Truncation order for power series
        blocks: Optional blocks for decomposition algorithm
        
    Returns:
        Tuple of (Rtp, Rti, Htp, Hti, PU, Pu, C_state, C_input)
    """
    # Compute homogeneous solution
    Htp, Hti, C_state = linsys.homogeneousSolution(X, timeStep, truncationOrder, blocks)
    
    # Compute particular solution due to constant input
    Pu, C_input_const, _ = linsys.particularSolution_constant(u, timeStep, truncationOrder, blocks)
    
    # Compute particular solution due to time-varying input
    PU = linsys.particularSolution_timeVarying(U, timeStep, truncationOrder, blocks)
    
    # For now, assume no curvature error from time-varying input (simplified)
    if blocks is None:
        C_input_tv = Zonotope.origin(linsys.nr_of_dims)
    else:
        if blocks.shape[0] == 1:
            dim = blocks[0, 1] - blocks[0, 0] + 1
            C_input_tv = Zonotope.origin(dim)
        else:
            C_input_tv = []
            for i in range(blocks.shape[0]):
                dim = blocks[i, 1] - blocks[i, 0] + 1
                C_input_tv.append(Zonotope.origin(dim))
    
    # Combine input corrections
    C_input = C_input_const + C_input_tv
    
    # Compute reachable sets
    Rtp = Htp + PU + Pu
    Rti = Hti + PU + Pu + C_input
    
    return Rtp, Rti, Htp, Hti, PU, Pu, C_state, C_input 