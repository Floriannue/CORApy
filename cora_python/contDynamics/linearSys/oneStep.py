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
from ...contSet.zonotope import Zonotope
from ...contSet.interval import Interval


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
    from .homogeneousSolution import homogeneousSolution
    Htp, Hti, C_state = homogeneousSolution(linsys, X, timeStep, truncationOrder, blocks)
    
    # Compute particular solution due to constant input
    from .particularSolution_constant import particularSolution_constant
    Pu, C_input_const = particularSolution_constant(linsys, u, timeStep, truncationOrder, blocks)
    
    # Compute particular solution due to time-varying input
    from .particularSolution_timeVarying import particularSolution_timeVarying
    PU, C_input_tv = particularSolution_timeVarying(linsys, U, timeStep, truncationOrder, blocks)
    
    # Combine input corrections
    C_input = C_input_const + C_input_tv
    
    # Compute reachable sets
    Rtp = Htp + PU + Pu
    Rti = Hti + PU + Pu + C_input
    
    return Rtp, Rti, Htp, Hti, PU, Pu, C_state, C_input 