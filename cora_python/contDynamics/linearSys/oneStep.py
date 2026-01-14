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
from typing import Tuple, Optional
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
    # MATLAB: narginchk(6,7);
    # MATLAB: blocks = setDefaultValues({[1,linsys.nrOfDims]},varargin);
    # by default no block decomposition, i.e., a single block
    if blocks is None:
        blocks = np.array([[1, linsys.nr_of_dims]])
    
    # MATLAB: PU = particularSolution_timeVarying(linsys,U,timeStep,truncationOrder,blocks);
    # compute time-varying input solution
    PU = linsys.particularSolution_timeVarying(U, timeStep, truncationOrder, blocks)
    
    # MATLAB: [Pu,C_input] = particularSolution_constant(linsys,u,timeStep,truncationOrder,blocks);
    # compute constant input solution
    Pu, C_input_const, _ = linsys.particularSolution_constant(u, timeStep, truncationOrder, blocks)
    
    # For time-varying input curvature error (simplified - assume zero for now)
    # This is handled in particularSolution_timeVarying, but we need to combine with C_input_const
    # Since C_input_tv is always zero, we can just use C_input_const directly
    # (C_input_const can be a single Zonotope or a list of Zonotopes depending on blocks)
    C_input = C_input_const
    
    # MATLAB: Htp = homogeneousSolution(linsys,X,timeStep,truncationOrder,blocks);
    # compute homogeneous time-point solution (only Htp, not Hti or C_state)
    # Note: homogeneousSolution decomposes X internally, but we'll decompose it again below
    Htp_result = linsys.homogeneousSolution(X, timeStep, truncationOrder, blocks)
    if isinstance(Htp_result, tuple):
        Htp = Htp_result[0]  # Only take Htp, ignore Hti and C_state
    else:
        Htp = Htp_result
    
    # MATLAB: Htp = block_operation(@plus,Htp,Pu);
    # extend to affine time-point solution
    from cora_python.g.functions.helper.sets.contSet.contSet import block_operation
    Htp = block_operation(lambda a, b: a + b, Htp, Pu)
    
    # MATLAB: X = decompose(X,blocks);
    # decompose start set (remains the same if no blocks given)
    X = X.decompose(blocks)
    
    # MATLAB: C_state = priv_curvatureState(linsys,X,timeStep,truncationOrder);
    # compute curvature error for the state
    from cora_python.contDynamics.linearSys.homogeneousSolution import priv_curvatureState
    C_state = priv_curvatureState(linsys, X, timeStep, truncationOrder)
    
    # MATLAB: Hti = block_operation(@plus,block_operation(@enclose,X,Htp),C_state);
    # compute affine time-interval solution
    from cora_python.g.functions.helper.sets.contSet.contSet import enclose
    Hti_approx = block_operation(enclose, X, Htp)
    Hti = block_operation(lambda a, b: a + b, Hti_approx, C_state)
    
    # MATLAB: Rtp = block_operation(@plus,Htp,PU);
    # reachable set as addition of affine and particular solution
    Rtp = block_operation(lambda a, b: a + b, Htp, PU)
    
    # MATLAB: Rti = block_operation(@plus,Hti,block_operation(@plus,PU,C_input));
    PU_C_input = block_operation(lambda a, b: a + b, PU, C_input)
    Rti = block_operation(lambda a, b: a + b, Hti, PU_C_input)
    
    return Rtp, Rti, Htp, Hti, PU, Pu, C_state, C_input
