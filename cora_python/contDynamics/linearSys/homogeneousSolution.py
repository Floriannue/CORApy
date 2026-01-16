"""
homogeneousSolution - computes the homogeneous solution after a given
   elapsed time Delta t and over a time interval [0, Delta t]. Since the
   time-point solution is required for computing the time-interval
   solution, we compute both here, but without overhead if only the
   time-point solution is requested; the truncation order is determined
   automatically if it is given as Inf

Syntax:
   Htp = homogeneousSolution(linsys,X,timeStep,truncationOrder)
   [Htp,Hti,C_state] = homogeneousSolution(linsys,X,timeStep,truncationOrder)
   [Htp,Hti,C_state] = homogeneousSolution(linsys,X,timeStep,truncationOrder,blocks)

Inputs:
   linsys - linearSys object
   X - start set at t = 0
   timeStep - time step size Delta t
   truncationOrder - truncation order for power series
   blocks - bx2 array with b blocks for decomposition algorithm

Outputs:
   Htp - homgeneous solution at t = Delta t
   Hti - homgeneous solution over [0, Delta t]
   C_state - curvature error over [0, Delta t]

Example:
   linsys = linearSys([-1 -4; 4 -1]);
   X = zonotope(ones(2,1),diag([0.2;0.5]));
   timeStep = 0.05;
   truncationOrder = 6;
   
   [Htp,Hti] = homogeneousSolution(linsys,X,timeStep,truncationOrder);
  
   figure; hold on; box on;
   plot(X,[1,2],'k');
   plot(Htp,[1,2],'b');
   plot(Hti,[1,2],'b');

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: linearSys/affineSolution

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       03-April-2024 (MATLAB)
Last update:   16-October-2024 (MW, integrate block decomposition) (MATLAB)
Last revision: ---
Python translation: 2025
"""

import numpy as np
from typing import Tuple, Optional, Union
from cora_python.g.functions.helper.sets.contSet.contSet import block_mtimes
from cora_python.g.functions.helper.sets.contSet.contSet import block_operation

from cora_python.g.classes.taylorLinSys import TaylorLinSys
from cora_python.contSet.zonotope import Zonotope


def homogeneousSolution(linsys, X, timeStep: float, truncationOrder: int, 
                       blocks: Optional[np.ndarray] = None) -> Tuple:
    """
    Computes the homogeneous solution after a given elapsed time Delta t 
    and over a time interval [0, Delta t]
    
    Args:
        linsys: LinearSys object
        X: Start set at t = 0
        timeStep: Time step size Delta t
        truncationOrder: Truncation order for power series
        blocks: Optional bx2 array with b blocks for decomposition algorithm
        
    Returns:
        Tuple of (Htp, Hti, C_state) where:
        - Htp: Homogeneous solution at t = Delta t
        - Hti: Homogeneous solution over [0, Delta t]
        - C_state: Curvature error over [0, Delta t]
    """
    # Set default blocks if not provided
    if blocks is None:
        blocks = np.array([[1, linsys.nr_of_dims]])
    
    # Since this function is public, we cannot assume that taylorLinSys has
    # already been instantiated
    if not hasattr(linsys, 'taylor') or not hasattr(linsys.taylor, 'getTaylor'):
        linsys.taylor = TaylorLinSys(linsys.A)
    
    # Propagation matrix
    eAdt = linsys.taylor.getTaylor('eAdt', timeStep=timeStep)
    
    # Decompose start set (remains the same if no blocks given)
    X_decomposed = X.decompose(blocks)
    
    # Homogeneous time-point solution
    Htp = block_mtimes(eAdt, X_decomposed)
    
    # Check if time-interval solution should also be computed
    # (based on number of expected outputs)
    import inspect
    frame = inspect.currentframe()
    try:
        # Get the calling frame to check how many outputs are expected
        caller_frame = frame.f_back
        # This is a heuristic - in practice, we'll compute all outputs
        compute_time_interval = True
    finally:
        del frame
    
    if compute_time_interval:
        # Curvature error
        C_state = priv_curvatureState(linsys, X_decomposed, timeStep, truncationOrder)
        
        # Homogeneous time-interval solution
        Hti_approx = block_operation(lambda a, b: a.enclose(b), X_decomposed, Htp)
        Hti = block_operation(lambda a, b: a + b, Hti_approx, C_state)
        
        return Htp, Hti, C_state
    else:
        return Htp


def priv_curvatureState(linsys, X, timeStep: float, truncationOrder: int):
    """
    Compute curvature error for the state
    
    MATLAB: C_state = priv_curvatureState(linsys,X,timeStep,truncationOrder);
    
    Args:
        linsys: LinearSys object
        X: Start set (possibly decomposed)
        timeStep: Time step size
        truncationOrder: Truncation order
        
    Returns:
        Curvature error set
    """
    # MATLAB: F = priv_correctionMatrixState(linsys,timeStep,truncationOrder);
    from cora_python.contDynamics.linearSys.private.priv_correctionMatrixState import priv_correctionMatrixState
    F = priv_correctionMatrixState(linsys, timeStep, truncationOrder)
    
    # MATLAB: try
    #     C_state = block_mtimes(F,X);
    # catch
    #     % convert set to interval if interval matrix * set not supported
    #     C_state = block_mtimes(F,block_operation(@interval,X));
    # end
    try:
        from cora_python.g.functions.helper.sets.contSet.contSet import block_mtimes
        C_state = block_mtimes(F, X)
    except (TypeError, ValueError, AttributeError):
        # convert set to interval if interval matrix * set not supported
        from cora_python.g.functions.helper.sets.contSet.contSet import block_operation
        from cora_python.g.functions.helper.sets.contSet.contSet import block_mtimes
        X_interval = block_operation(lambda x: x.interval(), X)
        C_state = block_mtimes(F, X_interval)
    
    return C_state 