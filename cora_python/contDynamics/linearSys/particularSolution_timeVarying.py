"""
particularSolution_timeVarying - computes the particular solution
   after a given step size for time-varying uncertainties, see [1, (3.7)]

Syntax:
   Ptp = particularSolution_timeVarying(linsys,U,timeStep,truncationOrder)
   Ptp = particularSolution_timeVarying(linsys,U,timeStep,truncationOrder,blocks)

Inputs:
   linsys - linearSys object
   U - set of time-varying uncertainties
   timeStep - time step size Delta t
   truncationOrder - truncation order for power series
   blocks - bx2 array with b blocks for decomposition algorithm

Outputs:
   Ptp - time-varying particular solution at t = Delta t

Example:
   linsys = linearSys([-1 -4; 4 -1]);
   U = zonotope(zeros(2,1),0.05*eye(2));
   timeStep = 0.05;
   truncationOrder = 6;
   Ptp = particularSolution_timeVarying(linsys,U,timeStep,truncationOrder);

References:
   [1] M. Althoff. "Reachability Analysis and its Application to the
       Safety Assessment of Autonomous Cars", PhD Dissertation, 2010.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: particularSolution_constant

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       03-April-2024 (MATLAB)
Last update:   16-October-2024 (MW, integrate block decomposition) (MATLAB)
Last revision: ---
Python translation: 2025
"""

import numpy as np
from typing import Optional, Union
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.contSet import decompose
from cora_python.g.functions.helper.sets.contSet.contSet import block_mtimes, block_operation


def particularSolution_timeVarying(linsys, U, timeStep: float, truncationOrder: int, 
                                  blocks: Optional[np.ndarray] = None):
    """
    Computes the particular solution after a given step size for time-varying uncertainties
    
    Args:
        linsys: LinearSys object
        U: Set of time-varying uncertainties
        timeStep: Time step size Delta t
        truncationOrder: Truncation order for power series
        blocks: Optional bx2 array with b blocks for decomposition algorithm
        
    Returns:
        Ptp: Time-varying particular solution at t = Delta t
    """
    # Set default blocks if not provided
    if blocks is None:
        blocks = np.array([[1, linsys.nr_of_dims]])
    
    # Quick exit if U is all-zero vector or set containing only the origin
    if U.representsa_('origin', np.finfo(float).eps):
        return _block_zeros(blocks)
    
    # Since this function is public, we cannot assume that taylorLinSys has
    # already been instantiated
    if not hasattr(linsys, 'taylor') or not hasattr(linsys.taylor, 'getTaylor'):
        from cora_python.g.classes.taylorLinSys import TaylorLinSys
        linsys.taylor = TaylorLinSys(linsys.A)
    
    # Set a maximum order in case truncation order is given as Inf (adaptive)
    truncationOrderInf = np.isinf(truncationOrder)
    if truncationOrderInf:
        truncationOrder = 75
    
    # Compute by sum until floating-point precision (if truncationOrder = Inf)
    # formula: \bigosum_{j=0}^\infty \frac{A^{j}}{j+1!} timeStep^{j+1} U
    
    # First term (eta = 0: A^0*dt^1/1 * U = dt*U)
    Ptp = timeStep * U
    Ptp = decompose(Ptp, blocks)
    
    # Decompose input set for iterative operations below
    U_decomp = decompose(U, blocks)
    
    # Loop until Asum no longer changes (additional values too small) or
    # truncation order is reached
    for eta in range(1, truncationOrder + 1):
        Apower_mm = linsys.taylor._computeApower(eta)
        dtoverfac = timeStep**(eta + 1) / np.math.factorial(eta + 1)
        
        # Additional term (only matrix)
        addTerm = Apower_mm * dtoverfac
        
        # Adaptive truncation order
        if truncationOrderInf:
            if np.any(np.isinf(addTerm)) or eta == truncationOrder:
                # Safety check (if time step size too large, then the sum
                # converges too late so we already have Inf values)
                raise RuntimeError('Time Step Size too big for computation.')
            elif np.all(np.abs(addTerm) <= np.finfo(float).eps):
                # If new term does not change stored values in Asum, i.e., all
                # entries are below floating-point accuracy -> stop loop
                break
        
        # Add term (including U!)
        Ptp_eta = block_mtimes(addTerm, U_decomp)
        Ptp = block_operation(lambda a, b: a + b, Ptp, Ptp_eta)
    
    return Ptp


def _block_zeros(blocks):
    """Create block of zero sets"""
    from cora_python.contSet.zonotope import Zonotope
    
    if blocks.shape[0] == 1:
        # Single block
        dim = blocks[0, 1] - blocks[0, 0] + 1
        return Zonotope.origin(dim)
    else:
        # Multiple blocks
        result = []
        for i in range(blocks.shape[0]):
            dim = blocks[i, 1] - blocks[i, 0] + 1
            result.append(Zonotope.origin(dim))
        return result 