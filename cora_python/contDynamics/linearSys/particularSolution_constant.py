"""
particularSolution_constant - computes the particular solution
   after a given step size for a constant vector or set, see [1, (3.6)];
   the truncation order is determined automatically if it is given as Inf

Syntax:
   Ptp = particularSolution_constant(linsys,U,timeStep,truncationOrder)
   [Ptp,C_input,Pti] = particularSolution_constant(linsys,U,timeStep,truncationOrder)
   [Ptp,C_input,Pti] = particularSolution_constant(linsys,U,timeStep,truncationOrder,blocks)

Inputs:
   linsys - linearSys object
   U - vector of set of constant inputs
   timeStep - time step size Delta t
   truncationOrder - truncation order for power series
   blocks - bx2 array with b blocks for decomposition algorithm

Outputs:
   Ptp - particular solution at t = Delta t
   C_input - curvature error for the input over [0, Delta t]
   Pti - particular solution over [0, Delta t]

Example:
   linsys = linearSys([-1 -4; 4 -1]);
   U = [1;0];
   timeStep = 0.05;
   truncationOrder = 6;
   Ptp = particularSolution_constant(linsys,U,timeStep,truncationOrder);

References:
   [1] M. Althoff. "Reachability Analysis and its Application to the
       Safety Assessment of Autonomous Cars", PhD Dissertation, 2010.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       03-April-2024 (MATLAB)
Last update:   16-October-2024 (MW, integrate block decomposition) (MATLAB)
Last revision: ---
Python translation: 2025
"""

import math
import numpy as np
from typing import Tuple, Optional, Union
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.g.functions.helper.sets.contSet.contSet import block_mtimes
from cora_python.g.functions.helper.sets.contSet.contSet import block_operation
from .private.priv_correctionMatrixInput import priv_correctionMatrixInput
from cora_python.g.classes.taylorLinSys import TaylorLinSys


def particularSolution_constant(linsys, U, timeStep: float, truncationOrder: int, 
                               blocks: Optional[np.ndarray] = None) -> Tuple:
    """
    Computes the particular solution after a given step size for a constant vector or set
    
    Args:
        linsys: LinearSys object
        U: Vector or set of constant inputs
        timeStep: Time step size Delta t
        truncationOrder: Truncation order for power series
        blocks: Optional bx2 array with b blocks for decomposition algorithm
        
    Returns:
        Tuple of (Ptp, C_input, Pti) where:
        - Ptp: Particular solution at t = Delta t
        - C_input: Curvature error for the input over [0, Delta t]
        - Pti: Particular solution over [0, Delta t]
    """
    # Set default blocks if not provided
    if blocks is None:
        blocks = np.array([[1, linsys.nr_of_dims]])
    
    # For ease of computation, convert a vector to a zonotope
    numericU = isinstance(U, np.ndarray) or np.isscalar(U)
    if numericU:
        U = Zonotope(U)
    
    # Quick exit if U is all-zero vector or set containing only the origin
    if U.representsa_('origin', np.finfo(float).eps):
        Ptp = _block_zeros(blocks)
        C_input = Ptp
        Pti = Ptp
        return Ptp, C_input, Pti
    
    # Since this function is public, we cannot assume that taylorLinSys has
    # already been instantiated
    if not hasattr(linsys, 'taylor') or not hasattr(linsys.taylor, 'getTaylor'):
        linsys.taylor = TaylorLinSys(linsys.A)
    
    # Set a maximum order in case truncation order is given as Inf (adaptive)
    truncationOrderInf = np.isinf(truncationOrder)
    if truncationOrderInf:
        truncationOrder = 75
    
    # Decompose input set (remains the same unless more than one block)
    U_decomp = U.decompose(blocks)
    
    # Check if inverse can be used
    Ainv = linsys.taylor.getTaylor('Ainv')
    if Ainv is not None:
        # Ainv would be None if there was no inverse
        eAdt = linsys.taylor.getTaylor('eAdt', timeStep=timeStep)
        Ptp = block_mtimes(Ainv @ (eAdt - np.eye(linsys.nr_of_dims)), U_decomp)
        
        # Compute time-interval solution if desired
        C_input = _priv_curvatureInput(linsys, U_decomp, timeStep, truncationOrder)
        Pti_approx = block_operation(_convHull, _block_zeros(blocks), Ptp)
        Pti = block_operation(lambda a, b: a + b, Pti_approx, C_input)
        
        # Re-convert time-point particular solution if U was numeric since
        # analytical solution returns a vector
        if numericU:
            Ptp = block_operation(_center, Ptp)
        
        return Ptp, C_input, Pti
    
    # Compute by sum until floating-point precision (if truncationOrder = Inf)
    # formula: \sum_{j=0}^\infty \frac{A^{j}}{j+1!} timeStep^{j+1}
    
    # First term (eta = 0)
    Asum = timeStep * np.eye(linsys.nr_of_dims)
    
    # Loop until Asum no longer changes (additional values too small) or
    # truncation order is reached
    for eta in range(1, truncationOrder + 1):
        Apower_mm = linsys.taylor._computeApower(eta)
        dtoverfac = timeStep**(eta + 1) / math.factorial(eta + 1)
        
        # Additional term
        addTerm = Apower_mm * dtoverfac
        
        # Adaptive truncation order
        if truncationOrderInf:
            if np.any(np.isinf(addTerm)) or eta == truncationOrder:
                # Safety check (if time step size too large, then the sum
                # converges too late so we already have Inf values)
                raise RuntimeError('Time Step Size too big for computation of Pu.')
            elif np.all(np.abs(addTerm) <= np.finfo(float).eps * np.abs(Asum)):
                # If new term does not change stored values in Asum, i.e., all
                # entries are below floating-point accuracy -> stop loop
                break
        
        # Add term to current Asum
        Asum = Asum + addTerm
    
    # If floating-point precision has not been reached, we require the remainder term
    if truncationOrderInf:
        Ptp = block_mtimes(Asum, U_decomp)
        if numericU:
            Ptp = block_operation(_center, Ptp)
    else:
        from .private.priv_expmRemainder import priv_expmRemainder
        E = priv_expmRemainder(linsys, timeStep, truncationOrder)
        # MATLAB: Ptp = block_operation(@plus,block_mtimes(Asum,U_decomp),block_mtimes(E*timeStep,U_decomp));
        Ptp = block_operation(lambda a, b: a + b, 
                              block_mtimes(Asum, U_decomp), 
                              block_mtimes(E * timeStep, U_decomp))
        if numericU:
            Ptp = block_operation(_center, Ptp)
    
    # Compute time-interval solution if desired
    C_input = _priv_curvatureInput(linsys, U_decomp, timeStep, truncationOrder)
    Pti_approx = block_operation(_convHull, _block_zeros(blocks), Ptp)
    Pti = block_operation(lambda a, b: a + b, Pti_approx, C_input)
    
    return Ptp, C_input, Pti


def _block_zeros(blocks):
    """Create block of zero sets"""
    
    if blocks.shape[0] == 1:
        # Single block - note: blocks are 1-indexed from MATLAB but 0-indexed ranges in Python
        start_idx = blocks[0, 0] - 1  # Convert from 1-indexed to 0-indexed
        end_idx = blocks[0, 1] - 1    # Convert from 1-indexed to 0-indexed
        dim = end_idx - start_idx + 1
        return Zonotope.origin(dim)
    else:
        # Multiple blocks
        result = []
        for i in range(blocks.shape[0]):
            start_idx = blocks[i, 0] - 1  # Convert from 1-indexed to 0-indexed
            end_idx = blocks[i, 1] - 1    # Convert from 1-indexed to 0-indexed
            dim = end_idx - start_idx + 1
            result.append(Zonotope.origin(dim))
        return result


def _priv_curvatureInput(linsys, U_decomp, timeStep, truncationOrder):
    """
    Compute curvature error for the input
    
    MATLAB: 
        G = priv_correctionMatrixInput(linsys,timeStep,truncationOrder);
        try
            C_input = block_mtimes(G,U);
        catch
            % convert set to interval if interval matrix * set not supported
            C_input = block_mtimes(G,block_operation(@interval,U));
        end
    """
    # Ensure taylorLinSys is initialized (required by priv_correctionMatrixInput)
    if not hasattr(linsys, 'taylor') or linsys.taylor is None:
        linsys.taylor = TaylorLinSys(linsys.A)
    
    # Compute correction matrix G (interval matrix, size n x n)
    # Note: After canonicalForm, U is already in state space (dimension n),
    # so G (n x n) can be multiplied directly with U (n-dimensional)
    G = priv_correctionMatrixInput(linsys, timeStep, truncationOrder)
    
    # Try block_mtimes(G, U_decomp)
    try:
        C_input = block_mtimes(G, U_decomp)
    except:
        # Convert set to interval if interval matrix * set not supported
        from cora_python.contSet.interval import Interval
        U_decomp_interval = block_operation(lambda x: Interval(x), U_decomp)
        C_input = block_mtimes(G, U_decomp_interval)
    
    return C_input


def _convHull(set1, set2):
    """Compute convex hull of two sets (MATLAB: convHull)."""
    from cora_python.contSet.contSet.convHull import convHull
    return convHull(set1, set2)


def _center(set_obj):
    """Get center of a set object"""
    if hasattr(set_obj, 'center'):
        return set_obj.center()
    elif hasattr(set_obj, 'c'):
        return set_obj.c
    else:
        return set_obj 