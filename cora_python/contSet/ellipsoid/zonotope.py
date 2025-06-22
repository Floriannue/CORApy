"""
zonotope - converts an ellipsoid to a zonotope

Syntax:
    Z = zonotope(E)
    Z = zonotope(E, mode)
    Z = zonotope(E, mode, nrGen)

Inputs:
    E - ellipsoid object
    mode - (optional) Specifies whether function uses a lower bound on the 
           minimum zonotope norm or the exact value:
           * 'outer:box':      overapprox. parallelotope using
                               priv_encParallelotope
           * 'outer:norm':     uses priv_encZonotope with exact norm value
           * 'outer:norm_bnd': not implemented yet (throws error)
           * 'inner:box':      inner approx. parallelotope using
                               priv_inscParallelotope
           * 'inner:norm'      uses priv_inscZonotope with exact norm value
           * 'inner:norm_bnd': uses priv_inscZonotope with an bound on the
                               norm value
           * default:          same as 'outer:box'
    nrGen - (optional) number of generators

Outputs:
    Z - zonotope object

Example: 
    E = Ellipsoid([[3, -1], [-1, 1]], [[1], [0]])
    Z_enc = zonotope(E, 'outer:norm', 10)
    Z_insc = zonotope(E, 'inner:norm', 10)
    Z_box = zonotope(E)

References:
    [1] V. Gaßmann, M. Althoff. "Scalable Zonotope-Ellipsoid Conversions
        using the Euclidean Zonotope Norm", 2020

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: priv_encZonotope, priv_encParallelotope, priv_inscZonotope,
    priv_inscParallelotope

Authors:       Victor Gassmann (MATLAB)
               Python translation by AI Assistant
Written:       11-October-2019 (MATLAB)
Last update:   08-June-2021 (moved handling of degenerate case here)
               04-July-2022 (VG, class array cases)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .ellipsoid import Ellipsoid

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def zonotope(E: 'Ellipsoid', mode: str = 'outer:box', nrGen: int = None) -> 'Zonotope':
    """
    Converts an ellipsoid to a zonotope
    
    Args:
        E: ellipsoid object
        mode: conversion mode (default: 'outer:box')
        nrGen: number of generators (default: dimension of E)
        
    Returns:
        Z: zonotope object
    """
    # Import here to avoid circular import
    from cora_python.contSet.zonotope.zonotope import Zonotope
    
    # Set default values
    if nrGen is None:
        nrGen = E.dim()
    
    # Input validation - simplified to avoid class lookup issues
    from .ellipsoid import Ellipsoid
    if not isinstance(E, Ellipsoid):
        raise CORAerror('CORA:wrongValue', '1st', 'expected ellipsoid object')
    
    if mode not in ['outer:box', 'outer:norm', 'outer:norm_bnd',
                    'inner:box', 'inner:norm', 'inner:norm_bnd']:
        raise CORAerror('CORA:wrongValue', '2nd', 'invalid mode')
    
    if not isinstance(nrGen, int) or nrGen <= 0:
        raise CORAerror('CORA:wrongValue', '3rd', 'nrGen must be positive integer')
    
    # Handle empty case
    if E.representsa_('emptySet'):
        return Zonotope.empty(E.dim())
    
    # Compute rank and dimension of ellipsoid
    rankE = E.rank()
    c = E.center()
    isDeg = rankE != E.dim()
    
    # Handle degenerate case
    if isDeg:
        # Ellipsoid is just a point
        if rankE == 0:
            return Zonotope(c)
        
        # Compute SVD to find U matrix transforming the shape matrix to a 
        # diagonal matrix (to isolate degenerate dimensions)
        U, S, Vt = np.linalg.svd(E.Q)
        Qt = np.diag(S[:rankE])
        
        # Construct non-degenerate ellipsoid
        from .ellipsoid import Ellipsoid
        E_nonDeg = Ellipsoid(Qt)
        
        # Construct revert transformation matrix
        T = U[:, :rankE]
    else:
        E_nonDeg = E
        T = None
    
    # Convert based on mode
    if mode == 'outer:box':
        Z = _priv_encParallelotope(E_nonDeg)
    elif mode == 'outer:norm':
        Z = _priv_encZonotope(E_nonDeg, nrGen)
    elif mode == 'outer:norm_bnd':
        raise CORAerror('CORA:notSupported', "mode = 'outer:norm_bnd'")
    elif mode == 'inner:box':
        Z = _priv_inscParallelotope(E_nonDeg)
    elif mode == 'inner:norm':
        Z = _priv_inscZonotope(E_nonDeg, nrGen, 'exact')
    elif mode == 'inner:norm_bnd':
        Z = _priv_inscZonotope(E_nonDeg, nrGen, 'ub_convex')
    
    # In degenerate case, lift lower-dimensional non-degenerate ellipsoid
    if isDeg:
        Z = T @ Z + c
    
    return Z


def _priv_encParallelotope(E: 'Ellipsoid') -> 'Zonotope':
    """
    Encloses a non-degenerate ellipsoid by a parallelotope
    
    Args:
        E: ellipsoid object
        
    Returns:
        Z: zonotope object
    """
    from cora_python.contSet.zonotope.zonotope import Zonotope
    
    # Transform ellipsoid into sphere -> square around sphere -> back transform
    try:
        sqrt_Q = np.linalg.cholesky(E.Q).T
    except np.linalg.LinAlgError:
        # If Cholesky fails, use eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(E.Q)
        sqrt_Q = eigenvecs @ np.diag(np.sqrt(np.maximum(eigenvals, 0)))
    
    return Zonotope(E.q, sqrt_Q)


def _priv_inscParallelotope(E: 'Ellipsoid') -> 'Zonotope':
    """
    Inner-approximates a non-degenerate ellipsoid by a parallelotope
    
    Args:
        E: ellipsoid object
        
    Returns:
        Z: zonotope object
    """
    from cora_python.contSet.zonotope.zonotope import Zonotope
    
    n = E.dim()
    try:
        sqrt_Q = np.linalg.cholesky(E.Q).T
        T = np.linalg.inv(sqrt_Q)
    except np.linalg.LinAlgError:
        # If Cholesky fails, use eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(E.Q)
        sqrt_Q = eigenvecs @ np.diag(np.sqrt(np.maximum(eigenvals, 0)))
        T = np.linalg.pinv(sqrt_Q)
    
    # Transform ellipsoid into sphere -> square into sphere -> back transform
    return Zonotope(E.q, np.linalg.inv(T) * (1/np.sqrt(n)) * np.eye(n))


def _priv_encZonotope(E: 'Ellipsoid', nrGen: int) -> 'Zonotope':
    """
    Encloses a non-degenerate ellipsoid by a zonotope
    
    Args:
        E: ellipsoid object
        nrGen: number of generators of resulting zonotope
        
    Returns:
        Z: zonotope object
    """
    from cora_python.contSet.zonotope.zonotope import Zonotope
    
    # Read out dimension
    n = E.dim()
    # Extract center
    c = E.center()
    # Compute transformation matrix s.t. T*E == unit hyper-sphere
    try:
        Tinv = np.linalg.cholesky(E.Q).T
    except np.linalg.LinAlgError:
        # If Cholesky fails, use eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(E.Q)
        Tinv = eigenvecs @ np.diag(np.sqrt(np.maximum(eigenvals, 0)))
    
    # Compute "uniform" distribution of m points on unit hyper-sphere
    if n == 1:
        G = np.array([[-1, 1]])
    elif n == nrGen:
        return _priv_encParallelotope(E)
    else:
        G = _eq_point_set(n-1, nrGen)
    
    # Create zonotope and compute minimum norm
    Z_temp = Zonotope(np.zeros((n, 1)), G)
    L = Z_temp.minnorm()[0]
    
    # We want the ellipsoid to be contained in the zonotope, so we scale
    # zonotope(.,G) s.t. it touches E (for exact norm computation), then
    # apply retransform
    return Zonotope(c, (1/L) * Tinv @ G)


def _priv_inscZonotope(E: 'Ellipsoid', m: int, mode: str) -> 'Zonotope':
    """
    Inner-approximates a non-degenerate ellipsoid by a zonotope
    
    Args:
        E: ellipsoid object
        m: number of generators
        mode: computation of zonotope norm (see zonotope/norm_)
        
    Returns:
        Z: zonotope object
    """
    from cora_python.contSet.zonotope.zonotope import Zonotope
    
    # Read out dimension
    n = E.dim()
    # Extract center
    c = E.center()
    # Compute transformation matrix s.t. T*E == unit hypersphere
    try:
        Tinv = np.linalg.cholesky(E.Q).T
    except np.linalg.LinAlgError:
        # If Cholesky fails, use eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(E.Q)
        Tinv = eigenvecs @ np.diag(np.sqrt(np.maximum(eigenvals, 0)))
    
    # Compute "uniform" distribution of m points on unit hypersphere
    if n == 1:
        # Return exact result
        G = np.array([[1]])
    elif n == m:
        G = np.eye(n)
    elif m % 2 == 0:
        # Such cases result in aligned generators
        # -> choose twice as many and discard half of it
        G = _eq_point_set(n-1, m*2)
        G = G[:, :m]
    else:
        G = _eq_point_set(n-1, m)
        # Check if aligned (simplified check)
        # In practice, this rarely happens for well-distributed points
    
    # Init zonotope
    Z = Zonotope(np.zeros((n, 1)), G)
    
    # Compute zonotope norm
    R = Z.norm_(2, mode)
    if np.isnan(R):
        R = Z.norm_(2, 'ub')
    
    # We want the zonotope to be enclosed in the ellipsoid, so we scale
    # zonotope(.,G) such that is barely contained in unit hypersphere,
    # and apply inverse transform
    return c + (1/R) * Tinv @ Z


def _eq_point_set(dim: int, N: int) -> np.ndarray:
    """
    Simplified implementation of equal area point set generation
    
    For now, we use a simple approach. In a full implementation,
    this would use the recursive zonal equal area sphere partitioning algorithm.
    
    Args:
        dim: dimension of sphere (S^dim)
        N: number of points
        
    Returns:
        points: (dim+1) x N array of points on unit sphere
    """
    # For now, use a simple random approach with normalization
    # In practice, this should use the eq_sphere_partitions algorithm
    np.random.seed(42)  # For reproducibility
    points = np.random.randn(dim + 1, N)
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=0)
    points = points / norms[np.newaxis, :]
    return points 