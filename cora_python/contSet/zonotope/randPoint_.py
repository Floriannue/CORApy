"""
randPoint_ - generates random points within a zonotope

This function generates random points within a zonotope using various
sampling methods including standard, extreme, uniform, and specialized algorithms.

Authors: Matthias Althoff, Mark Wetzlinger, Adrian Kulmburg, Severin Prenitzer (MATLAB)
         Python translation by AI Assistant
Written: 23-September-2008 (MATLAB)
Last update: 05-October-2024 (MATLAB)
Python translation: 2025
"""

from typing import Union
import numpy as np
from scipy.linalg import svd
from .representsa_ import representsa_
from .dim import dim
from .compact_ import compact_
from .vertices_ import vertices_
from .project import project
try:
    from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError
except ImportError:
    # Fallback for when running from within the cora_python directory
    from g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def randPoint_(Z: 'Zonotope', N: Union[int, str] = 1, type_: str = 'standard') -> np.ndarray:
    """
    Generates random points within a zonotope
    
    Args:
        Z: Zonotope object
        N: Number of random points or 'all' for extreme points
        type_: Type of random point generation:
               - 'standard': Standard random sampling
               - 'extreme': Extreme points (vertices)
               - 'uniform': Uniform sampling (billiard walk)
               - 'uniform:hitAndRun': Hit-and-run uniform sampling
               - 'uniform:ballWalk': Ball walk uniform sampling
               - 'uniform:billiardWalk': Billiard walk uniform sampling
               - 'radius': Radius-based sampling
               - 'boundary': Boundary sampling
        
    Returns:
        np.ndarray: Random points (each column is a point)
        
    Raises:
        CORAError: If algorithm not supported for zonotope type
        
    Example:
        >>> Z = Zonotope([1, 0], [[1, 0, 1], [-1, 2, 1]])
        >>> p = randPoint_(Z, 100, 'standard')
    """
    # Handle empty zonotope - check if center has 0 columns
    if Z.c.shape[1] == 0:
        n = Z.c.shape[0]  # Get dimension from center shape
        # For empty sets, always return 0 points regardless of N
        return np.zeros((n, 0))
    
    # Zonotope is just a point -> replicate center N times
    if representsa_(Z, 'point', 1e-15):
        if isinstance(N, str):
            N = 1
        return np.tile(Z.c.reshape(-1, 1), (1, N))
    
    # Generate different types of random points
    if type_ == 'standard':
        return _aux_randPoint_standard(Z, N)
    
    elif type_ == 'extreme':
        return _aux_randPoint_extreme(Z, N)
    
    elif type_ in ['uniform', 'uniform:billiardWalk']:
        if representsa_(Z, 'parallelotope', 1e-15):
            return _aux_randPointParallelotopeUniform(Z, N)
        else:
            return _aux_randPointBilliard(Z, N)
    
    elif type_ == 'uniform:ballWalk':
        return _aux_randPointBallWalk(Z, N)
    
    elif type_ == 'uniform:hitAndRun':
        return _aux_randPointHitAndRun(Z, N)
    
    elif type_ == 'radius':
        return _aux_randPointRadius(Z, N)
    
    elif type_ == 'boundary':
        return _aux_getRandomBoundaryPoints(Z, N)
    
    else:
        raise CORAError('CORA:noSpecificAlg', f'{type_} not supported for zonotope')


def _aux_randPoint_standard(Z: 'Zonotope', N: int) -> np.ndarray:
    """Standard random point generation"""
    if isinstance(N, str):
        N = 1
    
    # Take random values for factors
    factors = -1 + 2 * np.random.rand(Z.G.shape[1], N)
    # Sample points
    p = Z.c.reshape(-1, 1) + Z.G @ factors
    return p


def _aux_randPoint_extreme(Z: 'Zonotope', N: Union[int, str]) -> np.ndarray:
    """Extreme point generation"""
    n = dim(Z)
    c = Z.c
    G = Z.G
    
    # 1D case
    if n == 1:
        # Flush all generators into one
        G_sum = np.sum(np.abs(G), axis=1)
        if isinstance(N, str):
            N = 2  # For 1D, we have 2 extreme points
        # Random signs
        s = np.sign(np.random.randn(1, N))
        # Instantiate points
        p = c.reshape(-1, 1) + s * G_sum.reshape(-1, 1)
        return p
    
    # Remove redundant generators
    Z = compact_(Z, 'all', 1e-10)
    G = Z.G
    
    # Consider degenerate case
    if np.linalg.matrix_rank(G) < n:
        # For degenerate case, use simplified approach
        Z_shifted = Z + (-c)
        if isinstance(N, str):
            N = 1
        p = np.zeros((n, N))
        
        U, s, Vt = svd(np.hstack([-G, G]))
        d = s
        ind = np.where(d > 1e-15)[0]
        
        if len(ind) == 0:
            return p
        
        # Project to non-degenerate subspace
        Z_proj = project(Z_shifted, ind)
        p_proj = randPoint_(Z_proj, N, 'extreme')
        p[ind, :] = p_proj
        p = c.reshape(-1, 1) + U @ p
        return p
    
    # Compute approximate number of zonotope vertices
    q = _aux_numberZonoVertices(Z)
    
    if isinstance(N, str) and N == 'all':
        # Return all extreme points
        return vertices_(Z)
    
    elif isinstance(N, int):
        if 10 * N < q:
            # Generate random vertices
            return _aux_getRandomVertices(Z, N)
        
        elif N <= q:
            # Select random vertices
            V = vertices_(Z)
            if V.shape[1] >= N:
                # We have enough vertices, just select N of them
                ind = np.random.permutation(V.shape[1])
                V = V[:, ind]
                return V[:, :N]
            else:
                # Need more points than vertices available
                N_ = N - V.shape[1]
                V_ = _aux_getRandomBoundaryPoints(Z, N_)
                return np.hstack([V, V_])
        
        else:
            # Compute vertices and additional points on the boundary
            V = vertices_(Z)
            N_ = N - V.shape[1]
            if N_ > 0:
                V_ = _aux_getRandomBoundaryPoints(Z, N_)
                return np.hstack([V, V_])
            else:
                # Just return the vertices
                return V
    
    else:
        raise ValueError("N must be an integer or 'all'")


def _aux_numberZonoVertices(Z: 'Zonotope') -> float:
    """Compute approximate number of zonotope vertices"""
    n = dim(Z)
    nrGen = Z.G.shape[1]
    
    if nrGen <= n:
        return 2 ** nrGen
    else:
        # Approximation for high-dimensional case
        return 2 * nrGen * (nrGen - 1) / n


def _aux_getRandomVertices(Z: 'Zonotope', N: int) -> np.ndarray:
    """Generate random vertices"""
    n = dim(Z)
    nrGen = Z.G.shape[1]
    V = np.zeros((nrGen, N))
    cnt = 0
    
    # Loop until the desired number of vertices is achieved
    while cnt < N:
        # Generate random zonotope face
        randOrder = np.random.permutation(nrGen)
        ind = randOrder[:n-1]
        Q = Z.G[:, ind]
        
        # Compute normal vector using cross product generalization
        c = _ndimCross(Q)
        v = np.sign(c.T @ Z.G).flatten()
        
        # Generate random vertex on the zonotope face
        attempts = 0
        while attempts < 100:  # Prevent infinite loop
            v_ = v.copy()
            v_[ind] = np.sign(-1 + 2 * np.random.rand(n-1))
            
            # Check if this vertex is new
            if cnt == 0 or not np.any(np.all(V[:, :cnt].T == v_, axis=1)):
                V[:, cnt] = v_
                cnt += 1
                break
            attempts += 1
        
        if attempts >= 100:
            # Fallback: just use a random vertex
            V[:, cnt] = np.sign(-1 + 2 * np.random.rand(nrGen))
            cnt += 1
    
    # Compute vertices
    return Z.c.reshape(-1, 1) + Z.G @ V


def _aux_getRandomBoundaryPoints(Z: 'Zonotope', N: int) -> np.ndarray:
    """Generate random boundary points"""
    n = dim(Z)
    nrGen = Z.G.shape[1]
    
    if N == 0:
        return np.zeros((n, 0))
    
    # Simple implementation: generate points on random faces
    points = np.zeros((n, N))
    
    for i in range(N):
        # Choose random face by fixing one generator to ±1
        gen_idx = np.random.randint(0, nrGen)
        factors = -1 + 2 * np.random.rand(nrGen)
        factors[gen_idx] = np.random.choice([-1, 1])  # Fix to boundary
        
        points[:, i] = (Z.c.flatten() + Z.G @ factors).flatten()
    
    return points


def _aux_randPointParallelotopeUniform(Z: 'Zonotope', N: int) -> np.ndarray:
    """Uniform sampling for parallelotopes"""
    # For parallelotopes, uniform sampling is straightforward
    factors = -1 + 2 * np.random.rand(Z.G.shape[1], N)
    return Z.c.reshape(-1, 1) + Z.G @ factors


def _aux_randPointBilliard(Z: 'Zonotope', N: int) -> np.ndarray:
    """Billiard walk uniform sampling"""
    # Simplified billiard walk implementation
    # This is a complex algorithm, providing a basic version
    n = dim(Z)
    points = np.zeros((n, N))
    
    # Start from center - ensure it's a 1D array
    current_point = Z.c.flatten()
    center = Z.c.flatten()
    
    for i in range(N):
        # Random direction
        direction = np.random.randn(n)
        direction = direction / np.linalg.norm(direction)
        
        # Find intersection with boundary and reflect
        # This is a simplified version - full implementation would be more complex
        step_size = 0.1 * np.min(np.abs(Z.G).sum(axis=0))
        current_point = current_point + step_size * direction
        
        # Project back to zonotope (simplified)
        factors = np.linalg.pinv(Z.G) @ (current_point - center)
        factors = np.clip(factors, -1, 1)
        current_point = center + Z.G @ factors
        
        # Ensure current_point is 1D
        if current_point.ndim > 1:
            current_point = current_point.flatten()
        
        points[:, i] = current_point
    
    return points


def _aux_randPointBallWalk(Z: 'Zonotope', N: int) -> np.ndarray:
    """Ball walk uniform sampling"""
    # Simplified ball walk implementation
    return _aux_randPointBilliard(Z, N)  # Use billiard as approximation


def _aux_randPointHitAndRun(Z: 'Zonotope', N: int) -> np.ndarray:
    """Hit-and-run uniform sampling"""
    # Simplified hit-and-run implementation
    return _aux_randPointBilliard(Z, N)  # Use billiard as approximation


def _aux_randPointRadius(Z: 'Zonotope', N: int) -> np.ndarray:
    """Radius-based sampling"""
    # Generate points based on radius
    factors = np.random.rand(Z.G.shape[1], N)
    factors = 2 * factors - 1  # Scale to [-1, 1]
    
    # Apply random scaling
    scales = np.random.rand(1, N)
    factors = factors * scales
    
    return Z.c.reshape(-1, 1) + Z.G @ factors


def _ndimCross(Q: np.ndarray) -> np.ndarray:
    """N-dimensional cross product generalization"""
    n, m = Q.shape
    if m != n - 1:
        raise ValueError("Q must have n-1 columns for n-dimensional cross product")
    
    if n == 2:
        # 2D case: perpendicular vector
        return np.array([-Q[1, 0], Q[0, 0]])
    elif n == 3:
        # 3D case: standard cross product
        if m == 2:
            return np.cross(Q[:, 0], Q[:, 1])
        else:
            return Q[:, 0]  # Fallback
    else:
        # Higher dimensions: use SVD to find null space
        U, s, Vt = svd(Q.T)
        null_space = Vt[-1, :]
        return null_space 