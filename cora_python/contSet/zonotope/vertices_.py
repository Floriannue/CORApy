"""
vertices_ - returns potential vertices of a zonotope

WARNING: Do not use this function for high-order zonotopes as the
computational complexity grows exponentially!

Authors: Matthias Althoff, Niklas Kochdumper, Daniel Heß (MATLAB)
         Python translation by AI Assistant
Written: 14-September-2006 (MATLAB)
Last update: 11-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from scipy.spatial import ConvexHull
from scipy.linalg import svd
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from typing import TYPE_CHECKING
from itertools import product

if TYPE_CHECKING:
    from .zonotope import Zonotope


def vertices_(Z: 'Zonotope', alg: str = 'convHull') -> np.ndarray:
    """
    Returns potential vertices of a zonotope
    
    Args:
        Z: Zonotope object
        alg: Algorithm used:
             - 'convHull' (default): Convex hull computation
             - 'iterate': Iterative vertex computation
             - 'polytope': Convert to polytope first
        
    Returns:
        np.ndarray: Vertices (each column is a vertex)
        
    Raises:
        CORAerror: If computation fails
        
    Example:
        >>> Z = Zonotope([1, -1], [[1, 3, -2, 1, 0], [0, 2, 1, -2, 1]])
        >>> V = vertices_(Z)
    """
    # Different cases for different dimensions
    n = Z.dim()
    
    if n == 1:
        # Compute the two vertices for one-dimensional case
        c = Z.c
        temp = np.sum(np.abs(Z.G), axis=1)
        V = np.column_stack([c - temp, c + temp])
        
    elif n == 2:
        # Use fast method for 2D
        V = _aux_vertices2Dfast(Z)
        
    else:
        # Apply the selected algorithm
        if alg == 'iterate':
            V = _aux_verticesIterate(Z)
        elif alg == 'polytope':
            V = _aux_verticesPolytope(Z)
        else:
            V = _aux_verticesConvHull(Z)
    
    # Remove duplicates
    if V.size > 0:
        # Convert to unique rows and back
        V_unique = np.unique(V.T, axis=0)
        V = V_unique.T
    
    return V


def _aux_vertices2Dfast(Z: 'Zonotope') -> np.ndarray:
    """Fast method for 2D zonotopes"""
    # Empty case
    if Z.representsa_('emptySet', 1e-15):
        return np.zeros((Z.dim(), 0))
    
    # Delete zero generators
    Z = Z.compact_('zeros', 1e-15)
    
    # Obtain center and generator matrix
    c = Z.c
    G = Z.G
    
    # Obtain number of generators
    nrGens = G.shape[1]
    
    if nrGens == 0:
        return c.reshape(-1, 1)
    
    # Obtain size of enclosing interval hull of first two dimensions
    xmax = np.sum(np.abs(G[0, :]))
    ymax = np.sum(np.abs(G[1, :]))
    
    # Z with normalized direction: All generators pointing "up"
    Gnorm = G.copy()
    Gnorm[:, G[1, :] < 0] = -G[:, G[1, :] < 0]
    
    # Compute angles
    angles = np.arctan2(Gnorm[1, :], Gnorm[0, :])
    angles[angles < 0] = angles[angles < 0] + 2 * np.pi
    
    # Sort all generators by their angle
    IX = np.argsort(angles)
    
    # Cumsum the generators in order of angle
    V = np.zeros((2, nrGens + 1))
    for i in range(nrGens):
        V[:, i + 1] = V[:, i] + 2 * Gnorm[:, IX[i]]
    
    V[0, :] = V[0, :] + xmax - np.max(V[0, :])
    V[1, :] = V[1, :] - ymax
    
    # Flip/mirror upper half to get lower half of zonotope (point symmetry)
    V_lower = np.column_stack([
        V[0, -1] + V[0, 0] - V[0, 1:-1][::-1],
        V[1, -1] + V[1, 0] - V[1, 1:-1][::-1]
    ])
    
    V = np.column_stack([V, V_lower.T])
    
    # Consider center
    V = c.reshape(-1, 1) + V
    
    return V


def _aux_verticesPolytope(Z: 'Zonotope') -> np.ndarray:
    """Convert to polytope and compute vertices"""
    # This would require polytope implementation
    # For now, fall back to convex hull method
    return _aux_verticesConvHull(Z)


def _aux_verticesConvHull(Z: 'Zonotope') -> np.ndarray:
    """Compute vertices using convex hull"""
    # First vertex is the center of the zonotope
    c = Z.c
    G = Z.G
    n = Z.dim()
    nrGens = G.shape[1]
    
    if nrGens == 0:
        return c.reshape(-1, 1)
    
    # We project vertices to lower-dims if degenerate
    # Starting with 1 point (center); always degenerate
    V = c.reshape(-1, 1)
    last_n = 0
    
    # Generate further potential vertices in the loop
    for i in range(nrGens):
        # Expand vertices with current generator
        g = G[:, i].reshape(-1, 1)
        V = np.column_stack([V - g, V + g])
        
        # Remove inner points
        try:
            if last_n < n:
                # Last iteration it was degenerate,
                # assuming it still is; project to lower-dim
                
                # Shift by center
                V_centered = V - c.reshape(-1, 1)
                
                # Compute projection matrix via SVD
                U, S, Vt = svd(V_centered, full_matrices=False)
                tol = 1e-10
                P = U[:, S > tol]
                
                # Project into low-dim space
                V_proj = P.T @ V_centered
                
                # Check if still degenerate
                last_n = V_proj.shape[0]
                
                if last_n == 1:
                    # 1-dim can be computed quickly
                    imin = np.argmin(V_proj[0, :])
                    imax = np.argmax(V_proj[0, :])
                    indices = [imin, imax]
                elif last_n >= 2:
                    # Compute convex hull
                    try:
                        hull = ConvexHull(V_proj.T)
                        indices = np.unique(hull.vertices)
                    except:
                        # Fallback: use all points
                        indices = np.arange(V.shape[1])
                else:
                    indices = np.arange(V.shape[1])
                
                # Select corresponding points of original matrix
                V = V[:, indices]
                
            else:
                # Non-degenerate; normal convex hull computation
                try:
                    hull = ConvexHull(V.T)
                    indices = np.unique(hull.vertices)
                    V = V[:, indices]
                except:
                    # If convex hull fails, keep all points
                    pass
                    
        except Exception as e:
            # If anything fails, continue with current vertices
            pass
    
    return V


def _aux_verticesIterate(Z: 'Zonotope') -> np.ndarray:
    """Iterative vertex computation"""
    c = Z.c
    G = Z.G
    n = Z.dim()
    nrGens = G.shape[1]
    
    if nrGens == 0:
        return c.reshape(-1, 1)
    
    # Generate all possible combinations of ±1 for generators
    # This is exponential in the number of generators!
    if nrGens > 20:  # Prevent memory explosion
        raise CORAerror('CORA:dimensionMismatch', 
                       'Too many generators for iterative method')
    
    # Generate all combinations
    combinations = list(product([-1, 1], repeat=nrGens))
    
    # Compute all potential vertices
    V = np.zeros((n, len(combinations)))
    for i, combo in enumerate(combinations):
        factors = np.array(combo)
        V[:, i] = c + G @ factors
    
    # Remove duplicates and compute convex hull
    if V.shape[1] > 1:
        try:
            hull = ConvexHull(V.T)
            V = V[:, hull.vertices]
        except:
            # If convex hull fails, remove duplicates manually
            V_unique = np.unique(V.T, axis=0)
            V = V_unique.T
    
    return V 