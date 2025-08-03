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
from cora_python.g.functions.matlab.validate.postprocessing.CORAwarning import CORAwarning
from cora_python.g.functions.matlab.validate.check import withinTol
from typing import TYPE_CHECKING
from itertools import product, combinations

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
        # Convert to unique rows and back (matching MATLAB's unique(V','rows','stable')')
        V_unique = np.unique(V.T, axis=0)
        V = V_unique.T
    
    return V


def _aux_vertices2Dfast(Z: 'Zonotope') -> np.ndarray:
    """Fast method for 2D zonotopes - exact MATLAB translation"""
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
    # Ensure all angles are in [0, pi] range (matching MATLAB assertion)
    angles[angles > np.pi] = angles[angles > np.pi] - np.pi
    
    # Sort all generators by their angle
    IX = np.argsort(angles)
    
    # Cumsum the generators in order of angle
    V = np.zeros((2, nrGens + 1))
    for i in range(nrGens):
        V[:, i + 1] = V[:, i] + 2 * Gnorm[:, IX[i]]
    
    V[0, :] = V[0, :] + xmax - np.max(V[0, :])
    V[1, :] = V[1, :] - ymax
    
    # Flip/mirror upper half to get lower half of zonotope (point symmetry)
    # MATLAB: V = [V(1,:),V(1,end)+V(1,1)-V(1,2:end);...
    #              V(2,:),V(2,end)+V(2,1)-V(2,2:end)];
    # V(1,2:end) means from index 2 to end (1-based indexing)
    # In Python, this is V[0, 1:] (0-based indexing)
    V_lower_x = V[0, -1] + V[0, 0] - V[0, 1:][::-1]
    V_lower_y = V[1, -1] + V[1, 0] - V[1, 1:][::-1]
    
    # Combine exactly as MATLAB does: [upper_half, lower_half]
    # MATLAB creates a 2x(2*nrGens+1) matrix
    V = np.vstack([
        np.concatenate([V[0, :], V_lower_x]),
        np.concatenate([V[1, :], V_lower_y])
    ])
    
    # Consider center
    V = c.reshape(-1, 1) + V
    
    return V


def _aux_verticesPolytope(Z: 'Zonotope') -> np.ndarray:
    """Convert to polytope and compute vertices"""
    from .polytope import polytope
    from ..polytope.vertices_ import vertices_
    
    P = polytope(Z)
    V = vertices_(P)
    return V


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
                P = U[:, ~withinTol(S, 0)]
                
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
            CORAwarning('CORA:contSet', message='Convex hull failed. Continuing...')
    
    return V


def _aux_verticesIterate(Z: 'Zonotope') -> np.ndarray:
    """Iterative vertex computation - complete MATLAB implementation"""
    # Delete aligned and all-zero generators
    Z = Z.compact_('zeros', 1e-15)
    Z = Z.compact_('aligned', 1e-3)
    
    # Extract object data
    G = Z.generators()
    c = Z.center()
    n = G.shape[0]
    
    # Catch the case where the zonotope is not full-dimensional
    if G.shape[1] < n:
        V, suc = _aux_verticesIterateSVG(Z)
        if suc:
            return V
        else:
            # Fallback to convex hull method
            return _aux_verticesConvHull(Z)
    
    # Compute vertices of the parallelotope
    from ..interval.vertices_ import vertices_ as interval_vertices
    from ..interval.interval import Interval
    
    # Create interval [-1,1]^n
    inf = -np.ones(n)
    sup = np.ones(n)
    I = Interval(inf, sup)
    vert = interval_vertices(I)
    V = c.reshape(-1, 1) + G[:, :n] @ vert
    
    # Compute halfspaces of the parallelotope
    from .polytope import polytope
    try:
        P, _, isDeg = polytope(Zonotope(c, G[:, :n]), 'exact')
        if isDeg:
            V, suc = _aux_verticesIterateSVG(Z)
            if suc:
                return V
            else:
                return _aux_verticesConvHull(Z)
        else:
            A = P._A
    except:
        # If polytope conversion fails, fallback
        return _aux_verticesConvHull(Z)
    
    # Loop over all remaining generators
    for i in range(n, G.shape[1]):
        # Extract current generator
        g = G[:, i]
        
        # Compute potential vertices
        V = np.column_stack([V + g.reshape(-1, 1), V - g.reshape(-1, 1)])
        
        # Compute new halfspaces
        if n == 2:
            temp = _ndimCross(g.reshape(-1, 1))
            temp = temp / np.linalg.norm(temp)
            Anew = np.vstack([temp.T, -temp.T])
        else:
            # Generate combinations of n-2 generators
            comb = list(combinations(range(i), n-2))
            Anew = np.zeros((2 * len(comb), n))
            counter = 0
            
            for j, combo in enumerate(comb):
                # Get subset of generators plus current generator
                G_subset = np.column_stack([G[:, list(combo)], g.reshape(-1, 1)])
                temp = _ndimCross(G_subset)
                temp = temp / np.linalg.norm(temp)
                Anew[counter, :] = temp.T
                Anew[counter + 1, :] = -temp.T
                counter += 2
        
        A = np.vstack([A, Anew])
        
        # Compute halfspace offsets
        b = np.max(A @ V, axis=1)
        
        # Remove redundant vertices
        temp = np.max(A @ V - b.reshape(-1, 1), axis=0)
        nV = _aux_numVertices(i + 1, n)
        ind = np.argsort(temp)[::-1]  # Sort in descending order
        V = V[:, ind[:nV]]
    
    return V


def _aux_verticesIterateSVG(Z: 'Zonotope') -> tuple:
    """Compute vertices for the case that zonotope is not full-dimensional"""
    # Extract object data
    G = Z.generators()
    c = Z.center()
    n = Z.dim()
    
    # Singular value decomposition
    U, S, Vt = svd(G)
    
    if Vt.shape[0] < G.shape[0]:
        # Pad Vt with zeros if needed
        Vt_padded = np.zeros((G.shape[0], G.shape[0]))
        Vt_padded[:Vt.shape[0], :Vt.shape[1]] = Vt
        Vt = Vt_padded
    
    # State space transformation
    Z_transformed = U.T @ np.column_stack([c, G])
    
    # Remove dimensions with all zeros
    ind = np.where(np.diag(S) <= 1e-12)[0]
    ind_active = np.setdiff1d(np.arange(S.shape[0]), ind)
    
    if len(ind) > 0:
        # Compute vertices in transformed space
        Z_reduced = Zonotope(Z_transformed[ind_active, 0], Z_transformed[ind_active, 1:])
        V = vertices_(Z_reduced)
        
        # Transform back to original space
        V_full = np.zeros((n, V.shape[1]))
        V_full[ind_active, :] = V
        res = U @ V_full
        return res, True
    else:
        return np.array([]), False


def _aux_numVertices(m: int, n: int) -> int:
    """Compute number of zonotope vertices"""
    res = 0
    for i in range(n):
        res += _nchoosek(m - 1, i)
    res = 2 * res
    return res


def _nchoosek(n: int, k: int) -> int:
    """Compute binomial coefficient n choose k"""
    if k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)  # Take advantage of symmetry
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c


def _ndimCross(vectors: np.ndarray) -> np.ndarray:
    """Compute n-dimensional cross product for n-1 vectors in n dimensions"""
    n, k = vectors.shape
    
    if k != n - 1:
        raise ValueError("Need exactly n-1 vectors for n-dimensional cross product")
    
    if n == 2:
        # 2D case: cross product of 1 vector
        v = vectors[:, 0]
        return np.array([-v[1], v[0]])
    elif n == 3:
        # 3D case: cross product of 2 vectors
        v1, v2 = vectors[:, 0], vectors[:, 1]
        return np.cross(v1, v2)
    else:
        # General n-dimensional case using determinant
        result = np.zeros(n)
        for i in range(n):
            # Create submatrix by removing i-th row
            rows_to_keep = list(range(i)) + list(range(i+1, n))
            submatrix = vectors[rows_to_keep, :]
            
            # Compute determinant with alternating sign
            result[i] = ((-1) ** (i + 1)) * np.linalg.det(submatrix)
        return result 