"""
constraints - computes the halfspace representation of a polytope

Syntax:
    P = constraints(P)

Inputs:
    P - polytope object

Outputs:
    P - polytope object with halfspace representation

Example:
    V = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
    P = Polytope(V)
    P = constraints(P)
    # Now P.A and P.b are available

Authors:       Tobias Ladner, Mark Wetzlinger
Written:       16-July-2024
Last update:   ---
Last revision: ---
"""

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .polytope import Polytope

def constraints(P: 'Polytope') -> 'Polytope':
    """
    Computes the halfspace representation of a polytope from its vertex representation.
    
    Args:
        P: Polytope object
        
    Returns:
        P: Same polytope object with halfspace representation computed
    """
    # Check if halfspace representation already available
    if P.isHRep:
        return P
        
    # Check if vertex representation is available
    if not P.isVRep or P.V is None:
        raise CORAerror('CORA:specialError',
                       'Cannot compute constraints: no vertex representation available.')
    
    # Read out vertices
    V = P.V
    
    # Read out dimension and number of vertices
    if V.ndim == 1:
        # 1D case: V is a single value or array of values
        n = 1
        nrVert = V.size
        # Reshape to 2D for consistency
        V = V.reshape(1, -1)
    else:
        n, nrVert = V.shape
    
    # Special cases
    if nrVert == 0:
        # Empty polytope: no vertices, no constraints
        A = np.zeros((0, n))
        b = np.array([]).reshape(-1, 1)
        Ae = np.zeros((0, n))
        be = np.array([]).reshape(-1, 1)
    elif nrVert == 1:
        # Simple method for a single vertex: { x | I * x = V }
        A = np.zeros((0, n))
        b = np.array([]).reshape(-1, 1)
        Ae = np.eye(n)
        be = V
    elif n == 1:
        # 1D case
        A, b, Ae, be = _aux_1D(V)
    elif n == 2 and (nrVert <= 2 or np.linalg.matrix_rank(V) < 2):
        # 2D-degenerate case
        A, b, Ae, be = _aux_2D_degenerate(V)
    else:
        # General case
        A, b, Ae, be = _aux_nD(V, n)
    
    # Normalize None to empty arrays for equalities
    if Ae is None:
        Ae = np.zeros((0, n))
    if be is None:
        be = np.zeros((0, 1))

    # Save to object
    P._A = A
    P._b = b
    P._Ae = Ae
    P._be = be
    # Halfspace representation now computed
    P.isHRep = True
    
    # Reset lazy computation cache values, as the underlying representation has changed
    P._reset_lazy_flags()
    
    return P


def _aux_1D(V):
    """Auxiliary function for 1D case"""
    # Handle empty vertices case
    if V.size == 0:
        # Empty polytope: no constraints needed, but this case should be handled upstream
        A = np.zeros((0, 1))
        b = np.array([]).reshape(-1, 1)
        Ae = np.zeros((0, 1))
        be = np.array([]).reshape(-1, 1)
        return A, b, Ae, be
    
    # Find minimum and maximum value
    maxV = np.max(V)
    minV = np.min(V)
    
    # Check unboundedness
    maxV_inf = maxV == np.inf
    minV_inf = minV == -np.inf
    
    # Generally two inequality constraints
    # x <= maximum value && x >= minimum value
    if maxV_inf and minV_inf:
        # Both are infinity -> no constraints
        A = np.zeros((0, 1))
        b = np.array([]).reshape(-1, 1)
        Ae = np.zeros((0, 1))
        be = np.array([]).reshape(-1, 1)
        
    elif maxV_inf and not minV_inf:
        # Only bounded from below
        A = np.array([[-1]])
        b = np.array([[-minV]])
        Ae = np.zeros((0, 1))
        be = np.array([]).reshape(-1, 1)
        
    elif not maxV_inf and minV_inf:
        # Only bounded from above
        A = np.array([[1]])
        b = np.array([[maxV]])
        Ae = np.zeros((0, 1))
        be = np.array([]).reshape(-1, 1)
        
    elif withinTol(maxV, minV):
        # Single point -> use equality constraint
        A = np.zeros((0, 1))
        b = np.array([]).reshape(-1, 1)
        Ae = np.array([[1]])
        be = np.array([[maxV]])
        
    else:
        # Bounded from above and below, not a single point
        A = np.array([[1], [-1]])
        b = np.array([[maxV], [-minV]])
        Ae = np.zeros((0, 1))
        be = np.array([]).reshape(-1, 1)
    
    return A, b, Ae, be


def _aux_2D_degenerate(V):
    """Auxiliary function for 2D degenerate case"""
    Ae = None
    be = None

    if V.shape[1] == 0: # Handle empty V (no vertices)
        return np.zeros((0, V.shape[0])), np.array([]).reshape(-1, 1), np.zeros((0, V.shape[0])), np.array([]).reshape(-1, 1)

    # Only one distinct point given
    if np.allclose(V - V[:, [0]], 0):
        # Use axis-aligned normal vectors for simplicity
        A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        b = np.array([[V[0, 0]], [V[1, 0]], [-V[0, 0]], [-V[1, 0]]])
        return A, b, Ae, be
    
    # Collinear vertices -> polytope is just a line
    
    # Find minimum and maximum of line using x-value
    minIdx = np.argmin(V[0, :])
    maxIdx = np.argmax(V[0, :])
    
    # If indices are equal, that means x-value is constant -> choose y value for sorting
    if minIdx == maxIdx:
        minIdx = np.argmin(V[1, :])
        maxIdx = np.argmax(V[1, :])
    
    # Compute vector along the line and normalize
    dir_vec = V[:, maxIdx] - V[:, minIdx]
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    
    # Fill in normal vectors, including 90° turn
    dir_perp = np.array([-dir_vec[1], dir_vec[0]])
    A = np.array([dir_vec, -dir_vec, dir_perp, -dir_perp])
    b = np.zeros((4, 1))
    
    # Compute offset of start and end
    b[0, 0] = dir_vec @ V[:, maxIdx]
    b[1, 0] = -dir_vec @ V[:, minIdx]
    # For orthogonal normal vectors, we can use either maxIdx or minIdx
    b[2, 0] = dir_perp @ V[:, maxIdx]
    b[3, 0] = -dir_perp @ V[:, maxIdx]
    
    return A, b, Ae, be


def _aux_nD(V, n):
    """General case in nD"""
    # Shift vertices by mean
    c = np.mean(V, axis=1, keepdims=True)
    V_centered = V - c
    
    # Check for degeneracy (mean already subtracted)
    U, s, _ = np.linalg.svd(V_centered)
    # Dimension of subspace
    r = np.sum(~withinTol(s, 0, 1e-10))
    
    # Project vertices onto subspace
    if r < n:
        # Project vertices onto basis
        V_proj = U[:, :r].T @ V_centered
    else:
        V_proj = V_centered
    
    # Compute convex hull (possibly in subspace)
    try:
        hull = ConvexHull(V_proj.T)
        # Get equations in form Ax + b <= 0, we want Ax <= -b
        A_proj = hull.equations[:, :-1]
        b_proj = -hull.equations[:, -1:]
    except (QhullError, ValueError):
        try:
            # Try with different options
            hull = ConvexHull(V_proj.T, qhull_options='QJ')
            A_proj = hull.equations[:, :-1]
            b_proj = -hull.equations[:, -1:]
        except (QhullError, ValueError):
            # If still failing, return empty constraints
            A = np.zeros((0, n))
            b = np.array([]).reshape(-1, 1)
            Ae = np.zeros((0, n))
            be = np.array([]).reshape(-1, 1)
            return A, b, Ae, be
    
    # Project back to original space if needed
    if r < n:
        # Back-projection
        A = A_proj @ U[:, :r].T
        # Incorporate original mean into offset
        b = b_proj + A @ c
        
        # Equality constraints where singular values are (almost) zero
        Ae = U[:, r:].T
        # Translate offset by original center
        be = Ae @ c
    else:
        A = A_proj
        b = b_proj + A @ c
        Ae = np.zeros((0, n))
        be = np.array([]).reshape(-1, 1)
    
    return A, b, Ae, be 