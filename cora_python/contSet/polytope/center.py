"""
center - computes the Chebyshev center of a polytope

Note: polytope in vertex representation are converted to halfspace
representation, see [1], which is potentially time-consuming
Use method 'avg' for average of vertices

Syntax:
    c = center(P)
    c = center(P, method)

Inputs:
    P - polytope object
    method - 'chebyshev', 'avg' (for v-polytope)

Outputs:
    c - Chebyshev center of the polytope

Example:
    P = polytope([[-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]], [2, 3, 2, 3, 2])
    c = center(P)

Reference:
    [1] M. Wetzlinger, V. Kotsev, A. Kulmburg, M. Althoff. "Implementation
        of Polyhedral Operations in CORA 2024", ARCH'24.

Authors: Viktor Kotsev, Mark Wetzlinger, Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 28-March-2022 (MATLAB)
Last update: 25-February-2025 (TL, unconstrained polytope returns origin)
Python translation: 2025
"""

import numpy as np
from scipy.optimize import linprog
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from .polytope import Polytope

from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def center(P: 'Polytope', method: str = 'chebyshev') -> np.ndarray:
    """
    Computes the Chebyshev center of a polytope
    
    Args:
        P: polytope object
        method: 'chebyshev' or 'avg' (for v-polytope)
        
    Returns:
        c: Chebyshev center of the polytope
    """
    # Parse input
    # The function signature already handles the default value for 'method'
    # method = setDefaultValues(['chebyshev'], [method])[0] if method != 'chebyshev' else 'chebyshev'
    allowed_methods = ['chebyshev', 'avg']
    
    # Input validation
    inputArgsCheck([
        [P, 'att', 'polytope'],
        [method, 'str', allowed_methods]
    ])
    
    # Read out dimension
    n = P.dim()
    
    # Fullspace/empty case
    if P.representsa_('fullspace', 0):
        # Return origin; consistent with fullspace/center
        return np.zeros((n, 1))
    elif P.emptySet: # Simplified check for emptySet property
        # Return empty
        return np.zeros((n, 0))
    
    # Fast and simple computation for 1D
    if n == 1:
        return _aux_center_1D(P)
    
    # Switch method
    if method == 'chebyshev':
        return _aux_center_chebyshev(P)
    elif method == 'avg':
        return _aux_center_avg(P)
    else:
        raise CORAerror('CORA:wrongValue', f'Invalid method. Allowed: {allowed_methods}')


def _aux_center_1D(P: 'Polytope') -> np.ndarray:
    """Special method for 1D polytopes"""
    # Compute vertices
    V = P.vertices_()

    # For 1D, the Chebyshev center is simply the midpoint of the interval
    # bounded by the min/max vertices. If there are no vertices, it's empty.
    if V.size == 0:
        # Empty polytope, no center
        return np.zeros((1, 0))

    if np.any(np.isinf(V)):
        # Unbounded
        return np.array([np.nan])

    if V.shape[1] == 1: # Single point
        return V.flatten()
    else: # Interval
        return np.array([(np.min(V) + np.max(V)) / 2])


def _aux_center_chebyshev(P: 'Polytope') -> np.ndarray:
    """Compute Chebyshev center via linear program"""
    # Read out dimension
    n = P.dim()
    
    # Check whether there are only equalities: allows to avoid the linear
    # program from below (faster)
    if P.A.size == 0 and P.Ae.size > 0:
        return _aux_center_only_equalityConstraints(P, n)
    
    # General method: compute Chebyshev center via linear program; to this end,
    # we require the halfspace representation
    P.constraints()
    return _aux_center_LP(P, n)


def _aux_center_avg(P: 'Polytope') -> np.ndarray:
    """Compute average of vertices"""
    # Compute vertices
    V = P.vertices_()
    
    # Compute mean
    return np.mean(V, axis=1, keepdims=True)


def _aux_center_only_equalityConstraints(P: 'Polytope', n: int) -> np.ndarray:
    """Handle case with only equality constraints
    Three outcomes: unbounded, single point, infeasible
    """
    # Minimal halfspace representation: if two constraints are aligned and
    # cannot be fulfilled at the same time, an empty polytope is returned
    try:
        # Get normalized constraints
        # Use P.Ae and P.be directly, as they are guaranteed to be NumPy arrays.
        _, _, Ae, be = P._priv_normalizeConstraints(None, None, P.Ae, P.be, 'A')
        Ae, be, empty = P._priv_compact_alignedEq(Ae, be, 1e-12)
        
        # Check if emptiness has been determined during the computation of the
        # minimal representation
        if empty:
            return np.zeros((n, 0))
        
        # All constraints now are linearly independent, hence the relation of 
        # system dimension and number of constraints determines the solution
        if Ae.shape[0] < n:
            # Underdetermined -> unbounded
            return np.full((n, 1), np.nan)
        elif Ae.shape[0] > n:
            # Overdetermined -> no solution
            return np.zeros((n, 0))
        else:
            # Same number of constraints as system dimension -> single point
            return np.linalg.solve(Ae, be).reshape(-1, 1)
            
    except Exception:
        # Fallback to general LP method
        return _aux_center_LP(P, n)


def _aux_center_LP(P: 'Polytope', n: int) -> np.ndarray:
    """Linear program for the computation of the Chebyshev center"""
    
    # Get constraint matrices using public properties
    A_val = P.A
    b_val = P.b
    Ae_val = P.Ae
    be_val = P.be
    
    # Dimension and number of (in)equalities
    nrEq = Ae_val.shape[0]
    
    # 2-Norm of each row
    if A_val.size > 0:
        A_norm = np.sqrt(np.sum(A_val**2, axis=1, keepdims=True))
        
        # Extend inequality constraints by one column
        A_ext = np.hstack([A_val, A_norm])
    else:
        A_ext = None
        
    # Extend equality constraints by one column
    if nrEq > 0:
        Ae_ext = np.hstack([Ae_val, np.zeros((nrEq, 1))])
    else:
        Ae_ext = None
    
    # Cost function for linear program: minimize 2-norm of constraints
    f = np.zeros(n + 1)
    f[-1] = -1  # Maximize the radius (minimize negative radius)
    
    # Bounds: center can be anything, radius must be non-negative
    bounds = [(None, None)] * n + [(0, None)]
    
    try:
        # Solve LP using scipy.optimize.linprog
        result = linprog(
            c=f,
            A_ub=A_ext,
            b_ub=b_val.flatten() if b_val.size > 0 else None,
            A_eq=Ae_ext,
            b_eq=be_val.flatten() if be_val.size > 0 else None,
            bounds=bounds,
            method='highs'
        )
        
        if result.success:
            # Truncate solution (remove radius component)
            c = result.x[:n].reshape(-1, 1)
            return c
        elif result.status == 2:  # Infeasible
            # Set is empty
            return np.zeros((n, 0))
        elif result.status == 3:  # Unbounded
            # Unbounded
            return np.full((n, 1), np.nan)
        else:
            # Other solver issue
            raise CORAerror('CORA:solverIssue', 'Linear program solver failed')
            
    except Exception as e:
        if 'infeasible' in str(e).lower():
            return np.zeros((n, 0))
        elif 'unbounded' in str(e).lower():
            return np.full((n, 1), np.nan)
        else:
            raise CORAerror('CORA:solverIssue', f'Linear program solver failed: {str(e)}') 