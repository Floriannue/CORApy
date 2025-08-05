import numpy as np
from itertools import combinations
from scipy.optimize import linprog
import warnings
from typing import TYPE_CHECKING, Optional, Tuple

from cora_python.g.functions.matlab.validate.check import withinTol
from cora_python.contSet.polytope.private.priv_equalityToInequality import priv_equalityToInequality
from cora_python.contSet.polytope.private.priv_normalizeConstraints import priv_normalizeConstraints
from cora_python.contSet.polytope.private.priv_compact_all import priv_compact_all
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

# Placeholder for priv_vertices_1D
# Placeholder for priv_compact_all
# Placeholder for priv_normalizeConstraints

if TYPE_CHECKING:
    from cora_python.contSet.polytope.polytope import Polytope

def vertices_(P: 'Polytope', method: str = 'lcon2vert') -> np.ndarray:
    """
    Computes the vertices of a polytope.
    This is a Python translation of the 'lcon2vert' and 'comb' methods
    from the MATLAB CORA library.
    """
    tol = 1e-9 # Default tolerance, consistent with MATLAB in some places

    # If polytope has V-representation, return it (after ensuring minimality if needed)
    if P.isVRep:
        # MATLAB's vertices.m calls compact_(P, 'V', tol) here.
        # For now, we'll return P.V directly, assuming compact_ is handled by higher level call if needed.
        return P.V

    n = P.dim()

    # Check if polytope is known to be empty
    if P.isemptyobject(): # Use isemptyobject method like MATLAB
        # MATLAB lines 96-102: polytope is empty
        V = np.zeros((n, 0))
        P._V = V
        P._isVRep = True
        # Set cache values like MATLAB
        P._minVRep_val = True      # P.minVRep.val = true;
        P._emptySet_val = True     # P.emptySet.val = true;
        P._bounded_val = True      # P.bounded.val = true;
        P._fullDim_val = False     # P.fullDim.val = false;
        return V

    # 1D case quick
    if n == 1:
        # Convert to H-rep if not already
        if not P.isHRep:
            P.constraints()
        V = _priv_vertices_1D(P.A, P.b, P.Ae, P.be, tol)
        # Set properties after computation
        P._V = V
        P._isVRep = True
        P._isHRep = False # V-rep is now primary
        # Reset cache values as representation has changed
        P._reset_lazy_flags()
        P._minVRep_val = True # 1D vertices are always minimal
        return P.V

    # Compute Chebyshev center to detect unbounded/empty cases for nD
    # Note: Python's center returns center_vector (numpy array)
    c_vec = P.center() # center() now returns a single numpy array

    # If center calculation indicates an empty set or unbounded set
    # (e.g., center returns empty array or NaN values)
    if c_vec.size == 0 or np.any(np.isnan(c_vec)):
        # If it's unbounded, vertices cannot be computed
        if P.dim() > 0 and not P.isBounded(): # Using P.bounded property
            raise CORAerror('CORA:notSupported',
                            'Vertex computation requires a bounded polytope.')
        else:
            # Empty set or other degenerate case where center is not meaningful
            P._V = np.zeros((n, 0))
            P._isVRep = True
            P._emptySet_val = True
            P._minVRep_val = True
            return P.V

    # If the center calculation returns a valid center and radius (from chebyshev method)
    # and if the radius is very small, it means it's a point or empty
    if method == 'chebyshev': # Only if chebyshev was used
        # We need to re-call center with the certToggle to get the radius, or infer
        # this from the chebyshev computation.
        # For now, we will assume if the center is valid, it's not empty/unbounded by center.
        pass

    # Ensure H-representation is available for 'lcon2vert' and 'comb' methods
    if not P.isHRep:
        P.constraints()

    # Choose method
    if method == 'lcon2vert':
        V = _aux_vertices_lcon2vert(P, n, c_vec)
    elif method == 'comb':
        V = _aux_vertices_comb(P)
    # elif method == 'cdd':
    #     # Placeholder for cdd method
    #     # V = aux_vertices_cdd(P, c_vec)
    #     raise NotImplementedError("CDD method for vertex enumeration not implemented.")
    else:
        raise ValueError(f'Invalid method \'{method}\'. Allowed: \'lcon2vert\', \'comb\'.')

    # Set properties after computation
    P._V = V
    P._isVRep = True
    P._isHRep = False # V-rep is now primary; H-rep might not be minimal

    # Update cache values based on computed V
    P._emptySet_val = V.size == 0  # Emptiness is determined by existence of V  
    P._bounded_val = not np.any(np.isinf(V))  # If any vertex is inf, it's unbounded

    # Determine full-dimensionality: if rank of V is less than dimension, it's not full-dimensional
    if V.size > 0 and np.linalg.matrix_rank(V - np.mean(V, axis=1, keepdims=True)) < n:
        P._fullDim_val = False
    # Don't set _fullDim_val = True here - let isFullDim() compute it properly when needed

    # Don't set _minHRep_val - let it be computed when H-rep is computed
    P._minVRep_val = True  # V-rep is now minimal by construction (after duplicate removal)

    return P.V


def _priv_vertices_1D(A: np.ndarray, b: np.ndarray, Ae: np.ndarray, be: np.ndarray, tol: float) -> np.ndarray:
    """Auxiliary function for 1D case, extracting vertices from H-rep.
    Matches MATLAB's priv_vertices_1D.
    """
    # Collect all boundary points from inequalities and equalities
    boundaries = []

    # From inequalities: A*x <= b
    # For A[i,0] > 0, x <= b[i,0]/A[i,0] (upper bound)
    # For A[i,0] < 0, x >= b[i,0]/A[i,0] (lower bound, divide by negative)
    upper_bounds = []
    lower_bounds = []

    if A.size > 0:
        for i in range(A.shape[0]):
            if A[i, 0] > tol: # Check for positive coefficient
                upper_bounds.append(b[i, 0] / A[i, 0])
            elif A[i, 0] < -tol: # Check for negative coefficient
                lower_bounds.append(b[i, 0] / A[i, 0])
            # If A[i,0] is zero, it's a trivial constraint (0 <= b), ignored unless b is negative (empty set)
            elif withinTol(A[i,0], 0, tol) and b[i,0] < -tol: # 0*x <= negative -> empty
                return np.zeros((1, 0)) # Return empty for infeasible 0*x <= b

    # From equalities: Ae*x == be
    if Ae.size > 0:
        for i in range(Ae.shape[0]):
            if not withinTol(Ae[i, 0], 0, tol): # If coefficient is not zero
                val = be[i, 0] / Ae[i, 0]
                # For equality, this point is both an upper and lower bound
                upper_bounds.append(val)
                lower_bounds.append(val)
            elif withinTol(Ae[i,0], 0, tol) and not withinTol(be[i,0], 0, tol): # 0*x == non-zero -> empty
                 return np.zeros((1, 0)) # Return empty for infeasible 0*x == b

    # Determine the actual min and max, considering open/unbounded intervals
    min_val = -np.inf if not lower_bounds else np.max(lower_bounds) # Max of lower bounds is the true lower bound
    max_val = np.inf if not upper_bounds else np.min(upper_bounds) # Min of upper bounds is the true upper bound

    # Check for empty set (min > max)
    if min_val > max_val + tol: # Add tolerance for numerical stability
        return np.zeros((1, 0)) # Return empty array if the set is empty
    
    # If the set is a single point (min and max are very close)
    if withinTol(min_val, max_val, tol):
        return np.array([[min_val]]) # Return as single vertex

    # If the set is an interval, return its bounds as vertices
    if not np.isinf(min_val) and not np.isinf(max_val):
        return np.array([[min_val, max_val]])
    elif not np.isinf(min_val): # Only lower bound
        return np.array([[min_val]])
    elif not np.isinf(max_val): # Only upper bound
        return np.array([[max_val]])
    else: # Full space (unbounded in both directions)
        return np.array([[]]) # Return empty 1D array as a representation of R^1


def _aux_vertices_lcon2vert(P: 'Polytope', n: int, c: np.ndarray) -> np.ndarray:
    """
    Placeholder for lcon2vert method. This is a complex method in MATLAB
    that uses a different algorithm than simple combinations.
    For now, it will raise NotImplementedError.
    """
    # MATLAB's lcon2vert function internally checks for degeneracy and calls
    # a recursive version or handles single points. It's a robust vertex enumeration.
    # Python equivalent would be libraries like pycddlib or qhull bindings.
    # For simplicity, we will raise an error, or fallback to 'comb' method.

    # Fallback to 'comb' for now if 'lcon2vert' is explicitly requested
    warnings.warn("lcon2vert method is not fully implemented. Falling back to 'comb' method.")
    return _aux_vertices_comb(P)

    # A more complete implementation might use scipy.spatial.ConvexHull in reverse
    # (from vertices to halfspaces, which is already done in constraints.py)
    # or a dedicated vertex enumeration library like pycddlib.
    # raise NotImplementedError("lcon2vert method for vertex enumeration not implemented.")


def _aux_vertices_comb(P: 'Polytope') -> np.ndarray:
    """
    Simple vertex enumeration algorithm: this function returns a set of
    vertices that contains the true vertices; however, the minimal vertices
    are not the convex hull of the computed vertices.
    Matches MATLAB's aux_vertices_comb.
    """
    tol = 1e-12
    n = P.dim() # Define n here

    # Check if polytope is unbounded (already handled by main vertices_ function)
    # if not P.bounded: # Leverage P.bounded property
    #     raise CORAerror('CORA:notSupported',
    #                     'Vertex computation requires a bounded polytope.')

    # Rewrite as inequality constraints and normalize
    # These functions need to be imported or handled appropriately.
    # For now, we will use the P.A, P.b, P.Ae, P.be directly assuming they are up-to-date
    # after P.constraints() call in main vertices_.
    # Need to call priv_equalityToInequality and priv_normalizeConstraints and priv_compact_all

    A_orig = P.A
    b_orig = P.b
    Ae_orig = P.Ae
    be_orig = P.be

    A, b = priv_equalityToInequality(A_orig, b_orig, Ae_orig, be_orig)

    # Normalize rows
    A, b, _, _ = priv_normalizeConstraints(A, b, np.array([]).reshape(0,n), np.array([]).reshape(0,1), 'A')

    # Minimal H-representation (via compact_all)
    # The MATLAB version passes dim(P) as an argument. Use P.dim().
    A, b, _, _, empty, minHRep = priv_compact_all(A, b, np.array([]).reshape(0,0), np.array([]).reshape(0,1), P.dim(), tol)
    
    # Set cache value if minimal representation was obtained
    if minHRep:
        P._minHRep_val = True

    if empty:
        return np.zeros((P.dim(), 0))

    # Number of constraints and dimension
    nrCon, n = A.shape

    # Combinator does not work if n > nrCon, i.e., degenerate cases
    # If it's a degenerate case (n > nrCon), the 'comb' method cannot compute vertices.
    # We raise an error here because 'comb' is explicitly chosen or is the fallback.
    if n > nrCon:
        raise ValueError(f"Method 'comb' does not support cases where ambient dimension ({n}) is greater than the number of constraints ({nrCon}).")

    # All possible combinations of n constraints
    # Use combinator function to match MATLAB behavior (1-indexed)
    from cora_python.g.functions.matlab.validate.check.auxiliary import combinator
    comb = combinator(nrCon, n, 'c')
    nrComb = comb.shape[0]

    # Throw error if computational effort too high
    if nrComb > 10000:
        raise ValueError('Too many combinations.')

    # Init vertices
    V = np.zeros((n, nrComb))
    # Some combinations don't produce a vertex or the vertex is outside the
    # polytope... use logical indexing at the end to avoid resizing the matrix
    # that contains all vertices
    idxKeep = np.ones(nrComb, dtype=bool)

    # Toggle warning, since some intersection points will be -/+Inf
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning) # Ignore warnings like divide by zero
        warnings.filterwarnings('ignore', message='A_eq does not appear to be of full row rank.')
        
        # Loop over all combinations
        for i in range(nrComb):
            indices = comb[i, :] - 1  # Convert 1-indexed to 0-indexed
            A_sub = A[list(indices), :]
            b_sub = b[list(indices)]

            # If full-rank, then we have exactly one intersection point
            # MATLAB uses rank(A_,1e-8) < n for rank check
            if np.linalg.matrix_rank(A_sub, tol=1e-8) < n:
                idxKeep[i] = False
                continue

            # Compute intersection point of n halfspaces taken from A
            try:
                v = np.linalg.solve(A_sub, b_sub)
            except np.linalg.LinAlgError:
                # If singular, use pseudo-inverse (as MATLAB's \ operator does this sometimes)
                try:
                    v = np.linalg.pinv(A_sub) @ b_sub
                except np.linalg.LinAlgError:
                    idxKeep[i] = False # This combination of constraints is truly problematic
                    continue
            
            # Ensure v is column vector for multiplication
            v = v.reshape(-1, 1)
            V[:, i] = v.flatten()

            # Check if vertex is contained in polytope
            val = A @ v
            if not np.all( (val < b + 1e-8) | withinTol(val, b, 1e-8) ): # Check all constraints with MATLAB tolerance
                idxKeep[i] = False
                continue
            
            # Check if vertex is a duplicate (only if not first vertex)
            if i > 0:
                # vecnorm(V(:,1:i-1) - V(:,i)) equivalent
                # Compare current vertex to previously kept vertices
                existing_vertices = V[:, :i][:, idxKeep[:i]]
                if existing_vertices.shape[1] > 0: # Ensure there are existing vertices to compare against
                    distances = np.linalg.norm(existing_vertices - v, axis=0)
                    if np.any(withinTol(distances, 0, 1e-14)):
                        idxKeep[i] = False

    # Remove vertices at indices where there was no computation, or the
    # computed vertex is outside of the polytope
    V = V[:, idxKeep]

    # Remove vertices with Inf/NaN values
    V = V[:, np.all(np.isfinite(V), axis=0)]

    # Set cache values based on result (like MATLAB)
    P._V = V
    P._isVRep = True
    
    # Check if it's a single point like MATLAB (lines 191-197)
    if V.shape[1] == 1:
        P._minVRep_val = True      # P.minVRep.val = true;
        P._emptySet_val = False    # P.emptySet.val = false;
        P._fullDim_val = False     # P.fullDim.val = false; (no zero-dimensional sets)
        P._bounded_val = True      # P.bounded.val = true;
    elif V.shape[1] > 1:
        # Multiple vertices - set minimal V-rep based on MATLAB logic
        P._minVRep_val = V.shape[1] <= 1 or n == 1  # MATLAB: size(V,2) <= 1 || n == 1

    return V 