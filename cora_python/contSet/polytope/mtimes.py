from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Union
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check
from cora_python.g.functions.matlab.validate.preprocessing.find_class_arg import find_class_arg
from cora_python.contSet.polytope import Polytope
from cora_python.g.functions.matlab.validate.check import equal_dim_check
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol

if TYPE_CHECKING:
    from .polytope import Polytope
    from cora_python.matrixSet.intervalMatrix.intervalMatrix import IntervalMatrix


def mtimes(factor1: object, factor2: object) -> 'Polytope':
    """
    Matrix multiplication with polytope.
    
    Args:
        factor1: Matrix or scalar or polytope
        factor2: Polytope or scalar
        
    Returns:
        Polytope: Result of matrix multiplication
    """
    from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check
    from cora_python.g.functions.matlab.validate.preprocessing.find_class_arg import find_class_arg
    from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
    from .polytope import Polytope
    
    # order arguments correctly
    P_copy, matrix = find_class_arg(factor1, factor2, 'Polytope')
    
    # For scalar operations, no dimension check needed
    if np.isscalar(matrix):
        pass  # Skip dimension check for scalars
    # For matrix operations, check dimensions
    elif isinstance(matrix, np.ndarray):
        # Check if matrix-polytope multiplication is valid
        if isinstance(factor1, np.ndarray) and hasattr(factor2, 'dim'):
            # matrix @ polytope: matrix.shape[1] should equal polytope.dim()
            if matrix.shape[1] != factor2.dim():
                from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
                raise CORAerror('CORA:dimensionMismatch', factor1, factor2)
        elif hasattr(factor1, 'dim') and isinstance(factor2, np.ndarray):
            # polytope @ matrix: only scalars allowed for polytope * matrix
            if matrix.size > 1:
                raise ValueError("polytope * matrix case not supported (only scalars)")
    else:
        # For other types, use the general dimension check
        equal_dim_check(factor1, factor2)
    
    # copy polytope
    P_out = P_copy.copy()
    
    # read out dimension
    n = P_out.dim()
    
    # fullspace
    if P_out.representsa_('fullspace', 0):
        return Polytope.Inf(n)
    
    # Handle numeric matrix/scalar
    if isinstance(matrix, (np.ndarray, int, float, np.integer, np.floating)):
        
        # Convert scalar to array for consistency
        if np.isscalar(matrix):
            matrix = np.array([[matrix]])
        elif isinstance(matrix, np.ndarray) and matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        elif isinstance(matrix, np.ndarray) and matrix.ndim == 0:
            matrix = np.array([[matrix.item()]])
            
        # map with all-zero matrix
        if np.allclose(matrix, 0, atol=1e-12):
            return Polytope.origin(n)
        
        # Check for polytope * matrix case (only allow scalar)
        if isinstance(factor1, Polytope):
            if np.isscalar(factor2):
                return _aux_mtimes_scaling(P_out, float(factor2))
            else:
                raise ValueError("polytope * matrix case not supported (only scalars)")
        
        # special method for scaling only (and 1D)
        if matrix.size == 1:
            return _aux_mtimes_scaling(P_out, matrix.item())
        
        # simple method if matrix is invertible
        if (matrix.shape[0] == matrix.shape[1] and 
            np.linalg.matrix_rank(matrix, tol=1e-12) == matrix.shape[0]):
            return _aux_mtimes_square_inv(P_out, matrix)
        
        # For H-representation polytopes with non-invertible square matrices, raise error
        if (hasattr(P_out, '_has_h_rep') and P_out._has_h_rep and 
            not (hasattr(P_out, '_has_v_rep') and P_out._has_v_rep) and
            matrix.shape[0] == matrix.shape[1] and 
            np.linalg.matrix_rank(matrix, tol=1e-12) < matrix.shape[0]):
            raise NotImplementedError("Multiplication with non-invertible matrix not supported for H-representation polytopes")
        
        # quicker computation using V-representation
        if hasattr(P_out, '_has_v_rep') and P_out._has_v_rep and P_out._V is not None:
            # For d × n_vertices format: M @ V where M is (m × d) and V is (d × n_vertices)
            # Result should be (m × n_vertices)
            V_new = matrix @ P_out._V
            return Polytope(V_new)
        else:
            # method for general mappings
            m, n_mat = matrix.shape
            if m < n_mat:
                return _aux_mtimes_projection(P_out, matrix)
            else:
                return _aux_mtimes_lifting(P_out, matrix)
    
    # Handle interval matrix (placeholder for now)
    elif hasattr(matrix, 'inf') and hasattr(matrix, 'sup'):
        return _aux_mtimes_interval(P_out, matrix)
    
    else:
        raise TypeError(f"Unsupported matrix type: {type(matrix)}")





def _aux_mtimes_scaling(P: 'Polytope', fac: float) -> 'Polytope':
    """
    Simple method for scaling:
       M S = { M s | s in S }
    -> fac * I * S = { fac * I * x | A x <= b, Ae x == b}    set y = fac*I*x
                   = { y | A (1/fac*I*y) <= b, Ae (1/fac*I*y) == b }
                   = { y | A/fac y <= b, Ae/fac y == b }
                   = { y | A y <= b*fac, Ae y == b*fac }     if fac > 0
                OR = { y | -A y <= -b*fac, Ae y == b*fac }   if fac < 0
    (note: case with fac = 0 yields a polytope that is just the origin)
    """
    if fac == 0:
        # resulting polytope is only the origin
        return Polytope.origin(P.dim())
    
    P_new = P.copy()
    
    if hasattr(P_new, '_has_h_rep') and P_new._has_h_rep:
        if fac > 0:
            if P_new._b is not None:
                P_new._b = P_new._b * fac
        else:
            # fac < 0: flip inequality directions
            if P_new._A is not None:
                P_new._A = -P_new._A
            if P_new._b is not None:
                P_new._b = -P_new._b * fac
        
        # Equality constraints
        if P_new._be is not None:
            P_new._be = P_new._be * fac
    
    # map vertices if given
    if hasattr(P_new, '_has_v_rep') and P_new._has_v_rep and P_new._V is not None:
        # For d × n_vertices format: V_new = fac * V
        P_new._V = P_new._V * fac
        
    return P_new


def _aux_mtimes_square_inv(P: 'Polytope', M: np.ndarray) -> 'Polytope':
    """Matrix M is invertible."""
    P_new = P.copy()
    
    # apply well-known formula (constraints times inverse of matrix)
    if hasattr(P_new, '_has_h_rep') and P_new._has_h_rep:
        M_inv = np.linalg.inv(M)
        if P_new._A is not None:
            P_new._A = P_new._A @ M_inv
        if P_new._Ae is not None:
            P_new._Ae = P_new._Ae @ M_inv
    
    # map vertices if given
    if hasattr(P_new, '_has_v_rep') and P_new._has_v_rep and P_new._V is not None:
        # For d × n_vertices format: V_new = M @ V where M is (d × d) and V is (d × n_vertices)
        P_new._V = M @ P_new._V
    
    return P_new


def _aux_mtimes_projection(P: 'Polytope', M: np.ndarray) -> 'Polytope':
    """General matrix multiplication with projection, see [1, (24)]."""
    n = P.dim()
    
    # compute singular value decomposition of the matrix
    U, S, V = np.linalg.svd(M, full_matrices=True)
    # number of singular values
    r = np.sum(S > 1e-10)
    # get inverted diagonal matrix with non-zero singular values
    D_inv = np.diag(1.0 / S[:r])
    
    P_new = P.copy()
    
    # init polytope before projection
    transform_matrix = V.T @ np.block([
        [D_inv, np.zeros((r, n - r))],
        [np.zeros((n - r, r)), np.eye(n - r)]
    ])
    
    if hasattr(P_new, '_has_h_rep') and P_new._has_h_rep:
        if P_new._A is not None:
            P_new._A = P_new._A @ transform_matrix
        if P_new._Ae is not None:
            P_new._Ae = P_new._Ae @ transform_matrix
    
    # project onto first r dimensions
    P_proj = _project_polytope(P_new, list(range(r)))
    
    # multiply with orthogonal matrix
    return _aux_mtimes_square_inv(P_proj, U[:, :r])


def _aux_mtimes_lifting(P: 'Polytope', M: np.ndarray) -> 'Polytope':
    """Handle multiplication that results in dimension increase (lifting)."""
    n = P.dim()
    m = M.shape[0]
    
    # number of inequality constraints and equality constraints
    nr_ineq = P._A.shape[0] if P._A is not None else 0
    nr_eq = P._Ae.shape[0] if P._Ae is not None else 0
    
    # compute singular value decomposition of the matrix
    U, S, V = np.linalg.svd(M, full_matrices=True)
    # number of singular values
    r = np.sum(S > 1e-10)
    # get inverted diagonal matrix with non-zero singular values
    D_inv = np.diag(1.0 / S[:r])
    
    P_new = P.copy()
    
    if hasattr(P_new, '_has_h_rep') and P_new._has_h_rep:
        # Transform inequality constraints
        if P_new._A is not None:
            A_transform = np.block([P_new._A @ V.T[:, :r] @ D_inv, np.zeros((nr_ineq, m - r))])
            P_new._A = A_transform @ U.T
        
        # Transform equality constraints  
        if P_new._Ae is not None:
            Ae_transform = np.block([P_new._Ae @ V.T[:, :r] @ D_inv, np.zeros((nr_eq, m - r))])
            # Add new equality constraints for lifted dimensions
            Ae_lift = np.block([np.zeros((m - r, r)), np.eye(m - r)])
            Ae_combined = np.vstack([Ae_transform, Ae_lift])
            P_new._Ae = Ae_combined @ U.T
            
            # Extend be vector
            be_new = np.vstack([P_new._be, np.zeros((m - r, 1))])
            P_new._be = be_new
        else:
            # Create new equality constraints for the lifted dimensions
            Ae_lift = np.block([np.zeros((m - r, r)), np.eye(m - r)])
            P_new._Ae = Ae_lift @ U.T
            P_new._be = np.zeros((m - r, 1))
    
    return P_new


def _project_polytope(P: 'Polytope', dims: list) -> 'Polytope':
    """Project polytope onto specified dimensions."""
    P_new = P.copy()
    
    if hasattr(P_new, '_has_h_rep') and P_new._has_h_rep:
        if P_new._A is not None:
            P_new._A = P_new._A[:, dims]
        if P_new._Ae is not None:
            P_new._Ae = P_new._Ae[:, dims]
    
    if hasattr(P_new, '_has_v_rep') and P_new._has_v_rep and P_new._V is not None:
        P_new._V = P_new._V[dims, :]
    
    return P_new


def _aux_mtimes_interval(P: 'Polytope', matrix) -> 'Polytope':
    """Handle multiplication by interval matrix (following MATLAB approach)."""
    from cora_python.contSet.interval.interval import Interval
    from .polytope import Polytope
    
    # Only supported for square matrices
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Multiplication of interval matrix with polytope only supported for square interval matrices.")
    
    # Get minimum and maximum bounds
    M_min = matrix.inf if hasattr(matrix, 'inf') else matrix.infimum
    M_max = matrix.sup if hasattr(matrix, 'sup') else matrix.supremum
    
    # Get center of interval matrix
    T = 0.5 * (M_max + M_min)
    
    # Get symmetric interval matrix
    S_val = 0.5 * (M_max - M_min)
    S = Interval(-S_val, S_val)
    
    # Compute interval of polytope
    I = P.interval()
    
    # Polytope of interval computations
    I_add = S.__matmul__(I)  # S * I using interval matrix multiplication
    P_add = Polytope(I_add)  # Convert interval back to polytope
    
    # Compute new polytope
    P_T = _aux_mtimes_square_inv(P, T)
    return P_T.plus(P_add) 