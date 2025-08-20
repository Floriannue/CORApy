from __future__ import annotations
import numpy as np
from cora_python.g.functions.matlab.validate.preprocessing.find_class_arg import find_class_arg
from cora_python.contSet.polytope import Polytope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check



def mtimes(factor1: object, factor2: object) -> Polytope:
    """
    Matrix multiplication with polytope.
    
    Args:
        factor1: Matrix or scalar or polytope
        factor2: Polytope or scalar
        
    Returns:
        Polytope: Result of matrix multiplication
    """
    
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
        if (P_out.isHRep and 
            not P_out.isVRep and
            matrix.shape[0] == matrix.shape[1] and 
            np.linalg.matrix_rank(matrix, tol=1e-12) < matrix.shape[0]):
            raise NotImplementedError("Multiplication with non-invertible matrix not supported for H-representation polytopes")
        
        # quicker computation using V-representation
        # P_out.V is guaranteed to be a numpy array
        if P_out.isVRep:
            # For d × n_vertices format: M @ V where M is (m × d) and V is (d × n_vertices)
            # Result should be (m × n_vertices)
            V_new = matrix @ P_out.V
            new_P = Polytope(V_new)
            # Reset lazy flags for new_P since its representation was derived
            new_P._reset_lazy_flags() # Use the consolidated reset method
            return new_P
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
                   = { y | A y <= b*fac, Ae y == b*fac }     if fac > 0
                OR = { y | -A y <= -b*fac, Ae y == b*fac }   if fac < 0
    (note: case with fac = 0 yields a polytope that is just the origin)
    """
    # resulting polytope is only the origin
    if fac == 0:
        return Polytope.origin(P.dim())
    
    P_new = Polytope(P) # Create copy to modify
    
    if P_new.isHRep:
        if P_new.A.size > 0: # Only scale if A is not empty
            if fac > 0:
                P_new._b = P_new.b * fac # Write to private for efficiency within class method
            else:
                # fac < 0: flip inequality directions
                P_new._A = -P_new.A # Write to private
                P_new._b = -P_new.b * fac # Write to private
            
        if P_new.Ae.size > 0: # Only scale if Ae is not empty
            # Equality constraints
            P_new._be = P_new.be * fac # Write to private

        P_new._reset_lazy_flags() # Reset lazy flags after modifying H-representation
    
    # map vertices if given
    # P_new.V is guaranteed to be a numpy array
    if P_new.isVRep:
        if P_new.V.size > 0: # Only scale if V is not empty
            # For d × n_vertices format: V_new = fac * V
            P_new._V = P_new.V * fac # Write to private

        P_new._reset_lazy_flags() # Reset lazy flags after modifying V-representation
        
    return P_new


def _aux_mtimes_square_inv(P: 'Polytope', M: np.ndarray) -> 'Polytope':
    """Matrix M is invertible."""
    P_new = Polytope(P)
    
    # apply well-known formula (constraints times inverse of matrix)
    if P_new.isHRep:
        M_inv = np.linalg.inv(M)
        if P_new.A.size > 0:
            P_new._A = P_new.A @ M_inv
        else:
            # Ensure empty A maintains correct column dimension after multiplication
            P_new._A = np.zeros((0, M_inv.shape[0]))

        if P_new.Ae.size > 0:
            P_new._Ae = P_new.Ae @ M_inv
        else:
            # Ensure empty Ae maintains correct column dimension after multiplication
            P_new._Ae = np.zeros((0, M_inv.shape[0]))

        P_new._reset_lazy_flags() # Reset lazy flags after modifying H-representation
    
    # map vertices if given
    # P_new.V is guaranteed to be a numpy array
    if P_new.isVRep:
        if P_new.V.size > 0:
            # For d × n_vertices format: V_new = M @ V where M is (d × d) and V is (d × n_vertices)
            P_new._V = M @ P_new.V
        else:
            # Ensure empty V maintains correct row dimension after multiplication
            P_new._V = np.zeros((M.shape[0], 0))

        P_new._reset_lazy_flags() # Reset lazy flags after modifying V-representation
    
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
    
    P_new = Polytope(P)
    
    # init polytope before projection
    transform_matrix = V.T @ np.block([
        [D_inv, np.zeros((r, n - r))],
        [np.zeros((n - r, r)), np.eye(n - r)]
    ])
    
    if P_new.isHRep:
        P_new._A = P_new.A @ transform_matrix
        P_new._Ae = P_new.Ae @ transform_matrix

        P_new._reset_lazy_flags() # Reset lazy flags after modifying H-representation
    
    # project onto first r dimensions
    # If P_new is empty at this point, the projection should also be an empty polytope
    if P_new.isemptyobject(): # Check if the source polytope is already empty
        return Polytope.empty(r) # Return empty polytope of the projected dimension
    P_proj = _project_polytope(P_new, list(range(r)))
    
    # multiply with orthogonal matrix
    return _aux_mtimes_square_inv(P_proj, U[:, :r])


def _aux_mtimes_lifting(P: 'Polytope', M: np.ndarray) -> 'Polytope':
    """Handle multiplication that results in dimension increase (lifting)."""
    n = P.dim()
    m = M.shape[0]
    
    # number of inequality constraints and equality constraints
    nr_ineq = P.A.shape[0]
    nr_eq = P.Ae.shape[0]
    
    # compute singular value decomposition of the matrix
    U, S, V = np.linalg.svd(M, full_matrices=True)
    # number of singular values
    r = np.sum(S > 1e-10)
    # get inverted diagonal matrix with non-zero singular values
    D_inv = np.diag(1.0 / S[:r])
    
    P_new = Polytope(P)
    
    if P_new.isHRep:
        # Transform inequality constraints
        nr_ineq_actual = P_new.A.shape[0]
        if nr_ineq_actual > 0:
            A_transform_block = P_new.A @ V.T[:, :r] @ D_inv
            A_transform = np.block([A_transform_block, np.zeros((nr_ineq_actual, m - r))])
        else:
            A_transform = np.zeros((0, m)) # Ensure correct dimensions for empty A
        P_new._A = A_transform @ U.T
        
        # Transform equality constraints  
        nr_eq_actual = P_new.Ae.shape[0]
        if nr_eq_actual > 0:
            Ae_transform_block = P_new.Ae @ V.T[:, :r] @ D_inv
            Ae_transform = np.block([Ae_transform_block, np.zeros((nr_eq_actual, m - r))])
        else:
            Ae_transform = np.zeros((0, m)) # Ensure correct dimensions for empty Ae
        
        # Add new equality constraints for lifted dimensions
        Ae_lift = np.block([np.zeros((m - r, r)), np.eye(m - r)])
        
        # Combine existing and new equality constraints
        if Ae_transform.size > 0 and Ae_lift.size > 0: # Both non-empty
            Ae_combined = np.vstack([Ae_transform, Ae_lift])
        elif Ae_transform.size > 0: # Only existing transformed constraints
            Ae_combined = Ae_transform
        elif Ae_lift.size > 0: # Only new lifted constraints
            Ae_combined = Ae_lift
        else: # Both empty, result is empty with correct dimensions
            Ae_combined = np.zeros((0, m))

        P_new._Ae = Ae_combined @ U.T
        
        # Extend be vector
        if P_new.be.size > 0 or m - r > 0:
            be_new = np.vstack([P_new.be, np.zeros((m - r, 1))])
        else:
            be_new = np.zeros((0,1)) # Ensure empty be has correct dimension
        P_new._be = be_new

        P_new._reset_lazy_flags() # Reset lazy flags after modifying H-representation
    
    return P_new


def _project_polytope(P: 'Polytope', dims: list) -> 'Polytope':
    """Project polytope onto specified dimensions."""
    P_new = Polytope(P)
    
    if P_new.isHRep:
        P_new._A = P_new.A[:, dims]
        P_new._Ae = P_new.Ae[:, dims]
    
    if P_new.isVRep:
        P_new._V = P_new.V[dims, :]

    P_new._reset_lazy_flags() # Reset lazy flags after modifying representation

    return P_new


def _aux_mtimes_interval(P: 'Polytope', matrix) -> 'Polytope':
    """Handle multiplication by interval matrix (following MATLAB approach)."""
    
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