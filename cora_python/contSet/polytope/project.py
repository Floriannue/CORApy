import numpy as np
from typing import List, TYPE_CHECKING
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.polytope.polytope import Polytope

def _aux_fourier_motzkin_elimination(A: np.ndarray, b: np.ndarray, j: int) -> (np.ndarray, np.ndarray):
    """
    Project the polytope A*x <= b onto dimension j using Fourier-Motzkin elimination.
    Algorithm taken from MATLAB CORA implementation.
    """
    # number of constraints
    nr_con = A.shape[0]
    
    # divide j-th column of matrix A into entries = 0, > 0, and < 0
    Z = np.where(A[:, j] == 0)[0]  # zero coefficients
    N = np.where(A[:, j] < 0)[0]   # negative coefficients  
    P = np.where(A[:, j] > 0)[0]   # positive coefficients
    
    # compute cartesian product of sets P and N to get all combinations
    if len(P) > 0 and len(N) > 0:
        # Create all combinations of P and N
        list_combinations = []
        for n_idx in N:
            for p_idx in P:
                list_combinations.append([n_idx, p_idx])
        list_combinations = np.array(list_combinations)
        nr_comb = len(list_combinations)
    else:
        list_combinations = np.array([]).reshape(0, 2)
        nr_comb = 0
    
    # construct projection matrix: number of columns of the projection
    # matrix is the number of constraints of the projected polytope
    m = len(Z)
    U = np.zeros((m + nr_comb, nr_con))
    
    # for all Z, we have the Z(i)-th basis vector
    for i in range(m):
        U[i, Z[i]] = 1
    
    # for all pairs (s,t) in P x N, we have
    #   a_(t,j) * e_(s) - a_(s,j) * e_(t)
    for i in range(nr_comb):
        n_idx = list_combinations[i, 0]  # negative index
        p_idx = list_combinations[i, 1]  # positive index
        U[m + i, n_idx] = A[p_idx, j]   # a_(p,j) coefficient for negative constraint
        U[m + i, p_idx] = -A[n_idx, j]  # -a_(n,j) coefficient for positive constraint
    
    # perform projection
    A_new = U @ A
    b_new = U @ b.reshape(-1, 1)
    
    # remove the eliminated column j
    A_new = np.delete(A_new, j, axis=1)
    
    return A_new, b_new


def project(P: 'Polytope', dims: List[int], method: str = 'fourier') -> 'Polytope':

    from cora_python.contSet.polytope.polytope import Polytope
    n = P.dim()
    
    if sorted(dims) == list(range(1, n + 1)):
        return P.copy()
        
    if any(d > n for d in dims):
        raise CORAerror('CORA:wrongValue', 'second', f'Cannot compute projection on higher dimension than {n}.')

    if P._has_v_rep:
        # Adjust dims for 0-based indexing
        py_dims = [d - 1 for d in dims]
        return Polytope(P.V[py_dims, :])
    
    # Projection for H-representation
    from .private.priv_normalize_constraints import priv_normalize_constraints
    from .private.priv_compact_all import priv_compact_all
    from .private.priv_equality_to_inequality import priv_equality_to_inequality
    
    tol = 1e-12
    if P.isemptyobject():
        return Polytope.empty(len(dims))

    # Normalize and compact constraints
    A, b, Ae, be = priv_normalize_constraints(P.A, P.b, P.Ae, P.be, 'A')
    A, b, Ae, be, empty, _ = priv_compact_all(A, b, Ae, be, n, tol)
    
    if empty:
        return Polytope.empty(len(dims))

    if method == 'fourier':
        A, b = priv_equality_to_inequality(A, b, Ae, be)

        # Dimensions to be projected away
        remove_dims = sorted(list(set(range(n)) - set(d - 1 for d in dims)))
        
        A_proj, b_proj = A, b
        
        # Project away each dimension one by one
        for i in range(len(remove_dims)):
            # Project away current dimension
            A_proj, b_proj = _aux_fourier_motzkin_elimination(A_proj, b_proj, remove_dims[i])
            
            # Update indices to match projected polytope (dimensions shift down after removal)
            for k in range(i + 1, len(remove_dims)):
                if remove_dims[k] > remove_dims[i]:
                    remove_dims[k] -= 1
            
            # Only normalize constraints after each elimination (no compaction)
            if A_proj.size > 0:
                A_proj, b_proj, _, _ = priv_normalize_constraints(A_proj, b_proj, np.array([]), np.array([]), 'A')
        
        # Remove redundant constraints (all-zero rows)
        if A_proj.size > 0:
            zero_rows = []
            for i in range(A_proj.shape[0]):
                if np.allclose(A_proj[i], 0, atol=tol):
                    zero_rows.append(i)
            
            if zero_rows:
                mask = np.ones(A_proj.shape[0], dtype=bool)
                mask[zero_rows] = False
                A_proj = A_proj[mask]
                b_proj = b_proj[mask]
        
        # Sort dimensions of the remaining projected polytope according to dims
        if A_proj.size > 0:
            # The MATLAB code does: [~,ind] = sort(dims); A(:,ind) = A;
            # This reorders the columns to match the sorted dimension order
            dims_sorted = sorted(dims)
            if dims != dims_sorted:
                # Find permutation needed to reorder columns
                perm = [dims.index(d) for d in dims_sorted]
                A_proj = A_proj[:, perm]
        
        return Polytope(A_proj, b_proj)

    elif method == 'fourier_jones':
        try:
            import cdd
        except ImportError:
            raise CORAerror('CORA:thirdPartyError', "pycddlib (cdd module) is not available. Install with 'pip install pycddlib'.")

        # Convert constraints to cdd format
        # CDD expects the constraints in the form: b - A*x >= 0, i.e., [-A | b]
        # We have A*x <= b, so we need to convert: b - A*x >= 0
        
        # Combine inequality and equality constraints
        if Ae is not None and Ae.size > 0:
            # Convert equality constraints Ae*x = be to two inequalities:
            # Ae*x <= be and -Ae*x <= -be
            A_combined = np.vstack([A, Ae, -Ae])
            b_combined = np.hstack([b, be, -be])
        else:
            A_combined = A
            b_combined = b
        
        # Convert to CDD format: [-A | b]
        cdd_matrix = np.hstack([b_combined.reshape(-1, 1), -A_combined])
        
        # Create CDD matrix object
        mat = cdd.Matrix(cdd_matrix, number_type='float')
        mat.rep_type = cdd.RepType.INEQUALITY
        
        # Convert to vertex representation
        poly = cdd.Polyhedron(mat)
        vertices_mat = poly.get_generators()
        
        if vertices_mat is None or len(vertices_mat) == 0:
            # Empty polytope
            return Polytope.empty(len(dims))
        
        # Extract vertices (skip first column which is for rays/vertices type)
        vertices = np.array(vertices_mat)
        
        # Filter out rays (first column == 0) and keep only vertices (first column == 1)
        vertex_mask = vertices[:, 0] == 1
        if not np.any(vertex_mask):
            raise CORAerror('CORA:thirdPartyError', "Polytope is unbounded or degenerate - no vertices found.")
        
        vertices = vertices[vertex_mask, 1:]  # Remove first column
        
        # Project vertices to desired dimensions
        py_dims = [d - 1 for d in dims]  # Convert to 0-based indexing
        projected_vertices = vertices[:, py_dims]
        
        # Convert projected vertices back to half-space representation using CDD
        if projected_vertices.shape[0] == 0:
            return Polytope.empty(len(dims))
        
        # Add the vertex indicator column back
        vertex_matrix = np.hstack([np.ones((projected_vertices.shape[0], 1)), projected_vertices])
        
        # Create new CDD matrix from vertices
        proj_mat = cdd.Matrix(vertex_matrix, number_type='float')
        proj_mat.rep_type = cdd.RepType.GENERATOR
        
        # Convert back to inequality representation
        proj_poly = cdd.Polyhedron(proj_mat)
        ineq_mat = proj_poly.get_inequalities()
        
        if ineq_mat is None or len(ineq_mat) == 0:
            return Polytope.empty(len(dims))
        
        # Convert back from CDD format: [b | -A] to A, b
        ineq_array = np.array(ineq_mat)
        b_proj = ineq_array[:, 0]
        A_proj = -ineq_array[:, 1:]
        
        return Polytope(A_proj, b_proj)
    
    else:
        raise CORAerror('CORA:wrongValue', 'third', f"Unknown projection method '{method}'.") 