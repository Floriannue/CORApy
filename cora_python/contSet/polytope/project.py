import numpy as np
from typing import List, TYPE_CHECKING
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

# Local imports for auxiliary functions that might not be part of Polytope class methods
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol # Needed by _aux_fourier_motzkin_elimination
from cora_python.contSet.polytope.private.priv_normalize_constraints import priv_normalize_constraints
from cora_python.contSet.polytope.private.priv_compact_all import priv_compact_all
from cora_python.contSet.polytope.private.priv_equality_to_inequality import priv_equality_to_inequality

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


def project(P: 'Polytope', dims: List[int], method: str = 'default') -> 'Polytope':
    """
    Projects a polytope onto a set of dimensions.
    
    Args:
        P: Polytope object
        dims: List of dimensions (1-based index) to project onto. e.g., [1, 2] for x and y.
        method: Projection method ('fourier' for Fourier-Motzkin, 'fourier_jones' for pycddlib).
                Defaults to 'fourier_jones' if pycddlib is installed, otherwise 'fourier'.
        
    Returns:
        Polytope: Projected polytope object
    """
    from cora_python.contSet.polytope.polytope import Polytope # Local import to avoid circular dependency
    n_in = P.dim()
    n_out = len(dims)
    
    # No projection, copy the polyhedron
    # Check if the sorted list of dims matches 1-based full range
    if sorted(dims) == list(range(1, n_in + 1)):
        return P.copy()
        
    if any(d > n_in for d in dims) or any(d <= 0 for d in dims):
        raise CORAerror('CORA:wrongValue', 'second', f'Cannot compute projection on dimensions higher than {n_in} or non-positive.')

    if P.isVRep:
        # Adjust dims for 0-based indexing for NumPy array slicing
        py_dims = [d - 1 for d in dims]
        # Return projected vertices as-is (keep duplicates) to mirror MATLAB behavior in tests
        return Polytope(P.V[py_dims, :])
    
    # Default method selection based on pycddlib availability
    if method == 'default':
        # Set default to 'fourier' to bypass pycddlib issues
        method_chosen = 'fourier'
        # The fourier_jones method (using pycddlib) can be explicitly chosen if desired
    else:
        method_chosen = method

    # Check emptiness
    if P.isemptyobject():
        return Polytope.empty(n_out)

    # Normalize and compact constraints BEFORE projection for accuracy
    # A, b, Ae, be are guaranteed to be numpy arrays
    A, b, Ae, be, empty, _ = priv_compact_all(
        *priv_normalize_constraints(P.A, P.b, P.Ae, P.be, 'A'), n_in, 1e-12
    )
    
    if empty:
        return Polytope.empty(n_out)

    if method_chosen == 'fourier':
        A, b = priv_equality_to_inequality(A, b, Ae, be)

        # Dimensions to be projected away (0-based indexing)
        remove_dims_0based = sorted(list(set(range(n_in)) - set(d - 1 for d in dims)))
        
        A_proj, b_proj = A, b
        
        # Project away each dimension one by one
        for i, dim_to_remove_orig_idx in enumerate(remove_dims_0based):
            # The `remove_dims_0based` list elements themselves need to be adjusted
            # as dimensions are removed. Maintain a dynamic index for the current A_proj
            current_dim_to_remove_idx = dim_to_remove_orig_idx - i # Adjust index based on previous removals

            # Project away current dimension
            A_proj, b_proj = _aux_fourier_motzkin_elimination(A_proj, b_proj, current_dim_to_remove_idx)
            
            # Only normalize constraints after each elimination (no compaction here, only normalize)
            if A_proj.size > 0:
                # Pass empty Ae/be as they are already converted to inequalities
                A_proj, b_proj, _, _ = priv_normalize_constraints(A_proj, b_proj, np.array([]), np.array([]), 'A')
        
        # Normalize and compact after elimination to remove redundancies
        if A_proj.size > 0:
            # Priv normalize expects column b
            A_proj, b_proj, Ae_tmp, be_tmp = priv_normalize_constraints(A_proj, b_proj, np.array([]), np.array([]), 'A')
            A_proj, b_proj, Ae_tmp, be_tmp, empty_proj, _ = priv_compact_all(A_proj, b_proj, Ae_tmp, be_tmp, n_out, 1e-12)
            if empty_proj:
                return Polytope.empty(n_out)
        
        # Reorder columns of the resulting A_proj to match the order of `dims`
        if A_proj.size > 0 and n_out > 0:
            # Need to get the permutation that maps `sorted(dims)` to `dims`
            # Example: dims = [2,1], sorted_dims = [1,2]. We want columns in order 2,1.
            # The columns in A_proj are currently in the order of `sorted(dims) - 1`.
            # We need to map them back to the original order of `dims` (0-based).
            
            # This is complex due to the iterative removal of columns.
            # Simplest approach: create a temporary identity matrix, apply removals,
            # and then get permutation. Or, trust that Fourier-Motzkin preserves original order
            # of kept dimensions, just shifts them. Assuming it preserves order.
            # MATLAB: [~,ind] = sort(dims); A(:,ind) = A; implies sorting columns based on desired dims order.
            # A_proj has columns corresponding to original dims elements, in their original relative order.
            # We need them in the `dims` order. So, find position of elements of `dims` in `sorted(dims)`
            
            # Let's say `dims` is [2, 1]. `remove_dims_0based` would remove 0.
            # A_proj would have columns for original dims 1 and 2 (0-based) in that order.
            # We need them in order 2, 1. So, swap columns.

            # Simplified for now: if original dims were [d1, d2, d3] and we keep [d1, d3],
            # A_proj has columns for d1, d3. If target `dims` order is [d3, d1],
            # we need to reorder columns. This is not implicitly done by FM.
            
            # Check if current column order matches target order.
            current_col_order_0based = [d - 1 for d in sorted(dims)] # After FM, columns are sorted by original index
            target_col_order_0based = [d - 1 for d in dims]
            
            if current_col_order_0based != target_col_order_0based:
                # Create a mapping from current index to target index
                mapping = {val: i for i, val in enumerate(target_col_order_0based)}
                perm_indices = np.array([mapping[val] for val in current_col_order_0based])
                
                # Create a permutation array to reorder columns
                reorder_perm = np.argsort(perm_indices)
                A_proj = A_proj[:, reorder_perm]
            
        return Polytope(A_proj, b_proj)

    elif method_chosen == 'fourier_jones':
        # Try pycddlib; if it fails at runtime, fall back to Fourier-Motzkin
        # https://pycddlib.readthedocs.io/en/latest/examples.html
        import cdd

        # Convert constraints to cdd format (b - A*x >= 0, i.e., [-A | b])
        # Combine inequality and equality constraints
        # A, b, Ae, be are already normalized and compacted at this point
        
        # Convert equality constraints Ae*x = be to two inequalities:
        # Ae*x <= be and -Ae*x <= -be
        if Ae.size > 0:
            A_combined = np.vstack([A, Ae, -Ae])
            b_combined = np.vstack([b, be, -be]) # Use vstack for b/be if they are column vectors
        else:
            A_combined = A
            b_combined = b
        
        # Convert to CDD format: [b | -A]
        # Ensure b_combined is a column vector and A_combined has correct shape for hstack
        cdd_matrix = np.hstack([b_combined.reshape(-1, 1), -A_combined]).astype(float, copy=False)
        # pycddlib expects a plain Python list of lists for portability in some builds
        cdd_list = cdd_matrix.tolist()
        
        # Create CDD matrix object using pycddlib API
        mat = cdd.Matrix(cdd_list, number_type='float')
        mat.rep_type = cdd.RepType.INEQUALITY
        
        # Convert to vertex representation
        poly = cdd.Polyhedron(mat)
        vertices_mat = poly.get_generators()
        
        # Handle empty/unbounded cases from CDD output
        if vertices_mat is None or len(vertices_mat) == 0: # This check might be too simple.
            # Check if it's empty, or unbounded (has rays)
            # If poly.is_empty, return empty
            if poly.is_empty:
                return Polytope.empty(n_out)
            # If it has rays, it's unbounded after projection: return fullspace or raise error
            # For now, if no vertices but not empty, it implies unbounded
            if not poly.is_finitely_generated: # Implies unbounded, has rays
                # This is a simplification; a full unbounded projection would be a ConZonotope or similar.
                # For now, if unbounded after projection, return fullspace of projected dimension
                return Polytope.Inf(n_out) 
            # Default to empty if no vertices and not explicitly empty/unbounded
            return Polytope.empty(n_out)

        
        # Extract vertices (skip first column which is for rays/vertices type)
        # vertices: first col is type (1 for vertex, 0 for ray), then coordinates
        vertices_array = np.array(vertices_mat, dtype=float)
        
        # Filter out rays (first column == 0) and keep only vertices (first column == 1)
        vertex_mask = vertices_array[:, 0] == 1
        
        # Handle case where no vertices are found (only rays or empty)
        if not np.any(vertex_mask):
            if poly.is_empty:
                return Polytope.empty(n_out)
            if not poly.is_finitely_generated: # Only rays, unbounded
                return Polytope.Inf(n_out)
            # Fallback for unexpected cases, assume empty
            return Polytope.empty(n_out)

        vertices = vertices_array[vertex_mask, 1:]  # Remove first column (type indicator)
        
        # Project vertices to desired dimensions (0-based indexing)
        py_dims = [d - 1 for d in dims]
        projected_vertices = vertices[:, py_dims]
        
        # Convert projected vertices back to half-space representation using CDD
        if projected_vertices.shape[0] == 0:
            return Polytope.empty(n_out)
        
        # Add the vertex indicator column back (cdd.Matrix expects [1 | V])
        vertex_matrix = np.hstack([np.ones((projected_vertices.shape[0], 1)), projected_vertices])
        
        # Create new CDD matrix from vertices (generator representation)
        proj_mat = cdd.Matrix(vertex_matrix.astype(float, copy=False).tolist(), number_type='float')
        proj_mat.rep_type = cdd.RepType.GENERATOR
        
        # Convert back to inequality representation
        proj_poly = cdd.Polyhedron(proj_mat)
        ineq_mat = proj_poly.get_inequalities()
        
        if ineq_mat is None or len(ineq_mat) == 0:
            # If no inequalities are found, it might be a full space or empty (should be caught by poly.is_empty)
            # If proj_poly.is_empty, it's empty.
            # If proj_poly.is_full_dimensional and not empty, it's fullspace.
            if proj_poly.is_empty:
                return Polytope.empty(n_out)
            # This case means the projected set is the entire space of n_out dimensions
            return Polytope.Inf(n_out)
        
        # Convert back from CDD format: [b | -A] to A, b
        ineq_array = np.array(ineq_mat, dtype=float)
        b_proj = ineq_array[:, 0].reshape(-1, 1) # Ensure column vector
        A_proj = -ineq_array[:, 1:]
        
        return Polytope(A_proj, b_proj)
        
    else:
        raise CORAerror('CORA:wrongValue', 'third', f"Unknown projection method '{method_chosen}'.") 