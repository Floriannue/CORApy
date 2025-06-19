import numpy as np
from typing import List, TYPE_CHECKING
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError

if TYPE_CHECKING:
    from cora_python.contSet.polytope.polytope import Polytope

def _aux_fourier_motzkin_elimination(A: np.ndarray, b: np.ndarray, j: int) -> (np.ndarray, np.ndarray):
    
    Z_indices = np.where(A[:, j] == 0)[0]
    N_indices = np.where(A[:, j] < 0)[0]
    P_indices = np.where(A[:, j] > 0)[0]
    
    A_new = []
    b_new = []

    # Keep constraints where the j-th variable is not present
    if len(Z_indices) > 0:
        A_new.append(A[Z_indices, :])
        b_new.append(b[Z_indices])

    # Combine positive and negative constraints
    for p_idx in P_indices:
        for n_idx in N_indices:
            A_p, b_p = A[p_idx, :], b[p_idx]
            A_n, b_n = A[n_idx, :], b[n_idx]
            
            # Elimination step
            A_comb = A_p[j] * A_n - A_n[j] * A_p
            b_comb = A_p[j] * b_n - A_n[j] * b_p
            
            A_new.append(A_comb)
            b_new.append(b_comb)

    if not A_new:
        dim = A.shape[1] -1
        return np.empty((0, dim)), np.empty((0, 1))

    A_res = np.vstack(A_new)
    b_res = np.vstack(b_new)
    
    # Remove the eliminated column j
    A_res = np.delete(A_res, j, axis=1)

    return A_res, b_res


def project(P: 'Polytope', dims: List[int], method: str = 'fourier') -> 'Polytope':

    from cora_python.contSet.polytope.polytope import Polytope
    n = P.dim()
    
    if sorted(dims) == list(range(1, n + 1)):
        return P.copy()
        
    if any(d > n for d in dims):
        raise CORAError('CORA:wrongValue', 'second', f'Cannot compute projection on higher dimension than {n}.')

    if P._has_v_rep:
        # Adjust dims for 0-based indexing
        py_dims = [d - 1 for d in dims]
        return Polytope(P.V[py_dims, :])
    
    # Projection for H-representation
    from .private.priv_normalize_constraints import priv_normalize_constraints
    from .private.priv_compact_all import priv_compact_all
    from .private.priv_equality_to_inequality import priv_equality_to_inequality
    
    tol = 1e-12
    if P.isemptyobject:
        return Polytope.empty(len(dims))

    # Normalize and compact constraints
    A, b, Ae, be = priv_normalize_constraints(P.A, P.b, P.Ae, P.be, 'A')
    A, b, Ae, be = priv_compact_all(A, b, Ae, be, n, tol)
    
    if method == 'fourier':
        A, b = priv_equality_to_inequality(A, b, Ae, be)

        # Dimensions to be projected away
        remove_dims = sorted(list(set(range(n)) - set(d - 1 for d in dims)), reverse=True)
        
        A_proj, b_proj = A, b
        for j in remove_dims:
            A_proj, b_proj = _aux_fourier_motzkin_elimination(A_proj, b_proj, j)
            
            # Normalize and compact again after each elimination step
            A_proj, b_proj, _, _ = priv_normalize_constraints(A_proj, b_proj, np.array([]), np.array([]), 'A')
            A_proj, b_proj, _, _ = priv_compact_all(A_proj, b_proj, np.array([]), np.array([]), A_proj.shape[1], tol)
        
        # Reorder columns to match 'dims'
        original_dims_kept = sorted(list(set(range(n)) - set(remove_dims)))
        if original_dims_kept:
            py_dims = [d-1 for d in dims]
            order = [original_dims_kept.index(d) for d in py_dims]
            A_proj = A_proj[:, order]
        
        return Polytope(A_proj, b_proj)

    elif method == 'fourier_jones':
        import pypoman as pm

        ineq = (A, b)
        eq = (Ae, be) if Ae.size > 0 else None

        # Define projection matrix
        p = len(dims)
        E = np.zeros((p, n))
        py_dims = [d - 1 for d in dims]
        for i, dim_idx in enumerate(py_dims):
            E[i, dim_idx] = 1.0
        
        f = np.zeros(p)
        proj = (E, f)

        try:
            # Project polytope. This returns vertices of the projected polytope.
            vertices = pm.project_polytope(proj, ineq, eq, method='bretl')
            # Convert vertices back to half-space representation
            A_proj, b_proj = pm.compute_polytope_halfspaces(vertices)
            return Polytope(A_proj, b_proj)
        except Exception as e:
            # pypoman can fail if cdd is not installed correctly.
            # Or if the polytope is unbounded in a way that projection is complex.
            raise CORAError('CORA:thirdPartyError', f"Polytope projection with 'pypoman' failed. Error: {e}")
    
    else:
        raise CORAError('CORA:wrongValue', 'third', f"Unknown projection method '{method}'.") 