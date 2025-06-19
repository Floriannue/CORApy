import numpy as np
from scipy.spatial import ConvexHull, QhullError
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError

def priv_V_to_H(V):
    """
    Converts a vertex representation of a convex polytope to a half-space
    representation. It also handles degenerate (lower-dimensional) polytopes
    by finding their affine hull and extracting equality constraints.
    
    Args:
        V (np.ndarray): A matrix where each column is a vertex (d x n_vertices).
    
    Returns:
        tuple: (A, b, Ae, be) representing the polytope {x | A*x <= b, Ae*x = be}.
    """
    if V.shape[1] == 0:
        return np.empty((0, V.shape[0])), np.empty((0, 1)), None, None
        
    dim, n_vertices = V.shape

    # --- 1. Find the affine hull to extract equality constraints ---
    
    # Calculate the centroid and center the vertices
    centroid = np.mean(V, axis=1, keepdims=True)
    centered_V = V - centroid

    # Use SVD to find the basis of the subspace spanned by the centered vertices
    try:
        U, s, _ = np.linalg.svd(centered_V)
    except np.linalg.LinAlgError:
        # This can happen for very degenerate cases, e.g., all points are the same
        # Treat as a single point defined by equalities
        Ae = np.eye(dim)
        be = V[:, 0:1]
        return np.empty((0, dim)), np.empty((0, 1)), Ae, be

    # Determine the rank/dimension of the polytope
    tol = 1e-9 # Tolerance for singular values to be considered zero
    rank = np.sum(s > tol)

    Ae = None
    be = None

    if rank < dim:
        # The polytope is degenerate, lies in a lower-dimensional affine subspace
        # The normals to this subspace are the last (dim - rank) columns of U
        Ae = U[:, rank:].T
        # Calculate the RHS of the equality constraints
        be = Ae @ centroid
        
        # Project vertices onto their affine hull for ConvexHull calculation
        # The first `rank` columns of U form an orthonormal basis for the subspace
        basis = U[:, :rank]
        projected_V = basis.T @ centered_V
        
    else:
        # The polytope is full-dimensional, no equality constraints
        projected_V = V

    # --- 2. Find inequality constraints using ConvexHull ---

    if projected_V.shape[1] <= projected_V.shape[0]:
        # Not enough points to form a convex hull in this dimension
        # This can happen if, e.g., 3 points are co-linear in 2D space after projection.
        # In this case, there are no inequality constraints, only equalities.
        A = np.empty((0, dim))
        b = np.empty((0, 1))
        return A, b, Ae, be

    try:
        # Transpose for ConvexHull which expects points as rows
        hull = ConvexHull(projected_V.T, qhull_options='QJ')
        
        # Equations are given as Ax + b <= 0, we want Ax <= -b
        A_proj = hull.equations[:, :-1]
        b_proj = -hull.equations[:, -1:]
        
        if rank < dim:
            # Lift the half-spaces back to the original dimension
            A = A_proj @ basis.T
            # The offset `b` needs to account for the projection and centroid
            b = b_proj + (A_proj @ basis.T @ centroid)
        else:
            A = A_proj
            b = b_proj

    except (QhullError, ValueError):
        # QhullError can happen for degenerate inputs not caught earlier
        # e.g., co-linear points in 2D.
        # Treat as having no inequality constraints.
        A = np.empty((0, dim))
        b = np.empty((0, 1))

    return A, b, Ae, be 