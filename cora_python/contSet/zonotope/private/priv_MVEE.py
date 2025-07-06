import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def priv_MVEE(Z):
    """
    Computes the Minimum-Volume-Enclosing ellipsoid (enclosing Z)
    
    Syntax:
        E = priv_MVEE(Z)
    
    Inputs:
        Z - zonotope object
    
    Outputs:
        E - ellipsoid object
    
    Example: 
        Z = zonotope([1;0],[2 -1 3; 0 1 2]);
        E = ellipsoid(Z,'outer:exact');
        
        figure; hold on;
        plot(Z); plot(E);
    
    References:
        [1] S. Boyd and L. Vandenberghe, Convex optimization. Cambridge
            university press, 2004
    """
    n, nrGen = Z.G.shape
    
    # check if efficient computation is possible
    if n == nrGen and np.linalg.matrix_rank(Z.G) == n:
        E = Ellipsoid(n * (Z.G @ Z.G.T), Z.c.reshape(-1, 1))
        return E
    
    # ATTENTION: Exact computation, requires vertices -> scales badly
    # see [1], Sec. 8.4.1 for more details
    # compute zonotope vertices
    # Bug (?) in vertices: First and last element are the same
    V = Z.vertices_()
    # Remove duplicates and ensure proper format
    V_unique = np.unique(V, axis=1)
    
    # Verify center computation
    center_check = np.mean(V_unique, axis=1).reshape(-1, 1)
    if not np.allclose(center_check, Z.c, atol=1e-10):
        raise CORAerror('CORA:wrongValue', 'mean of vertices must result in center')
    
    # Use the complete enclosePoints implementation with 'min-vol' method
    from cora_python.contSet.ellipsoid.enclosePoints import enclosePoints
    E = enclosePoints(V_unique, 'min-vol')
    
    return E


def _approximate_MVEE(V, center):
    """
    Approximate MVEE using bounding box approach.
    This is a fallback when exact MVEE computation is not available.
    """
    # Compute bounding box of vertices
    min_vals = np.min(V, axis=1)
    max_vals = np.max(V, axis=1)
    
    # Compute center and radii
    c = (min_vals + max_vals) / 2
    r = (max_vals - min_vals) / 2
    
    # Create diagonal shape matrix based on radii
    # Scale by n to ensure containment (similar to parallelotope case)
    n = len(r)
    Q = n * np.diag(r**2)
    
    return Ellipsoid(Q, c.reshape(-1, 1)) 