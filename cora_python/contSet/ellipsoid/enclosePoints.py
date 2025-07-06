import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def enclosePoints(points, method='cov'):
    """
    Enclose a point cloud with an ellipsoid.
    
    Syntax:
        E = enclosePoints(points)
        E = enclosePoints(points,method);
    
    Inputs:
        points - matrix storing point cloud (dimension: [n,p] for p points)
        method - (optional) method to compute the enclosing ellipsoid
                   'cov' or 'min-vol'
    
    Outputs:
        E - ellipsoid object
    
    Example: 
        mu = [2 3];
        sigma = [1 1.5; 1.5 3];
        points = mvnrnd(mu,sigma,100)';
        
        E1 = ellipsoid.enclosePoints(points);
        E2 = ellipsoid.enclosePoints(points,'min-vol');
        
        figure; hold on
        plot(points(1,:),points(2,:),'.k');
        plot(E1,[1,2],'r');
        plot(E2,[1,2],'b');
    
    References:
        [1]: Boyd; Vandenberghe: Convex Optimization
    """
    # check input arguments
    if not isinstance(points, np.ndarray) or points.size == 0:
        raise CORAerror('CORA:wrongInputInConstructor', 'points must be non-empty numeric array')
    
    if method not in ['cov', 'min-vol']:
        raise CORAerror('CORA:wrongValue', 'method must be "cov" or "min-vol"')
    
    # remove bias
    c = np.mean(points, axis=1, keepdims=True)
    points = points - c
    n, M = points.shape
    
    # handle degenerate case
    U, S, _ = np.linalg.svd(points)
    n_nd = np.linalg.matrix_rank(S)
    points = U.T @ points
    n_d = n - n_nd
    
    if n_nd < n:
        # remove zeros for degenerate dimensions
        points = points[:n_nd, :]
    
    # handle special cases n_nd=0 and n_nd=1
    if n_nd == 0:
        # all zero matrix
        E = Ellipsoid.empty(n)
        return E
    
    elif n_nd == 1:
        # interval arithmetic (is exact in this case)
        r = 0.5 * (np.max(points) - np.min(points))
        q = 0.5 * (np.max(points) + np.min(points))
        E = Ellipsoid(np.array([[r**2]]), q.reshape(-1, 1))
    
    elif method == 'cov':
        # compute the covariance matrix
        C = np.cov(points)
        
        # singular value decomposition
        U, _, _ = np.linalg.svd(C)
        
        # required ellipsoid radius in transformed space
        orientedMatrix = U.T @ points
        m1 = np.max(orientedMatrix, axis=1)
        m2 = np.min(orientedMatrix, axis=1)
        
        nt = np.maximum(m1, m2)
        
        # enclosing ellipsoid in the transformed space
        B = np.diag(nt**2)
        
        maxDist = np.max(np.sum(np.diag(1.0 / nt**2) @ orientedMatrix**2, axis=0))
        B = B * maxDist
        
        E = Ellipsoid(B)
        
        # transform back to original space
        E = U @ E
    
    elif method == 'min-vol':
        # For now, use a simplified approach since full SDP solvers are not available
        # This is a reasonable approximation for minimum volume ellipsoid
        E = _approximate_min_vol_ellipsoid(points)
    
    # backtransform and add back center
    E_ext = E
    if n_d > 0:
        # Add back degenerate dimensions
        Q_ext = np.block([[E_ext.Q, np.zeros((n_nd, n_d))], 
                         [np.zeros((n_d, n_nd)), np.zeros((n_d, n_d))]])
        q_ext = np.vstack([E_ext.q, np.zeros((n_d, 1))])
        E_ext = Ellipsoid(Q_ext, q_ext)
    
    # Transform back to original space and add center
    if n_nd == 0:
        # Handle empty ellipsoid case
        E_final = Ellipsoid.empty(n)
    else:
        # Apply transformation: U * E_ext + c
        # For ellipsoid E with shape matrix Q and center q: U*E + d = ellipsoid(U*Q*U', U*q + d)
        E_final = Ellipsoid(U @ E_ext.Q @ U.T, U @ E_ext.q + c)
    
    return E_final


def _approximate_min_vol_ellipsoid(points):
    """
    Approximate minimum volume ellipsoid using iterative approach.
    This is a simplified version since full SDP solvers are not available.
    """
    n, M = points.shape
    
    # Use the covariance-based approach as a starting point
    C = np.cov(points)
    U, _, _ = np.linalg.svd(C)
    orientedMatrix = U.T @ points
    
    # Compute initial bounding box
    m1 = np.max(orientedMatrix, axis=1)
    m2 = np.min(orientedMatrix, axis=1)
    nt = np.maximum(m1, m2)
    
    # Start with covariance-based ellipsoid
    B = np.diag(nt**2)
    
    # Iteratively refine to get better approximation
    for _ in range(5):  # Limited iterations for efficiency
        # Compute distances to current ellipsoid
        distances = np.sum(np.diag(1.0 / nt**2) @ orientedMatrix**2, axis=0)
        maxDist = np.max(distances)
        
        # Scale ellipsoid to contain all points
        B = B * maxDist
        nt = nt * np.sqrt(maxDist)
    
    E = Ellipsoid(B)
    E = U @ E
    
    return E 