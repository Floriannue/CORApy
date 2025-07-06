import numpy as np
from scipy.linalg import sqrtm
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


def priv_encEllipsoid(Z, mode):
    """
    Encloses a non-degenerate zonotope by an ellipsoid
    
    Syntax:
        E = priv_encEllipsoid(Z,mode)
    
    Inputs:
        Z - zonotope object
        mode - norm computation of zonotope: 'exact', 'ub_convex'
    
    Outputs:
        E - ellipsoid object
    
    Example:
        Z = zonotope([1;0],[2 -1 3; 0 1 2]);
        E = ellipsoid(Z,'outer:norm_bnd');
        
        figure; hold on;
        plot(Z);
        plot(E);
    
    References:
        [1] M. Cerny, "Goffin's algorithm for zonotopes" Kybernetika, vol. 48,
            no. 5, pp. 890--906, 2012
    """
    # extract zonotope information, i.e. center and generator matrix
    c = Z.c
    G = Z.G
    n, m = G.shape
    
    # there are more generators than dimensions: cannot "fit" the ellipse
    # directly to generators
    if n < m:
        # compute initial guess for ellipsoid containing Z ([1], Sec. 4.1) 
        Q0 = m * (G @ G.T)
        TOL = 1e-6  # Default tolerance from ellipsoid.empty(1).TOL
        Z0 = Z - c  # Center at origin
        
        # compute transformation matrix T s.t. T*ellipsoid(Q0) == unit hyper sphere
        T = np.linalg.inv(sqrtm(Q0))
        
        # apply same transform to zonotope (corrected)
        Zt = type(Z)(T @ Z0.c, T @ Z0.G)
        
        # Compute norm of transformed zonotope
        lambda_val = Zt.norm_(2, mode)
        
        # since transformed zonotope is still contained in transf. ellipsoid and
        # since radius of said ell. =1, we can choose radius of ell =
        # norm(Zt) and apply inverse transform inv(T) (implicitly done here)
        E = Ellipsoid((lambda_val + TOL)**2 * Q0, c.reshape(-1, 1))
    else:
        # direct mapping possible
        E = Ellipsoid(n * (G @ G.T), c.reshape(-1, 1))
    
    return E 