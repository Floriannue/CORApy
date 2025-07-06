import numpy as np
from scipy.linalg import sqrtm
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


def priv_inscEllipsoid(Z):
    """
    Underapproximates a non-degenerate zonotope by an ellipsoid
    
    Syntax:
        E = priv_inscEllipsoid(Z)
    
    Inputs:
        Z - zonotope object
    
    Outputs:
        E - ellipsoid object
    
    Example: 
        Z = zonotope([1;0],[2 -1 3; 0 1 2]);
        E = ellipsoid(Z,'inner:norm');
        
        figure; hold on;
        plot(Z); plot(E);
    
    References:
        [1] M. Cerny, "Goffin's algorithm for zonotopes" Kybernetika, vol. 48,
            no. 5, pp. 890-906, 2012
    """
    # extract zonotope information, i.e. center and generator matrix
    c = Z.c
    G = Z.G
    n, m = G.shape
    
    # initial guess contained in Z analogously to [1], Sec. 4.1
    E0 = G @ G.T
    
    # cannot directly fit ellipsoid to zonotope
    if n < m:
        # transformation matrix T s.t. T*ellipsoid(E0) == unit hyper sphere
        T = np.linalg.inv(sqrtm(E0))
        # transform Z (corrected)
        Zt = type(Z)(T @ Z.c, T @ Z.G)
        # Compute minimum norm of transformed zonotope
        l, _ = Zt.minnorm()
        # since transf. ellipsoid Et has radius 1 and Et \in Zt, scaling with l 
        # results in Et touching Zt (for exact norm computation at least at some 
        # point (thus optimal in that sense) (implicit application of inv(T))
        E = Ellipsoid(l**2 * E0, c.reshape(-1, 1))
    # can directly compute the optimal E
    else:
        E = Ellipsoid(E0, c.reshape(-1, 1))
    
    return E 