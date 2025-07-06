import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from .private.priv_MVEE import priv_MVEE
from .private.priv_MVIE import priv_MVIE
from .private.priv_encEllipsoid import priv_encEllipsoid
from .private.priv_inscEllipsoid import priv_inscEllipsoid

def ellipsoid(Z, mode='outer:norm_bnd'):
    """
    Converts a zonotope to an ellipsoid.
    
    Syntax:
        E = ellipsoid(Z)
        E = ellipsoid(Z,mode)
    
    Inputs:
        Z - zonotope object
        mode - (optional) specifies whether function uses a bound on the 
               respective zonotope norm or the exact value:
               - 'outer:exact':    Uses MVEE(Z)
               - 'outer:norm':     Uses encEllipsoid with exact norm value
               - 'outer:norm_bnd': Uses encEllipsoid with upper bound for norm value (default)
               - 'inner:exact':    Uses MVIE(Z)
               - 'inner:norm'      Uses inscEllipsoid with exact norm value
    
    Outputs:
        E - ellipsoid object
    
    Example:
        Z = zonotope([1;-1],[2 -4 3 2 1; 3 2 -4 -2 1]);
        E = ellipsoid(Z);
    """
    valid_modes = ['outer:exact', 'outer:norm', 'outer:norm_bnd', 'inner:exact', 'inner:norm']
    if mode not in valid_modes:
        raise CORAerror('CORA:wrongValue', 'second',
                        "'outer:exact','outer:norm','outer:norm_bnd','inner:exact' or 'inner:norm'")
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
    
    # obtain information about zonotope
    n = Z.dim()
    
    # zonotope is just a point
    if Z.representsa_('point', eps=1e-14):
        E = Ellipsoid(np.zeros((n, n)), Z.c.reshape(-1, 1))
        return E
    
    # exact information
    c = Z.c
    G = Z.G
    Grank = np.linalg.matrix_rank(G)
    
    # reduce dimension of zonotope if degenerate
    if n != Grank:
        # compute QR decomposition
        Q, R = np.linalg.qr(G)
        
        # obtain reduced zonotope centered around the origin
        from cora_python.contSet.zonotope.zonotope import Zonotope
        Z_reduced = Zonotope(np.hstack([np.zeros((Grank, 1)), R[:Grank, :]]))
        
        # obtain transformation matrix
        T = Q[:, :Grank]
        
        # update for processing
        Z = Z_reduced
        c_orig = c
        c = Z.c
        G = Z.G
        n = Grank
    
    # Zonotope is a parallelotope
    if G.shape[1] == n:
        fac = n
        if mode.startswith('inner'):
            fac = 1
        E = Ellipsoid(fac * (G @ G.T), c.reshape(-1, 1))
    # Zonotope is not a parallelotope  
    else:
        if mode == 'outer:exact':
            # Exact outer enclosure using MVEE
            E = priv_MVEE(Z)
        elif mode == 'outer:norm':
            # Norm-based outer enclosure with exact norm value
            E = priv_encEllipsoid(Z, 'exact')
        elif mode == 'outer:norm_bnd':
            # Norm-based outer enclosure with upper bound for norm value
            E = priv_encEllipsoid(Z, 'ub_convex')
        elif mode == 'inner:exact':
            # Exact inner enclosure using MVIE
            E = priv_MVIE(Z)
        elif mode == 'inner:norm':
            # Norm-based inner enclosure
            E = priv_inscEllipsoid(Z)
    
    # convert to original space if zonotope is not full-dimensional
    if 'T' in locals():
        # E = T*E + c_orig (affine transformation of ellipsoid)
        # For ellipsoid E with shape matrix Q and center q: T*E + d = ellipsoid(T*Q*T', T*q + d)
        E_shape = T @ E.Q @ T.T
        E_center = T @ E.q + c_orig.reshape(-1, 1)
        E = Ellipsoid(E_shape, E_center)
    
    return E 