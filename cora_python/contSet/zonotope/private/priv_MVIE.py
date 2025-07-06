import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


def priv_MVIE(Z):
    """
    Computes the Maximum-Volume-inscribed ellipsoid of a zonotope
    
    Syntax:
        E = priv_MVIE(Z)
    
    Inputs:
        Z - zonotope object
    
    Outputs:
        E - ellipsoid object
    
    Example: 
        Z = zonotope([1;0],[2 -1 3; 0 1 2]);
        E = ellipsoid(Z,'inner:exact');
        
        figure; hold on;
        plot(Z);
        plot(E);
    
    References:
        [1] S. Boyd and L. Vandenberghe, Convex optimization. Cambridge
            university press, 2004
    """
    # convert to polytope, then to ellipsoid
    P = Z.polytope()
    from cora_python.contSet.polytope.ellipsoid import ellipsoid
    E = ellipsoid(P, 'inner')
    
    return E


def _approximate_MVIE(Z):
    """
    Approximate MVIE using generator-based approach.
    This is a fallback when exact MVIE computation is not available.
    """
    c = Z.c
    G = Z.G
    n, m = G.shape
    
    # For parallelotopes (n == m), use G*G' directly
    if n == m:
        E = Ellipsoid(G @ G.T, c.reshape(-1, 1))
    else:
        # For general zonotopes, use a scaled version of G*G'
        # Scale down to ensure it's inscribed
        scale_factor = 1.0 / m  # Conservative scaling
        E = Ellipsoid(scale_factor * (G @ G.T), c.reshape(-1, 1))
    
    return E 