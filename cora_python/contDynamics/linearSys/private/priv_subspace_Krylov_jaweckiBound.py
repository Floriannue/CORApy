"""
priv_subspace_Krylov_jaweckiBound - computes the Krylov subspaces for the center and
generators of a zonotope
however, the error bound is computed as given in [1]

Syntax:
    [V_c,H_c,V_g,H_g,Hlast_c] = priv_subspace_Krylov_jaweckiBound(A,Z,options)

Inputs:
    A - state matrix of linear system
    Z - zonotope
    options - reachability options

Outputs:
    V_c - orthonormal basis of center
    H_c - Hessenberg matrix of center
    V_g - orthonormal basis of generators
    H_g - Hessenberg matrix of generators
    Hlast_c - last H value for center

Example: 
    -

References:
    [1] Computable upper error bounds for Krylov approximations to
    matrix exponentials and associated phi-functions, Jawecki et al, 2019

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Maximilian Perschl
Written:       25-April-2025
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from scipy.sparse import issparse
from typing import Tuple, Dict, Any, List, Optional
from cora_python.contSet.zonotope.zonotope import Zonotope
from .priv_subspace_Krylov_individual_Jawecki import priv_subspace_Krylov_individual_Jawecki


def priv_subspace_Krylov_jaweckiBound(A: np.ndarray, Z: Zonotope, 
                                      options: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], 
                                                                       List[np.ndarray], List[np.ndarray], Optional[float]]:
    """
    Computes the Krylov subspaces for the center and generators of a zonotope
    
    Args:
        A: state matrix of linear system
        Z: zonotope
        options: reachability options
        
    Returns:
        V_c: orthonormal basis of center (or None if center is zero)
        H_c: Hessenberg matrix of center (or None if center is zero)
        V_g: list of orthonormal bases of generators
        H_g: list of Hessenberg matrices of generators
        Hlast_c: last H value for center (or None if center is zero)
    """
    
    # Convert to numpy arrays if sparse
    if issparse(A):
        A = A.toarray()
    
    # obtain center and generators of previous reachable set
    c = Z.center() if hasattr(Z, 'center') else Z.c
    G = Z.generators() if hasattr(Z, 'generators') else Z.G
    
    # Convert to numpy arrays
    if issparse(c):
        c = c.toarray().flatten()
    else:
        c = np.asarray(c).flatten()
    
    if issparse(G):
        G = G.toarray()
    else:
        G = np.asarray(G)
    
    # init Krylov order
    KrylovOrder = 15
    
    # check if center is zero
    c_norm = np.linalg.norm(c)
    if c_norm == 0:
        V_c = None
        H_c = None
        Hlast_c = None
    else:
        # Arnoldi; Krylov order is passed on to computation of next generator
        V_c, H_c, KrylovOrder, Hlast_c = priv_subspace_Krylov_individual_Jawecki(A, c, KrylovOrder, options)
    
    # preallocation
    nrOfGens = G.shape[1] if G.ndim > 1 and G.shape[1] > 0 else 0
    V_g = []
    H_g = []
    
    # obtain generators using the Arnoldi iteration
    if nrOfGens > 0:
        for iGen in range(nrOfGens):
            g = G[:, iGen]
            g_norm = np.linalg.norm(g)
            if g_norm == 0:
                V_g.append(None)
                H_g.append(None)
            else:
                # Arnoldi
                V_gen, H_gen, KrylovOrder, _ = priv_subspace_Krylov_individual_Jawecki(A, g, KrylovOrder, options)
                V_g.append(V_gen)
                H_g.append(H_gen)
    else:
        V_g = []  # no generators
        H_g = []  # no generators
    
    return V_c, H_c, V_g, H_g, Hlast_c

