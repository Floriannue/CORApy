"""
priv_subspace_Krylov_individual_Jawecki - computes the Krylov subspace for a single
vector given the accuracy to be achieved; the a-posteriori approach in
equation 3.6 of [1] is used for tight error computation

Syntax:
    [V,H,krylovOrder,Hlast] = priv_subspace_Krylov_individual_Jawecki(A,v,initKrylovOrder,options)

Inputs:
    A - system matrix
    v - vector
    initKrylovOrder - Krylov error that is first tested
    options - reachability options
              needs: krylovError, krylovStep and tFinal

Outputs:
    V - orthonormal basis
    H - Hessenberg matrix
    KrylovOrder - dimension of the reduced system
    Hlast - last computed H_ij scalar of the Arnoldi iteration

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
from scipy.special import factorial
from typing import Tuple, Dict, Any
from cora_python.g.functions.helper.dynamics.contDynamics.linearSys.arnoldi import arnoldi


def priv_subspace_Krylov_individual_Jawecki(A: np.ndarray, v: np.ndarray, 
                                            initKrylovOrder: int, 
                                            options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """
    Computes the Krylov subspace for a single vector given the accuracy to be achieved
    
    Args:
        A: system matrix
        v: vector
        initKrylovOrder: Krylov error that is first tested
        options: reachability options (needs: krylovError, krylovStep, tFinal)
        
    Returns:
        V: orthonormal basis
        H: Hessenberg matrix
        krylovOrder: dimension of the reduced system
        Hlast: last computed H_ij scalar of the Arnoldi iteration
    """
    
    # Convert to numpy arrays if sparse
    if issparse(A):
        A = A.toarray()
    if issparse(v):
        v = v.toarray().flatten()
    else:
        v = np.asarray(v).flatten()
    
    # set precision for variable precision toolbox
    # NOTE: MATLAB uses mp (multiple precision) for high precision
    # In Python, we'll use regular float for now, but note where precision matters
    precision = 34  # Not used directly, but kept for reference
    
    # compute norm of v
    v_norm = np.linalg.norm(v)
    
    # maximum allowed error
    maxRelError = options.get('krylovError', 1e-6)
    
    # initialize Krylov order and normalized error
    krylovOrder = initKrylovOrder - options.get('krylovStep', 1)
    if krylovOrder <= 0:
        krylovOrder = 1
    
    errorBound = np.inf
    dim = len(A)
    
    # Krylov order should not be larger than dimension
    if krylovOrder > dim:
        krylovOrder = dim
    
    Hlast = 0.0
    V = None
    H = None
    
    while (v_norm * errorBound > maxRelError) and (krylovOrder <= dim):
        
        # increment Krylov order
        krylovOrder = krylovOrder + options.get('krylovStep', 1)
        
        # perform Arnoldi iteration
        V, H, Hlast, happyBreakdown = arnoldi(A, v, krylovOrder)
        
        # sparsify (convert to sparse if needed, but keep as dense for now)
        # V = sparse(V);  # In Python, we can use scipy.sparse if needed
        # H = sparse(H);
        
        # compute error if happy breakdown did not occur
        if not happyBreakdown:
            # compute/convert necessary parameters
            # NOTE: MATLAB uses mp (multiple precision) here
            # For Python, we'll use regular float but note precision may be lost
            tau = float(Hlast)
            
            gamma = 1.0
            
            for i in range(krylovOrder - 1):
                gamma = gamma * float(H[i + 1, i])
            
            # compute bound according to (Err_a) in [1]
            # for t we choose t_f since it leads to the largest error using this
            # timestep size
            # this line is computed separately so the product isn't done in mp
            tFinal = options.get('tFinal', 1.0)
            right_half = (tFinal ** krylovOrder) / factorial(krylovOrder)
            errorBound = tau * gamma * right_half
            
            if np.isnan(errorBound):  # if error is not a number (NaN)
                errorBound = np.inf
        else:
            # decrease Krylov order
            krylovOrder = V.shape[1] if V is not None else krylovOrder
            break
    
    return V, H, krylovOrder, Hlast

