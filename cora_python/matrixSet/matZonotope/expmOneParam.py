"""
expmOneParam - operator for the exponential matrix of a matrix zonotope,
   evaluated dependently. Higher order terms are computed via interval
   arithmetic.

Syntax:
    [eZ,eI,zPow,iPow,E,RconstInput] = expmOneParam(matZ, r, maxOrder)
    [eZ,eI,zPow,iPow,E,RconstInput] = expmOneParam(matZ, r, maxOrder, params)

Inputs:
    matZ - matZonotope object
    r - time step size
    maxOrder - maximum Taylor series order until remainder is computed
    params - model parameters (inputs, optional)

Outputs:
    eZ - matrix zonotope exponential part
    eI - interval matrix exponential part
    zPow - list of matrix zonotope powers
    iPow - empty list (no interval powers for one parameter case)
    E - interval matrix for the remainder
    RconstInput - constant input zonotope

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

import math
import numpy as np
from typing import Tuple, List, Optional, Dict, Any, TYPE_CHECKING
from .matZonotope import matZonotope
# center is attached to matZonotope class, use object.center()
from .dim import dim
from .numgens import numgens
from cora_python.contSet.zonotope import Zonotope
from cora_python.matrixSet.intervalMatrix import IntervalMatrix
from cora_python.matrixSet.intervalMatrix.exponentialRemainder import exponentialRemainder

if TYPE_CHECKING:
    from .matZonotope import matZonotope


def expmOneParam(matZ: 'matZonotope', r: float, maxOrder: int, 
                 params: Optional[Dict[str, Any]] = None) -> Tuple['matZonotope', 'IntervalMatrix', 
                                                                    List['matZonotope'], List, 
                                                                    'IntervalMatrix', Zonotope]:
    """
    Computes exponential matrix for one-parameter matrix zonotope
    
    Args:
        matZ: matZonotope object (must have 1 generator)
        r: time step size
        maxOrder: maximum Taylor series order
        params: Model parameters dict (optional, may contain 'uTrans')
        
    Returns:
        eZ: Matrix zonotope exponential part
        eI: Interval matrix exponential part
        zPow: List of matrix zonotope powers
        iPow: Empty list
        E: Interval matrix remainder
        RconstInput: Constant input zonotope
    """
    # Cannot directly use u as input since zonotope has preference over matZonotopes
    # MATLAB: if length(varargin) == 1
    if params is not None:
        # MATLAB: if ~isa(params.uTrans,'zonotope')
        if 'uTrans' in params:
            uTrans = params['uTrans']
            if not isinstance(uTrans, Zonotope):
                # MATLAB: u = zonotope([params.uTrans,zeros(size(params.uTrans))]);
                uTrans_array = np.asarray(uTrans)
                if uTrans_array.ndim == 1:
                    uTrans_array = uTrans_array.reshape(-1, 1)
                u = Zonotope(uTrans_array, np.zeros((uTrans_array.shape[0], 0)))
            else:
                # MATLAB: u = zonotope(params.uTrans);
                u = Zonotope(uTrans.center(), uTrans.generators())
        else:
            u = Zonotope(np.array([[0.0]]), np.zeros((1, 0)))
    else:
        # MATLAB: u = zonotope([0,0]);
        u = Zonotope(np.array([[0.0]]), np.zeros((1, 0)))
    
    # Obtain matrix center and generator
    # MATLAB: C = matZ.center;
    # MATLAB: G1 = matZ.G(:,:,1);
    C = matZ.center()
    G1 = matZ.G[:, :, 0]  # Python 0-indexed
    
    # Obtain center and generator of input uTrans
    # MATLAB: c_u = u.c;
    # MATLAB: g_u = u.G;
    c_u = u.center()
    g_u = u.generators()
    
    # Pre-allocate matrices D, E (center and generators of powers)
    # MATLAB: D = nan([size(C),maxOrder]);
    # MATLAB: E = nan([size(G1),maxOrder,maxOrder]);
    n_dims, m_dims = C.shape
    D = np.full((n_dims, m_dims, maxOrder), np.nan)
    E = np.full((n_dims, m_dims, maxOrder, maxOrder), np.nan)
    
    if np.isscalar(c_u) or (isinstance(c_u, np.ndarray) and c_u.size == 1):
        # Use dimension of state
        # MATLAB: D_u = nan(size(D));
        # MATLAB: E_u = nan(size(E));
        D_u = np.full((n_dims, m_dims, maxOrder), np.nan)
        E_u = np.full((n_dims, m_dims, maxOrder, maxOrder), np.nan)
    else:
        # Use dimension of input
        # MATLAB: D_u = nan([size(c_u),maxOrder]);
        # MATLAB: E_u = nan([size(g_u),maxOrder,maxOrder]);
        c_u_shape = c_u.shape if isinstance(c_u, np.ndarray) else (len(c_u), 1)
        g_u_shape = g_u.shape if g_u.size > 0 else (c_u_shape[0], 0)
        D_u = np.full((*c_u_shape, maxOrder), np.nan)
        E_u = np.full((*g_u_shape, maxOrder, maxOrder), np.nan)
    
    zPow = [None] * maxOrder
    
    # Init first entry
    # MATLAB: D(:,:,1) = C;
    # MATLAB: E(:,:,1,1) = G1;
    D[:, :, 0] = C
    E[:, :, 0, 0] = G1
    
    # MATLAB: D_u(:,:,1) = c_u;
    # MATLAB: E_u(:,:,1,1) = g_u;
    D_u[:, :, 0] = c_u if c_u.ndim == 2 else c_u.reshape(-1, 1)
    if g_u.size > 0:
        E_u[:, :, 0, 0] = g_u if g_u.ndim == 2 else g_u.reshape(-1, 1)
    
    # Update power cell
    # MATLAB: zPow{1} = matZ*r;
    zPow[0] = matZ * r
    
    # The first cell index refers to the power!
    # MATLAB: for n = 2:maxOrder
    for n in range(2, maxOrder + 1):
        # MATLAB: D(:,:,n) = D(:,:,n-1)*C;
        D[:, :, n - 1] = D[:, :, n - 2] @ C
        # MATLAB: E(:,:,1,n) = D(:,:,n-1)*G1 + E(:,:,1,n-1)*C;
        E[:, :, 0, n - 1] = D[:, :, n - 2] @ G1 + E[:, :, 0, n - 2] @ C
        # MATLAB: for i = 2:(n-1)
        for i in range(2, n):
            # MATLAB: E(:,:,i,n) = E(:,:,i-1,n-1)*G1 + E(:,:,i,n-1)*C;
            E[:, :, i - 1, n - 1] = E[:, :, i - 2, n - 2] @ G1 + E[:, :, i - 1, n - 2] @ C
        # MATLAB: E(:,:,n,n) = E(:,:,n-1,n-1)*G1;
        E[:, :, n - 1, n - 1] = E[:, :, n - 2, n - 2] @ G1
        
        # Input
        # MATLAB: D_u(:,:,n) = D(:,:,n-1)*c_u;
        D_u[:, :, n - 1] = D[:, :, n - 2] @ c_u
        # MATLAB: E_u(:,:,1,n) = D(:,:,n-1)*g_u + E(:,:,1,n-1)*c_u;
        if g_u.size > 0:
            E_u[:, :, 0, n - 1] = D[:, :, n - 2] @ g_u + E[:, :, 0, n - 2] @ c_u
        # MATLAB: for i = 2:(n-1)
        for i in range(2, n):
            # MATLAB: E_u(:,:,i,n) = E(:,:,i-1,n-1)*g_u + E(:,:,i,n-1)*c_u;
            if g_u.size > 0:
                E_u[:, :, i - 1, n - 1] = E[:, :, i - 2, n - 2] @ g_u + E[:, :, i - 1, n - 2] @ c_u
        # MATLAB: E_u(:,:,n,n) = E(:,:,n-1,n-1)*g_u;
        if g_u.size > 0:
            E_u[:, :, n - 1, n - 1] = E[:, :, n - 2, n - 2] @ g_u
        
        # Update power cell
        # MATLAB: zPow{n} = matZonotope(D(:,:,n),E(:,:,1:n,n))*r^n;
        zPow[n - 1] = matZonotope(D[:, :, n - 1], E[:, :, :n, n - 1]) * (r ** n)
    
    # Compute exponential matrix
    # Preallocate
    # MATLAB: E_sum = nan([size(E,1:2),maxOrder]);
    # MATLAB: E_u_sum = nan([size(E_u,1:2),maxOrder]);
    E_sum = np.full((n_dims, m_dims, maxOrder), np.nan)
    # E_u_sum shape depends on whether g_u is empty
    if E_u.shape[1] == 0:  # g_u was empty
        # E_u has shape (n_u, 0, maxOrder, maxOrder), so E_u_sum should be (n_u, 0, maxOrder)
        # But we need to handle this case specially
        E_u_sum = np.full((E_u.shape[0], 0, maxOrder), np.nan)
    else:
        E_u_sum = np.full((*E_u.shape[:2], maxOrder), np.nan)
    
    # Generators
    # MATLAB: for n = 1:maxOrder
    for n in range(1, maxOrder + 1):
        # MATLAB: factor = r^n/factorial(n);
        factor = r ** n / math.factorial(n)
        # MATLAB: E_sum(:,:,n) = E(:,:,1,n)*factor;
        E_sum[:, :, n - 1] = E[:, :, 0, n - 1] * factor
        # MATLAB: E_u_sum(:,:,n) = E_u(:,:,1,n)*factor;
        if E_u_sum.shape[1] > 0:  # Only if g_u was not empty
            E_u_sum[:, :, n - 1] = E_u[:, :, 0, n - 1] * factor
        # MATLAB: for i=(n+1):maxOrder
        for i in range(n + 1, maxOrder + 1):
            # MATLAB: factor = r^i/factorial(i);
            factor = r ** i / math.factorial(i)
            # MATLAB: E_sum(:,:,n) = E_sum(:,:,n) + E(:,:,n,i)*factor;
            E_sum[:, :, n - 1] = E_sum[:, :, n - 1] + E[:, :, n - 1, i - 1] * factor
            # MATLAB: E_u_sum(:,:,n) = E_u_sum(:,:,n) + E_u(:,:,n,i)*factor;
            if E_u_sum.shape[1] > 0:  # Only if g_u was not empty
                E_u_sum[:, :, n - 1] = E_u_sum[:, :, n - 1] + E_u[:, :, n - 1, i - 1] * factor
    
    # Center
    # MATLAB: D_sum = eye(dim(matZ)) + D(:,:,1)*r;
    n_dim = dim(matZ)[0]
    D_sum = np.eye(n_dim) + D[:, :, 0] * r
    # MATLAB: D_u_sum = D_u(:,:,1)*r;
    D_u_sum = D_u[:, :, 0] * r
    # MATLAB: for i = 2:maxOrder
    for i in range(2, maxOrder + 1):
        # MATLAB: factor = r^i/factorial(i);
        factor = r ** i / math.factorial(i)
        # MATLAB: D_sum = D_sum + D(:,:,i)*factor;
        D_sum = D_sum + D[:, :, i - 1] * factor
        # MATLAB: D_u_sum = D_u_sum + D_u(:,:,i)*factor;
        D_u_sum = D_u_sum + D_u[:, :, i - 1] * factor
    
    # Reduce size of generators for even numbers and add to center
    # MATLAB: for n = 1:floor(maxOrder/2)
    for n in range(1, int(np.floor(maxOrder / 2)) + 1):
        # MATLAB: E_sum(:,:,2*n) = 0.5*E_sum(:,:,2*n);
        E_sum[:, :, 2 * n - 1] = 0.5 * E_sum[:, :, 2 * n - 1]
        # MATLAB: D_sum = D_sum + E_sum(:,:,2*n);
        D_sum = D_sum + E_sum[:, :, 2 * n - 1]
        
        # MATLAB: E_u_sum(:,:,2*n) = 0.5*E_u_sum(:,:,2*n);
        if E_u_sum.shape[1] > 0:  # Only if g_u was not empty
            E_u_sum[:, :, 2 * n - 1] = 0.5 * E_u_sum[:, :, 2 * n - 1]
        # MATLAB: D_u_sum = D_u_sum + E_u_sum(:,:,2*n);
        # In MATLAB, this works even if E_u_sum is empty (no-op)
        # In Python, we need to check if the slice is non-empty before adding
        if E_u_sum.shape[1] > 0:
            D_u_sum = D_u_sum + E_u_sum[:, :, 2 * n - 1]
    
    # Compute remainder
    # MATLAB: matI = intervalMatrix(matZ*r);
    matI = IntervalMatrix(matZ * r)
    # MATLAB: E = exponentialRemainder(matI,maxOrder);
    E = exponentialRemainder(matI, maxOrder)
    
    # Write result to eZ and eI
    # MATLAB: eZ = matZonotope(D_sum, E_sum);
    eZ = matZonotope(D_sum, E_sum)
    eI = E
    
    # Obtain constant input zonotope
    # MATLAB: RconstInput = zonotope(matZonotope(D_u_sum, E_u_sum));
    # MATLAB algorithm for zonotope(matZonotope):
    #   c = reshape(matZ.C,[],1);
    #   [m,n,h] = size(matZ.G);
    #   G = reshape(matZ.G,n*m,h);
    #   Z = zonotope([c,G]);
    # D_u_sum is (n_u, m_u) and E_u_sum is (n_u, m_u, maxOrder)
    c_u_flat = D_u_sum.reshape(-1, 1)  # Reshape to column vector
    if E_u_sum.size > 0:
        m_u, n_u, h_u = E_u_sum.shape
        G_u_flat = E_u_sum.reshape(n_u * m_u, h_u)  # Reshape to (n*m, h)
        RconstInput = Zonotope(c_u_flat, G_u_flat)
    else:
        # Empty generators case - need to specify dimension from center
        n_dims = c_u_flat.shape[0]
        RconstInput = Zonotope(c_u_flat, np.zeros((n_dims, 0)))
    
    # No powers based on interval matrix
    # MATLAB: iPow = [];
    iPow = []
    
    return eZ, eI, zPow, iPow, E, RconstInput
