"""
poly2bernstein - Convert a polynomial to a bernstein polynomial

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Niklas Kochdumper
         Python: AI Assistant
"""

import numpy as np
from typing import Tuple, List
from cora_python.contSet.interval.interval import Interval


def poly2bernstein(G: np.ndarray, E: np.ndarray, dom: Interval) -> List[np.ndarray]:
    """
    Convert a polynomial to a bernstein polynomial.
    
    Args:
        G: generator matrix containing the coefficients of the polynomial
        E: exponent matrix containing the exponents of the polynomial
        dom: interval that represents the domain for each variable
        
    Returns:
        B: coefficients of the bernstein polynomial
        
    See also: taylm/interval, polyZonotope/interval
    """
    
    # Handle edge cases first
    if E.size == 0 or G.size == 0:
        # Empty case - return empty array matching MATLAB behavior
        return [np.array([]).reshape(0, 0)]
    
    l = E.shape[0]  # number of variables
    n = np.max(E)   # maximum degree
    h = G.shape[0]  # number of dimensions
    
    # Handle special case: constant polynomial (n=0)
    if n == 0:
        # For constant polynomial, return the constant value
        if h == 1:
            return [np.array([[G[0, 0]]])]
        else:
            return [np.array([[G[i, 0]]]) for i in range(h)]
    
    # Preprocessing: compute coefficient matrix A
    len_coeff = (n + 1) ** (l - 1)
    A = [np.zeros((n + 1, len_coeff)) for _ in range(h)]
    
    for i in range(E.shape[1]):
        ind = _aux_getIndices(E[:, i], len_coeff, n)
        for j in range(h):
            A[j][ind[0], ind[1]] = G[j, i]
    
    # Step 1: Compute the binomial coefficients
    C = _aux_Binomial_coefficient(n)
    
    # Step 2: Compute inverse of U_x
    Ux = _aux_InverseUx(n, C)
    
    # Step 3: Iterate
    M = [None] * l
    infi = dom.inf
    sup = dom.sup
    
    for r in range(l):
        if r == 0 or infi[r] != infi[0] or sup[r] != sup[0]:
            # Compute inverse of V_x
            Vx = _aux_InverseVx(n, dom[r])
            
            # Compute inverse of W_x
            Wx = _aux_InverseWx(n, dom[r], C)
            
            # Product of all the inverse matrices
            M[r] = Ux @ Vx @ Wx
        else:
            M[r] = M[0]
    
    # Step 4: Iterate
    for j in range(h):
        for r in range(l):
            A[j] = _aux_transposeMatrix(A[j], r, len_coeff, l, n)
            A[j] = M[r] @ A[j]
            A[j] = _aux_transposeMatrix(A[j], r, len_coeff, l, n)
    
    # Return Bernstein coefficients
    return A


def _aux_getIndices(exp: np.ndarray, len_coeff: int, n: int) -> Tuple[int, int]:
    """Get indices for coefficient matrix - Python 0-based indexing version of MATLAB aux_getIndices"""
    # column index = exponent of first variable (0-based)
    ind = np.zeros(2, dtype=int)
    ind[0] = exp[0]  # Python uses 0-based indexing
    
    # row index -> loop over all remaining variables 
    len_ = len_coeff
    for i in range(len(exp) - 1, 0, -1):  # loop from last to second
        len_ = len_ // (n + 1)
        ind[1] = ind[1] + exp[i] * len_
    
    # No need to add 1 for Python 0-based indexing
    
    return ind[0], ind[1]


def _aux_Binomial_coefficient(n: int) -> np.ndarray:
    """Implementation of the "Binomial_coefficient" algorithm in [1] - Python 0-based indexing"""
    C = np.full((n + 1, n + 1), np.nan)
    C[1, 0] = 1  # C(2,1) in MATLAB = C[1,0] in Python
    
    for i in range(2, n + 1):  # i from 2 to n
        C[i, i] = 1
        if i < n:  # Only set C[i+1,0] if i+1 is within bounds
            C[i + 1, 0] = 1
        
        for k in range(1, i):  # k from 1 to (i-1)
            if i < n:  # Only set C[i+1,k] if i+1 is within bounds
                C[i + 1, k] = i / (i - k) * C[i, k]
    
    return C


def _aux_InverseUx(n: int, C: np.ndarray) -> np.ndarray:
    """Implementation of the "InverseUx" algorithm in [1] - Python 0-based indexing"""
    Ux = np.zeros((n + 1, n + 1))
    
    # Ux(1:end,1) = ones(n+1,1) -> Ux[:, 0] = 1
    Ux[:, 0] = 1
    # Ux(n+1,2:end) = ones(1,n) -> Ux[n, 1:] = 1
    Ux[n, 1:] = 1
    
    for i in range(1, n):  # i from 1 to n-1
        for j in range(1, i + 1):  # j from 1 to i
            Ux[i, j] = C[i, i - j] / C[n, j]  # Ux(i+1,j+1) = C(i+1,i-j+1)/C(n+1,j+1)
    
    return Ux


def _aux_InverseVx(n: int, dom_i: Interval) -> np.ndarray:
    """Implementation of the "InverseVx" algorithm in [1] - matches MATLAB exactly"""
    wid_x = 2 * (dom_i.sup - dom_i.inf) / 2  # 2*rad(x)
    
    Vx = np.zeros((n + 1, n + 1))
    Vx[0, 0] = 1
    
    wid_x_ = np.zeros(n + 1)
    wid_x_[0] = 1
    
    for i in range(1, n + 1):  # i from 1 to n
        wid_x_[i] = wid_x_[i - 1] * wid_x
        Vx[i, i] = wid_x_[i]
    
    return Vx


def _aux_InverseWx(n: int, dom_i: Interval, C: np.ndarray) -> np.ndarray:
    """Implementation of the "InverseWx" algorithm in [1] - matches MATLAB exactly"""
    # All the powers of the infimum
    inf_x = dom_i.inf
    
    inf_x_ = np.zeros(n + 1)
    inf_x_[0] = 1
    
    for i in range(1, n + 1):  # i from 1 to n
        inf_x_[i] = inf_x_[i - 1] * inf_x
    
    # Construction of the inverse matrix Wx
    Wx = np.zeros((n + 1, n + 1))
    Wx[0, 0] = 1
    
    for i in range(n):  # i from 0 to n-1
        for j in range(i + 1, n + 1):  # j from i+1 to n
            Wx[i, j] = C[j, i] * inf_x_[j - i]  # Wx(i+1,j+1) = C(j+1,i+1) * inf_x_(j-i+1)
        
        Wx[i + 1, i + 1] = 1
    
    return Wx


def _aux_transposeMatrix(A: np.ndarray, r: int, len_coeff: int, l: int, n: int) -> np.ndarray:
    """Implementation of matrix transposition as required for the evaluation of equation (63) in [1] - Python 0-based indexing"""
    if r == 1:  # r == 2 in MATLAB (1-based) -> r == 1 in Python (0-based)
        # A = reshape(permute(reshape(A,[n+1,n+1,len/(n+1)]),[2,1,3]),[n+1,len]);
        A_reshaped = A.reshape(n + 1, n + 1, len_coeff // (n + 1))
        A_permuted = np.transpose(A_reshaped, (1, 0, 2))
        A = A_permuted.reshape(n + 1, len_coeff)
    elif r != 0:  # r ~= 1 in MATLAB (1-based) -> r != 0 in Python (0-based)
        if r == l - 1:  # r == l in MATLAB (1-based) -> r == l-1 in Python (0-based)
            A = _aux_transpose_(A)
        else:
            len_ = (n + 1) ** r  # r-1 in MATLAB -> r in Python
            # Split matrix into chunks and transpose each
            chunks = []
            for i in range(0, len_coeff, len_):
                chunk = A[:, i:i + len_]
                chunks.append(_aux_transpose_(chunk))
            A = np.hstack(chunks)
    
    return A


def _aux_transpose_(A: np.ndarray) -> np.ndarray:
    """Helper function for matrix transposition - matches MATLAB aux_transpose_ exactly"""
    n = A.shape[0]
    p = A.shape[1] // n
    
    # temp = reshape(permute(reshape(A,[n,p,n]),[2,1,3]),[1,n*p,n]);
    A_reshaped = A.reshape(n, p, n)
    A_permuted = np.transpose(A_reshaped, (1, 0, 2))
    temp = A_permuted.reshape(1, n * p, n)
    
    # A = permute(temp,[3,2,1]);
    A = np.transpose(temp, (2, 1, 0))
    
    # The result should be 2D, not 3D - reshape back to 2D
    return A.reshape(A.shape[0], A.shape[1])
