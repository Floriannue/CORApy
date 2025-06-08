"""
generateRandom - Generates a random linear system

This function generates a random linear system of the form:
    x' = Ax + Bu
    y  = Cx

Authors: Florian Nüssel (Python implementation)
Date: 2025-06-08
"""

import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .linearSys import LinearSys


def generateRandom(state_dimension: Optional[int] = None,
                  input_dimension: Optional[int] = None,
                  output_dimension: Optional[int] = None,
                  real_interval: Optional[tuple] = None,
                  imaginary_interval: Optional[tuple] = None) -> 'LinearSys':
    """
    Generates a random linear system
    
    Args:
        state_dimension: State dimension (default: random 4-10)
        input_dimension: Input dimension (default: random 1-3)
        output_dimension: Output dimension (default: random 1-2)
        real_interval: Interval for real part of eigenvalues (default: (-1-10*rand, -rand))
        imaginary_interval: Interval for imaginary part of eigenvalues (default: (-10*rand, 10*rand))
        
    Returns:
        LinearSys: Random linear system
        
    Example:
        linsys1 = LinearSys.generateRandom()
        linsys2 = LinearSys.generateRandom(state_dimension=3)
        linsys3 = LinearSys.generateRandom(state_dimension=5, 
                                         real_interval=(-5, -1),
                                         imaginary_interval=(-1, 1))
    """
    
    # Set default values if not provided
    if state_dimension is None:
        state_dimension = np.random.randint(4, 11)
    
    if input_dimension is None:
        input_dimension = np.random.randint(1, 4)
    
    if output_dimension is None:
        output_dimension = np.random.randint(1, 3)
    
    if real_interval is None:
        real_interval = (-1 - 10*np.random.rand(), -np.random.rand())
    
    if imaginary_interval is None:
        imag_max = 10*np.random.rand()
        imaginary_interval = (-imag_max, imag_max)
    
    # Validate inputs
    if state_dimension <= 0:
        raise ValueError("State dimension must be positive")
    if input_dimension <= 0:
        raise ValueError("Input dimension must be positive")
    if output_dimension <= 0:
        raise ValueError("Output dimension must be positive")
    
    n = state_dimension
    
    # Compute value with minimum distance to zero (imaginary parts are always complex conjugate)
    imag_max = min(abs(imaginary_interval[0]), abs(imaginary_interval[1]))
    
    # Determine number of complex (maximum floor(n/2)) and real eigenvalues
    n_conj_max = n // 2
    n_conj = np.random.randint(0, n_conj_max + 1)
    n_real = n - 2 * n_conj
    
    # Generate real eigenvalues
    vals_real = np.random.uniform(real_interval[0], real_interval[1], n_real)
    
    # Generate real parts for complex eigenvalues
    val_imag_real = np.random.uniform(real_interval[0], real_interval[1], n_conj)
    
    # Generate imaginary parts for complex eigenvalues
    vals_imag = np.random.uniform(0, imag_max, n_conj)
    
    # Build Jordan form matrix
    J_real = np.diag(vals_real) if n_real > 0 else np.empty((0, 0))
    
    # Build complex conjugate blocks
    J_conj_blocks = []
    for i in range(n_conj):
        block = np.array([[val_imag_real[i], -vals_imag[i]],
                         [vals_imag[i], val_imag_real[i]]])
        J_conj_blocks.append(block)
    
    # Combine all blocks
    if n_real > 0 and n_conj > 0:
        J_conj = np.block([[J_conj_blocks[0]]])
        for i in range(1, n_conj):
            J_conj = np.block([[J_conj, np.zeros((J_conj.shape[0], 2))],
                              [np.zeros((2, J_conj.shape[1])), J_conj_blocks[i]]])
        J = np.block([[J_real, np.zeros((n_real, 2*n_conj))],
                     [np.zeros((2*n_conj, n_real)), J_conj]])
    elif n_real > 0:
        J = J_real
    elif n_conj > 0:
        J = J_conj_blocks[0]
        for i in range(1, n_conj):
            J = np.block([[J, np.zeros((J.shape[0], 2))],
                         [np.zeros((2, J.shape[1])), J_conj_blocks[i]]])
    else:
        J = np.zeros((n, n))
    
    # Generate random transformation matrix
    P = np.random.randn(n, n)
    
    # Ensure P is invertible
    while np.linalg.cond(P) > 1e12:
        P = np.random.randn(n, n)
    
    # Compute A = P * J * P^(-1)
    A = P @ J @ np.linalg.inv(P)
    
    # Generate input matrix
    B = np.random.randn(n, input_dimension)
    
    # Generate output matrix
    C = np.random.randn(output_dimension, n)
    
    # Import here to avoid circular imports
    from .linearSys import LinearSys
    
    # Create and return linear system
    return LinearSys(A, B, None, C) 