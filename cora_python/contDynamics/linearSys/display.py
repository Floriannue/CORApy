"""
display - Displays a linearSys object

This function provides a formatted display of the linear system's matrices
and properties, similar to the MATLAB implementation.

Authors: Florian Lercher (Python implementation)
Date: 2025-06-08
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .linearSys import LinearSys


def display(linsys: 'LinearSys') -> None:
    """
    Displays a linearSys object with formatted output
    
    Args:
        linsys: LinearSys object to display
        
    Example:
        A = [[-0.3780, 0.2839, 0.5403, -0.2962],
             [0.1362, 0.2742, 0.5195, 0.8266],
             [0.0502, -0.1051, -0.6572, 0.3874],
             [1.0227, -0.4877, 0.8342, -0.2372]]
        B = 0.25 * [[-2, 0, 3], [2, 1, 0], [0, 0, 1], [0, -2, 1]]
        c = 0.05 * [[-4], [2], [3], [1]]
        C = [[1, 1, 0, 0], [0, -0.5, 0.5, 0]]
        D = [[0, 0, 1], [0, 0, 0]]
        k = [[0], [0.02]]
        linsys = LinearSys(A, B, c, C, D, k)
        display(linsys)
    """
    
    # Display basic system information
    print(f"\nLinear System: {linsys.name}")
    print(f"States: {linsys.nr_of_dims}")
    print(f"Inputs: {linsys.nr_of_inputs}")
    print(f"Outputs: {linsys.nr_of_outputs}")
    print(f"Disturbances: {linsys.nr_of_disturbances}")
    print(f"Noises: {linsys.nr_of_noises}")
    
    print("\nType: Linear continuous-time time-invariant system")
    
    # State equation
    print("\nState-space equation: x' = Ax + Bu + c + Ew")
    
    # Display state matrix
    print("\nSystem matrix:")
    _display_matrix_vector(linsys.A, "A")
    
    # Display input matrix
    print("\nInput matrix:")
    _display_matrix_vector(linsys.B, "B")
    
    # Display constant offset
    print("\nConstant offset:")
    _display_matrix_vector(linsys.c, "c")
    
    # Display disturbance matrix
    print("\nDisturbance matrix:")
    _display_matrix_vector(linsys.E, "E")
    
    # Check if there is a non-trivial output equation
    is_output = (not np.isscalar(linsys.C) or linsys.C != 1 or 
                np.any(linsys.D) or np.any(linsys.k) or np.any(linsys.F))
    
    # Output equation
    if not is_output:
        print("\nOutput equation: y = x")
    else:
        print("\nOutput equation: y = Cx + Du + k + Fv")
        
        # Display output matrix
        print("\nOutput matrix:")
        _display_matrix_vector(linsys.C, "C")
        
        # Display feedthrough matrix
        print("\nFeedthrough matrix:")
        _display_matrix_vector(linsys.D, "D")
        
        # Display constant offset
        print("\nConstant offset:")
        _display_matrix_vector(linsys.k, "k")
        
        # Display noise matrix
        print("\nNoise matrix:")
        _display_matrix_vector(linsys.F, "F")


def _display_matrix_vector(matrix: np.ndarray, name: str) -> None:
    """
    Helper function to display a matrix or vector with proper formatting
    
    Args:
        matrix: NumPy array to display
        name: Name of the matrix/vector
    """
    if matrix.size == 0:
        print(f"{name} = []")
        return
    
    # Handle scalar case
    if np.isscalar(matrix) or matrix.size == 1:
        print(f"{name} = {matrix}")
        return
    
    # Format matrix display
    print(f"{name} =")
    
    # Set print options for better formatting
    with np.printoptions(precision=4, suppress=True, linewidth=100):
        # Add indentation for matrix rows
        matrix_str = str(matrix)
        lines = matrix_str.split('\n')
        for line in lines:
            print(f"    {line}") 