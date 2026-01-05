"""
display - Displays a linearSys object

This function provides a formatted display of the linear system's matrices
and properties, similar to the MATLAB implementation.

Authors: Florian Nüssel (Python implementation)
Date: 2025-06-08
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .linearSys import LinearSys


def display_(linsys: 'LinearSys') -> str:
    """
    Displays a linearSys object (internal function that returns string)
    
    Args:
        linsys: LinearSys object to display
        
    Returns:
        str: Formatted string representation
        
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
    lines = []
    
    # Display basic system information
    lines.append(f"\nLinear System: {linsys.name}")
    lines.append(f"States: {linsys.nr_of_dims}")
    lines.append(f"Inputs: {linsys.nr_of_inputs}")
    lines.append(f"Outputs: {linsys.nr_of_outputs}")
    lines.append(f"Disturbances: {linsys.nr_of_disturbances}")
    lines.append(f"Noises: {linsys.nr_of_noises}")
    
    lines.append("\nType: Linear continuous-time time-invariant system")
    
    # State equation
    lines.append("\nState-space equation: x' = Ax + Bu + c + Ew")
    
    # Display state matrix
    lines.append("\nSystem matrix:")
    lines.extend(_display_matrix_vector(linsys.A, "A"))
    
    # Display input matrix
    lines.append("\nInput matrix:")
    lines.extend(_display_matrix_vector(linsys.B, "B"))
    
    # Display constant offset
    lines.append("\nConstant offset:")
    lines.extend(_display_matrix_vector(linsys.c, "c"))
    
    # Display disturbance matrix
    lines.append("\nDisturbance matrix:")
    lines.extend(_display_matrix_vector(linsys.E, "E"))
    
    # Check if there is a non-trivial output equation
    is_output = (not np.isscalar(linsys.C) or linsys.C != 1 or 
                np.any(linsys.D) or np.any(linsys.k) or np.any(linsys.F))
    
    # Output equation
    if not is_output:
        lines.append("\nOutput equation: y = x")
    else:
        lines.append("\nOutput equation: y = Cx + Du + k + Fv")
        
        # Display output matrix
        lines.append("\nOutput matrix:")
        lines.extend(_display_matrix_vector(linsys.C, "C"))
        
        # Display feedthrough matrix
        lines.append("\nFeedthrough matrix:")
        lines.extend(_display_matrix_vector(linsys.D, "D"))
        
        # Display constant offset
        lines.append("\nConstant offset:")
        lines.extend(_display_matrix_vector(linsys.k, "k"))
        
        # Display noise matrix
        lines.append("\nNoise matrix:")
        lines.extend(_display_matrix_vector(linsys.F, "F"))
    
    return '\n'.join(lines)


def display(linsys: 'LinearSys') -> None:
    """
    Displays a linearSys object (prints to stdout)
    
    Args:
        linsys: LinearSys object to display
    """
    print(display_(linsys), end='')


def _display_matrix_vector(matrix: np.ndarray, name: str) -> list:
    """
    Helper function to display a matrix or vector with proper formatting
    
    Args:
        matrix: NumPy array to display (can be None)
        name: Name of the matrix/vector
        
    Returns:
        list: List of strings representing the formatted matrix/vector
    """
    lines = []
    if matrix is None or matrix.size == 0:
        lines.append(f"{name} = []")
        return lines
    
    # Handle scalar case
    if np.isscalar(matrix) or matrix.size == 1:
        lines.append(f"{name} = {matrix}")
        return lines
    
    # Format matrix display
    lines.append(f"{name} =")
    
    # Set print options for better formatting
    with np.printoptions(precision=4, suppress=True, linewidth=100):
        # Add indentation for matrix rows
        matrix_str = str(matrix)
        matrix_lines = matrix_str.split('\n')
        for line in matrix_lines:
            lines.append(f"    {line}")
    
    return lines 