"""
display - Displays the properties of a spectraShadow object (center
    vector, generator matrix, and coefficient matrix) on the command
    window

Syntax:
    display(SpS)

Inputs:
    SpS - spectraShadow object

Outputs:
    str_repr - string representation

Example: 
    SpS = SpectraShadow(np.array([[1, 0, 1, 0], [0, 1, 0, 1]]))
    display_str = SpS.display()

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Adrian Kulmburg (MATLAB)
               Python translation by AI Assistant
Written:       01-August-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .spectraShadow import SpectraShadow


def display(SpS: 'SpectraShadow') -> str:
    """
    Displays the properties of a spectraShadow object
    
    Args:
        SpS: spectraShadow object
        
    Returns:
        str_repr: string representation
    """
    
    if SpS.isemptyobject():
        return f"SpectraShadow (empty set in R^{SpS.dim()})"
    
    # Build the string representation
    lines = []
    lines.append(f"SpectraShadow in R^{SpS.dim()}")
    lines.append("")
    
    # Display center
    lines.append("c:")
    center_str = _format_matrix(SpS.c)
    lines.append(center_str)
    lines.append("")
    
    # Display generators
    lines.append("G:")
    generators_str = _format_matrix(SpS.G, name="G")
    lines.append(generators_str)
    lines.append("")
    
    # Display coefficient matrix
    lines.append("A:")
    coeff_str = _format_matrix(SpS.A, name="A")
    lines.append(coeff_str)
    lines.append("")
    
    # Display properties
    lines.append(f"Bounded?              {_prop_to_string(SpS.bounded)}")
    lines.append(f"Empty set?            {_prop_to_string(SpS.emptySet)}")
    lines.append(f"Full-dimensional set? {_prop_to_string(SpS.fullDim)}")
    
    return "\n".join(lines)


def _format_matrix(matrix: np.ndarray, name: str = None, max_display_size: int = 10) -> str:
    """
    Format a matrix for display with optional truncation for large matrices
    
    Args:
        matrix: numpy array to format
        name: optional name for the matrix
        max_display_size: maximum size to display before truncating
        
    Returns:
        formatted string representation of the matrix
    """
    if matrix.size == 0:
        return "[]"
    
    # Convert to dense if sparse
    if hasattr(matrix, 'toarray'):
        matrix = matrix.toarray()
    
    # Check if matrix is too large to display
    if matrix.shape[0] > max_display_size or matrix.shape[1] > max_display_size:
        if name:
            return f"{name} ∈ R^{matrix.shape[0]}×{matrix.shape[1]} (too large to display)"
        else:
            return f"Matrix ∈ R^{matrix.shape[0]}×{matrix.shape[1]} (too large to display)"
    
    # Format the matrix with proper alignment
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    
    # Convert to string with proper formatting
    with np.printoptions(precision=4, suppress=True, linewidth=120):
        matrix_str = str(matrix)
    
    # Add proper indentation
    lines = matrix_str.split('\n')
    indented_lines = ['  ' + line for line in lines]
    
    return '\n'.join(indented_lines)


def _prop_to_string(prop) -> str:
    """
    Convert a property object to string representation
    
    Args:
        prop: property object with 'val' attribute
        
    Returns:
        string representation of the property
    """
    if not hasattr(prop, 'val') or prop.val is None:
        return 'Unknown'
    elif prop.val:
        return 'true'
    elif not prop.val:
        return 'false'
    else:
        return 'Unknown' 