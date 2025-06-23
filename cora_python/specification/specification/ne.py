"""
ne - overloads '!=' operator for comparison of specification objects

Syntax:
    res = spec1 != spec2
    res = ne(spec1, spec2)
    res = ne(spec1, spec2, tol)

Inputs:
    spec1 - specification object
    spec2 - specification object
    tol - (optional) tolerance

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       29-April-2023 (MATLAB)
Last update:   ---
Last revision: ---
Python translation: 2025
"""

from typing import Optional


def ne(spec1, spec2, tol: Optional[float] = None) -> bool:
    """
    Overloaded '!=' operator for comparison of specification objects
    
    Args:
        spec1: First specification object
        spec2: Second specification object
        tol: Optional tolerance for comparison
        
    Returns:
        bool: True if specifications are not equal, False otherwise
    """
    
    # Import here to avoid circular imports
    
    return not spec1.isequal(spec2, tol) 