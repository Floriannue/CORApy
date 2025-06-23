"""
isequal - check if two specifications are equal

Syntax:
    res = isequal(spec1, spec2)
    res = isequal(spec1, spec2, tol)

Inputs:
    spec1 - specification object
    spec2 - specification object  
    tol - tolerance (optional)

Outputs:
    res - true/false

Example:
    spec1 = specification(polytope([-1;1],[1;1]),'safeSet');
    spec2 = specification(polytope([-1;1],[1;1]),'safeSet');
    res = isequal(spec1, spec2)

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 30-April-2023 (MATLAB)
Last update: --- (MATLAB)
Python translation: 2025
"""

from typing import Optional
from .specification import Specification


def isequal(spec1, spec2, tol: Optional[float] = None) -> bool:
    """
    Check if two specification objects are equal
    
    Args:
        spec1: First specification object
        spec2: Second specification object
        tol: Tolerance for comparison (optional)
        
    Returns:
        bool: True if specifications are equal, False otherwise
    """
    
    # Import here to avoid circular imports
    
    # Check if both are specification objects
    if not isinstance(spec1, Specification) or not isinstance(spec2, Specification):
        return False
    
    # Check type equality
    if spec1.type != spec2.type:
        return False
    
    # Check set equality with tolerance if provided
    try:
        if hasattr(spec1.set, 'isequal') and callable(getattr(spec1.set, 'isequal')):
            if tol is not None:
                sets_equal = spec1.set.isequal(spec2.set, tol)
            else:
                sets_equal = spec1.set.isequal(spec2.set)
        elif hasattr(spec1.set, '__eq__'):
            sets_equal = spec1.set == spec2.set
        else:
            # Fallback: assume not equal if we can't check
            sets_equal = False
    except:
        sets_equal = False
    
    if not sets_equal:
        return False
    
    # Check time equality
    if spec1.time is None and spec2.time is None:
        return True
    elif spec1.time is None or spec2.time is None:
        return False
    else:
        # Check time interval equality with tolerance
        try:
            if hasattr(spec1.time, 'isequal') and callable(getattr(spec1.time, 'isequal')):
                if tol is not None:
                    return spec1.time.isequal(spec2.time, tol)
                else:
                    return spec1.time.isequal(spec2.time)
            elif hasattr(spec1.time, '__eq__'):
                return spec1.time == spec2.time
            else:
                return False
        except:
            return False 