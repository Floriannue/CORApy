"""
eq - overloaded '==' operator for specifications

Syntax:
    res = eq(spec1, spec2)

Inputs:
    spec1 - specification object
    spec2 - specification object

Outputs:
    res - true/false

Example:
    spec1 = specification(polytope([-1;1],[1;1]),'safeSet');
    spec2 = specification(polytope([-1;1],[1;1]),'safeSet');
    res = eq(spec1, spec2)

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant  
Written: 30-April-2023 (MATLAB)
Last update: --- (MATLAB)
Python translation: 2025
"""

from typing import Union, List
from .specification import Specification

def eq(spec1: Specification, spec2: Specification) -> bool:
    """
    Check equality of two specification objects
    
    Args:
        spec1: First specification object
        spec2: Second specification object
        
    Returns:
        bool: True if specifications are equal, False otherwise
    """
    
    # Check if both are specification objects
    if not isinstance(spec1, Specification) or not isinstance(spec2, Specification):
        return False
    
    # Check type equality
    if spec1.type != spec2.type:
        return False
    
    # Check set equality
    try:
        if hasattr(spec1.set, 'isequal') and callable(getattr(spec1.set, 'isequal')):
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
        # Check time interval equality
        try:
            if hasattr(spec1.time, 'isequal') and callable(getattr(spec1.time, 'isequal')):
                return spec1.time.isequal(spec2.time)
            elif hasattr(spec1.time, '__eq__'):
                return spec1.time == spec2.time
            else:
                return False
        except:
            return False 