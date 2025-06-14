"""
contains_ - check if an interval contains given points or sets

Syntax:
    res = contains_(I, S, tol)

Inputs:
    I - interval object
    S - points to check (numpy array) or interval object
    tol - tolerance for containment check

Outputs:
    res - true if all points/sets are contained, false otherwise

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 19-June-2015 (MATLAB)
Python translation: 2025
"""

import numpy as np


def contains_(obj, S, tol: float = 0) -> bool:
    """
    Check if interval contains given point(s) or interval(s)
    
    Args:
        obj: Interval object
        S: Point(s) to check (numpy array) or interval object
        tol: Tolerance for containment check
        
    Returns:
        True if all points/sets are contained, False otherwise
    """
    # Handle empty set
    if obj.inf.size == 0:
        # Empty interval only contains empty sets
        if hasattr(S, 'representsa_'):
            return S.representsa_('emptySet')
        else:
            return False
    
    # Point(s) in interval containment
    if isinstance(S, (int, float, list, tuple, np.ndarray)):
        S = np.asarray(S)
        
        # Handle multiple points (each column is a point)
        if S.ndim == 1:
            S = S.reshape(-1, 1)
        
        # Check dimension compatibility
        expected_dim = obj.inf.shape[0] if obj.inf.ndim > 0 else 0
        if S.shape[0] != expected_dim:
            raise ValueError(f"Dimension mismatch: interval has dimension {expected_dim}, "
                           f"but point has dimension {S.shape[0]}")
        
        # Check containment for each point
        results = []
        for i in range(S.shape[1]):
            point = S[:, i:i+1]  # Keep as column vector for consistent shape
            
            # Ensure point and bounds have compatible shapes for comparison
            # Flatten both to 1D for comparison if bounds are 2D
            if obj.inf.ndim > 1:
                point_flat = point.flatten()
                inf_flat = obj.inf.flatten()
                sup_flat = obj.sup.flatten()
            else:
                point_flat = point.flatten()
                inf_flat = obj.inf
                sup_flat = obj.sup
            
            # Check if point is within bounds (with tolerance)
            contained = (np.all(point_flat >= inf_flat - tol) and 
                        np.all(point_flat <= sup_flat + tol))
            results.append(contained)
        
        # Return single boolean if single point, array if multiple points
        if len(results) == 1:
            return results[0]
        else:
            return np.array(results)
    
    # Interval in interval containment
    elif hasattr(S, 'inf') and hasattr(S, 'sup'):
        # Check if S is empty
        if hasattr(S, 'representsa_') and S.representsa_('emptySet'):
            return True
        
        # Check if S is contained in obj
        # S is contained in obj if obj.inf <= S.inf and S.sup <= obj.sup
        inf_contained = np.all(obj.inf <= S.inf + tol)
        sup_contained = np.all(S.sup <= obj.sup + tol)
        
        return inf_contained and sup_contained
    
    else:
        # Unsupported type
        raise TypeError(f"Unsupported type for containment check: {type(S)}") 
