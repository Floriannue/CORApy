"""
compact - removes redundancies in the representation of a set

This function removes redundancies in the representation of a set, resulting
in a set that is equal to the original set up to a given tolerance, but
minimal in its representation.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 29-July-2023 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING, Optional
import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def compact(S: 'ContSet', method: Optional[str] = None, tol: Optional[float] = None) -> 'ContSet':
    """
    Removes redundancies in the representation of a set
    
    The resulting set is equal to the original set up to a given tolerance,
    but minimal in its representation.
    
    Args:
        S: contSet object
        method: Method for redundancy removal (depends on set type)
        tol: Tolerance for redundancy removal
        
    Returns:
        ContSet: Compacted set
        
    Raises:
        CORAerror: If method not implemented for set type
        ValueError: If invalid method or tolerance
        
    Example:
        >>> S = zonotope(center, generators)
        >>> S_compact = compact(S, 'zeros', 1e-10)
    """
    # Classes that are always in their minimal representation
    minimal_classes = ['Capsule', 'Ellipsoid', 'EmptySet', 'Fullspace', 
                      'Halfspace', 'Interval', 'ZonoBundle', 'SpectraShadow', 'Taylm']
    
    class_name = type(S).__name__
    if class_name in minimal_classes:
        return S
    
    # Set default values based on set type
    if method is None or tol is None:
        if class_name == 'Zonotope':
            method = method or 'zeros'
            tol = tol or np.finfo(float).eps
            valid_methods = ['all', 'zeros', 'aligned']
            # Reset tolerance for 'aligned' method
            if method == 'aligned' and tol == np.finfo(float).eps:
                tol = 1e-3
                
        elif class_name == 'Polytope':
            method = method or 'all'
            tol = tol or 1e-9
            valid_methods = ['all', 'zeros', 'A', 'Ae', 'aligned', 'V', 'AtoAe']
            
        elif class_name == 'ConZonotope':
            method = method or 'all'
            tol = tol or np.finfo(float).eps
            valid_methods = ['all', 'zeros']
            
        elif class_name == 'PolyZonotope':
            method = method or 'all'
            tol = tol or np.finfo(float).eps
            valid_methods = ['all', 'states', 'exponentMatrix']
            
        elif class_name == 'ConPolyZono':
            method = method or 'all'
            tol = tol or np.finfo(float).eps
            valid_methods = ['all', 'states', 'constraints', 'exponentMatrix']
            
        elif class_name == 'LevelSet':
            method = method or 'all'
            tol = tol or np.finfo(float).eps
            valid_methods = ['all']
            
        elif class_name == 'Polygon':
            method = method or 'all'
            tol = tol or 0.01
            valid_methods = ['all', 'simplify', 'douglasPeucker']
            
        elif class_name == 'SpectraShadow':
            method = method or 'all'
            tol = tol or np.finfo(float).eps
            valid_methods = ['all', 'zeros']
            
        else:
            # Default values for unknown types
            method = method or 'all'
            tol = tol or np.finfo(float).eps
            valid_methods = None
    
    # Validate method if we know the valid methods
    if 'valid_methods' in locals() and valid_methods is not None:
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}' for {class_name}. Use one of {valid_methods}")
    
    # Validate tolerance
    if not isinstance(tol, (int, float)) or tol < 0:
        raise ValueError("tol must be a non-negative number")
    
    try:
        # Call subclass method
        return S.compact_(method, tol)
    except Exception as ME:
        if hasattr(ME, 'identifier') and ME.identifier == '':
            raise CORAerror('CORA:noops', f'compact not implemented for {class_name}')
        else:
            raise ME 