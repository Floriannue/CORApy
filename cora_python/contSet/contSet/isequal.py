"""
isequal - checks if two sets are equal

This function provides the core equality comparison for contSet objects.
It is meant to be overridden in subclasses that implement specific equality logic.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 12-September-2023 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING, Any, Optional
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def isequal(S1: 'ContSet', S2: Any, tol: Optional[float] = None, *args, **kwargs) -> bool:
    """
    Checks if two sets are equal
    
    This implementation uses polymorphic dispatch to call the appropriate
    isequal function based on the type of S1.
    
    Args:
        S1: First contSet object
        S2: Second object to compare with
        tol: Optional tolerance for comparison
        *args: Additional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        bool: True if objects are equal, False otherwise
        
    Example:
        >>> S1 = interval([1, 2], [3, 4])
        >>> S2 = interval([1, 2], [3, 4])
        >>> result = isequal(S1, S2)
    """
    # Check if objects are the same instance
    if S1 is S2:
        return True
    
    # Check if they're the same type
    if type(S1) != type(S2):
        return False
    
    # For contSet objects, try to find a specific isequal implementation
    # Look for class-specific isequal function (e.g., interval/isequal.py)
    module_name = type(S1).__name__.lower()
    try:
        # Try to import the specific isequal function
        import importlib
        isequal_module = importlib.import_module(f'cora_python.contSet.{module_name}.isequal')
        if hasattr(isequal_module, 'isequal'):
            return isequal_module.isequal(S1, S2, tol, *args, **kwargs)
    except (ImportError, AttributeError):
        pass
    
    # Fallback - basic comparison for simple cases
    # For objects with same attributes, compare them
    if hasattr(S1, '__dict__') and hasattr(S2, '__dict__'):
        try:
            import numpy as np
            s1_dict = S1.__dict__.copy()
            s2_dict = S2.__dict__.copy()
            
            # Compare each attribute
            if set(s1_dict.keys()) != set(s2_dict.keys()):
                return False
                
            for key in s1_dict.keys():
                v1, v2 = s1_dict[key], s2_dict[key]
                if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                    if tol is not None:
                        if not np.allclose(v1, v2, atol=tol, rtol=tol):
                            return False
                    else:
                        if not np.array_equal(v1, v2):
                            return False
                elif v1 != v2:
                    return False
            return True
        except:
            pass
    
    # Final fallback - throw error if not implemented
    raise CORAerror('CORA:noops', 
                   f'isequal not implemented for {type(S1).__name__} and {type(S2).__name__}') 