"""
convHull_ - computes an enclosure for the convex hull of a set and another set or a point (internal use)

This function provides the internal implementation for convex hull computation.
It should be overridden in subclasses to provide specific convex hull logic.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 12-September-2023 (MATLAB)
Last update: 30-September-2024 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING, Optional
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet


def convHull_(S: 'ContSet', S2: Optional['ContSet'] = None, method: str = 'exact'):
    """
    Computes the convex hull of a set or two sets (internal use)
    
    This function uses polymorphic dispatch to call the appropriate subclass
    implementation of convHull_, or provides the base implementation.
    
    Args:
        S: contSet object
        S2: optional second contSet object
        method: method for convex hull computation
        
    Returns:
        ContSet: Convex hull of the set(s)
        
    Raises:
        CORAerror: If convHull_ is not implemented for the specific set type
    """
    # Check if subclass has overridden convHull_ method
    base_class = type(S).__bases__[0] if type(S).__bases__ else None
    if (hasattr(type(S), 'convHull_') and 
        base_class and hasattr(base_class, 'convHull_') and
        type(S).convHull_ is not base_class.convHull_):
        return type(S).convHull_(S, S2, method)
    else:
        # Base implementation - throw error as this method should be overridden
        raise CORAerror("CORA:noops", f"Function convHull_ not implemented for class {type(S).__name__}") 