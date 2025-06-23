"""
isempty - (DEPRECATED -> representsa(S,'emptySet'))

This function checks if a contSet object represents an empty set.
It is deprecated and users should use representsa(S, 'emptySet') instead.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 19-July-2023 (MATLAB)
Python translation: 2025
"""

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet


def isempty(S: 'ContSet') -> bool:
    """
    Checks if a contSet object represents an empty set (DEPRECATED)
    
    This function is deprecated. Use representsa(S, 'emptySet') instead.
    
    Args:
        S: contSet object to check
        
    Returns:
        bool: True if the set is empty, False otherwise
        
    Warning:
        This function is deprecated. Use representsa(S, 'emptySet') instead.
        
    Example:
        >>> S = emptySet(2)
        >>> result = isempty(S)  # Deprecated - use representsa(S, 'emptySet')
        >>> # result is True
    """
    # Issue deprecation warning
    warnings.warn(
        "contSet/isempty is deprecated. "
        "When updating the code, please replace every function call 'isempty(S)' "
        "with 'representsa(S, \"emptySet\")'. "
        "The main reason is that the function 'isempty' is also called implicitly "
        "by Python in various circumstances. For contSet classes with the set "
        "property 'emptySet', this would return different results depending on "
        "running or debugging.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return S.representsa_('emptySet', 1e-15)  # eps equivalent 