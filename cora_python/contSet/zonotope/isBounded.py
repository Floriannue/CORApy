"""
isBounded - determines if a zonotope is bounded

Syntax:
    res = isBounded(Z)

Inputs:
    Z - zonotope object

Outputs:
    res - true/false

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 24-July-2023 (MATLAB)
Last update: ---
Python translation: 2025
"""


def isBounded(Z: 'Zonotope') -> bool:
    """
    Determines if a zonotope is bounded
    
    Args:
        Z: Zonotope object
        
    Returns:
        bool: Always True since zonotopes are always bounded
        
    Example:
        >>> Z = Zonotope([1, 0], [[1, 0], [0, 1]])
        >>> result = isBounded(Z)
        >>> # result is True
    """
    # Zonotopes are always bounded
    return True 