"""
This module contains the function for checking if an ellipsoid is full-dimensional.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


def isFullDim(E: 'Ellipsoid') -> bool:
    """
    Checks if the dimension of the affine hull of an ellipsoid is
    equal to the dimension of its ambient space.
    
    Args:
        E: ellipsoid object
        
    Returns:
        res: true/false
    """
    if E.representsa_('emptySet', 1e-15):
        return False
    else:
        return E.rank() == E.dim() 