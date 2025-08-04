"""
This module contains the function for checking if an ellipsoid is full-dimensional.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


def isFullDim(E: 'Ellipsoid') -> bool:
    """
    isFullDim - checks if the dimension of the affine hull of an ellipsoid is
    equal to the dimension of its ambient space

    Syntax:
        res = isFullDim(E)

    Inputs:
        E - ellipsoid object

    Outputs:
        res - true/false

    Example:
        E1 = Ellipsoid(np.eye(2));
        isFullDim(E1)

        E2 = Ellipsoid(np.array([[1, 0], [0, 0]]));
        isFullDim(E2)

    Authors:       Niklas Kochdicker, Mark Wetzlinger
    Written:       02-January-2020 
    Last update:   24-March-2022 (remove dependency on object property)
                   06-July-2022 (VG, support class array case)
    Last revision: ---
                   Automatic python translation: Florian Nüssel BA 2025
    """
    if E.representsa_('emptySet', 1e-15):
        return False
    else:
        return E.rank() == E.dim() 