"""
center - returns the center of an ellipsoid object

Syntax:
   c = center(E)

Inputs:
   E - ellipsoid object

Outputs:
   c - center of ellipsoid

Example:
   E = ellipsoid([1 0; 0 1]);
   c = center(E);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Victor Gassmann
Written:       13-March-2019
Last update:   04-July-2022 (VG, avoid class array problems)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ellipsoid import Ellipsoid


def center(E: 'Ellipsoid') -> np.ndarray:
    """
    Returns the center of an ellipsoid object
    
    Args:
        E: ellipsoid object
        
    Returns:
        c: center of ellipsoid
    """
    return E.q 