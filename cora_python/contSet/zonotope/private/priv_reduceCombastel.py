"""
priv_reduceCombastel - Combastel's method for zonotope order reduction (Sec. 3.2 in [4])

Authors:       Matthias Althoff
Written:       24-January-2007 
Last update:   15-September-2007
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.zonotope import Zonotope


def priv_reduceCombastel(Z: 'Zonotope', order: int) -> 'Zonotope':
    """
    Combastel's method for zonotope order reduction (Sec. 3.2 in [4])
    
    This is a simplified implementation for now.
    """
    # For now, fall back to Girard's method
    from .priv_reduceGirard import priv_reduceGirard
    return priv_reduceGirard(Z, order) 