"""
dim - returns the dimension of the ambient space of an ellipsoid

Syntax:
    n = dim(E)

Inputs:
    E - ellipsoid object

Outputs:
    n - dimension of the ambient space

Example: 
    E = Ellipsoid([[1,0],[0,1]])
    n = dim(E) 

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger, Victor Gassmann (MATLAB)
               Python translation by AI Assistant
Written:       15-September-2019 (MATLAB)
Last update:   16-March-2021 (comp independent of property, MATLAB)
               04-July-2022 (VG, support class arrays, MATLAB)
               10-January-2024 (MW, simplify, MATLAB)
               05-October-2024 (MW, remove class arrays, MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ellipsoid import Ellipsoid


def dim(E: 'Ellipsoid') -> int:
    """
    Returns the dimension of the ambient space of an ellipsoid
    
    Args:
        E: ellipsoid object
        
    Returns:
        n: dimension of the ambient space
    """
    # Take dimension of center
    return E.q.shape[0] 