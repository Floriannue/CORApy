"""
dim - returns the dimension of the ambient space of an ellipsoid

Syntax:
    n = dim(E)

Inputs:
    E - ellipsoid object

Outputs:
    n - dimension of the ambient space

Example: 
    E = Ellipsoid(np.array([[1,0],[0,1]]));
    n = dim(E) 

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger, Victor Gassmann
Written:       15-September-2019 
Last update:   16-March-2021 (comp independent of property)
               04-July-2022 (VG, support class arrays)
               10-January-2024 (MW, simplify)
               05-October-2024 (MW, remove class arrays)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ellipsoid import Ellipsoid

def dim(E: 'Ellipsoid') -> int:
    """
    dim - returns the dimension of the ambient space of an ellipsoid

    Syntax:
        n = dim(E)

    Inputs:
        E - ellipsoid object

    Outputs:
        n - dimension of the ambient space

    Example: 
        E = Ellipsoid(np.array([[1,0],[0,1]]));
        n = dim(E) 

    Authors:       Mark Wetzlinger, Victor Gassmann
    Written:       15-September-2019 
    Last update:   16-March-2021 (comp independent of property)
                   04-July-2022 (VG, support class arrays)
                   10-January-2024 (MW, simplify)
                   05-October-2024 (MW, remove class arrays)
    Last revision: ---
                   Automatic python translation: Florian Nüssel BA 2025
    """
    # take dimension of center
    n = E.q.shape[0]
    return n 