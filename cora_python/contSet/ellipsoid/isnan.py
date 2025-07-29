"""
isnan - checks if any value in the ellipsoid is NaN

Syntax:
   res = isnan(E)

Inputs:
   E - ellipsoid object

Outputs:
   res - false

Example: 
   E = Ellipsoid(np.array([[5, 7],[7, 13]]),np.array([[1],[2]]))
   res = E.isnan()

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       01-June-2022
Last update:   04-July-2022 (VG, support class arrays)
Last revision: ---

Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np

def isnan(self) -> bool:
    """
    Checks if any value in the ellipsoid is NaN.
    Args:
        self: The ellipsoid object.
    Returns:
        False, as ellipsoid objects do not contain NaN values.
    """
    # In MATLAB, for a single object E, size(E) is [1,1], so false(size(E)) returns false.
    # For an array of ellipsoid objects, it would return a logical array of false values.
    # Assuming a single ellipsoid object for now, as class arrays are not fully handled yet.
    return False 