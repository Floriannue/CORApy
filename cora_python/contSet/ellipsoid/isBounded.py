"""
isBounded - determines if an ellipsoid is bounded

Syntax:
   res = isBounded(E)

Inputs:
   E - ellipsoid object

Outputs:
   res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       24-July-2023
Last update:   ---
Last revision: ---

Automatic python translation: Florian Nüssel BA 2025
"""

def isBounded(self) -> bool:
    """
    Determines if an ellipsoid is bounded.
    Args:
        self: The ellipsoid object.
    Returns:
        True, as ellipsoids are always bounded.
    """
    return True 