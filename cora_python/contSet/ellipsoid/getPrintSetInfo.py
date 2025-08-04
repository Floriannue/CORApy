"""
getPrintSetInfo - returns all information to properly print a set 
   to the command window 

Syntax:
   [abbrev,propertyOrder] = getPrintSetInfo(E)

Inputs:
   E - ellipsoid

Outputs:
   abbrev - set abbreviation
   propertyOrder - order of the properties

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Tobias Ladner
Written:       10-October-2024
Last update:   ---
Last revision: ---

Automatic python translation: Florian Nüssel BA 2025
"""

from typing import List, Tuple

def getPrintSetInfo(self) -> Tuple[str, List[str]]:
    """
    Returns all information to properly print an ellipsoid set to the command window.

    Args:
        self: The ellipsoid object.

    Returns:
        Tuple containing:
            - abbrev (str): Set abbreviation.
            - propertyOrder (List[str]): Order of the properties.
    """
    abbrev = 'E'
    propertyOrder = ['Q','q','TOL']
    return abbrev, propertyOrder 