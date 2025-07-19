"""
getPrintSetInfo - returns all information to properly print a set 
   to the command window 

Syntax:
   [abbrev,propertyOrder] = getPrintSetInfo(fs)

Inputs:
   fs - fullspace

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

----------------------------- BEGIN CODE ------------------------------
"""

from typing import List, Tuple

def getPrintSetInfo(fs) -> Tuple[str, List[str]]:
    """
    Returns all information to properly print a set to the command window
    
    Args:
        fs: fullspace object
        
    Returns:
        abbrev: set abbreviation
        propertyOrder: order of the properties
    """
    abbrev = 'fs'
    propertyOrder = ['dimension']
    
    return abbrev, propertyOrder

# ------------------------------ END OF CODE ------------------------------ 