"""
getPrintSetInfo - returns all information to properly print a set 
   to the command window 

Syntax:
   [abbrev,propertyOrder] = getPrintSetInfo(O)

Inputs:
   O - emptySet

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

def getPrintSetInfo(self):
    """
    Returns all information to properly print a set to the command window
    
    Returns:
        tuple: (abbrev, propertyOrder) where abbrev is the set abbreviation
               and propertyOrder is the order of the properties
    """
    abbrev = 'O'
    propertyOrder = ['dimension']
    
    return abbrev, propertyOrder 