"""
getPrintSetInfo - returns all information to properly print a set to the command window

Syntax:
    abbrev, propertyOrder = getPrintSetInfo(Z)

Inputs:
    Z - zonotope

Outputs:
    abbrev - set abbreviation
    propertyOrder - order of the properties

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       10-October-2024 (MATLAB)
Last update:   --- (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

def getPrintSetInfo(self):
    """
    Returns all information to properly print a set to the command window.
    """
    abbrev = 'Zonotope'
    propertyOrder = {'c', 'G'}

    return abbrev, propertyOrder 