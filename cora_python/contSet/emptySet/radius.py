"""
radius - computes the radius of the enclosing hyperball of an empty set

Syntax:
   val = radius(O)

Inputs:
   O - fullspace object

Outputs:
   val - radius

Example: 
   O = emptySet(2);
   val = radius(O);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       22-March-2023
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

def radius(self):
    """
    Computes the radius of the enclosing hyperball of an empty set
    
    Returns:
        val: radius (always 0 for empty set)
    """
    return 0 