"""
vertices_ - returns the vertices of a empty set

Syntax:
   V = vertices_(O)

Inputs:
   O - emptySet object

Outputs:
   V - vertices

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/vertices

Authors:       Tobias Ladner
Written:       11-October-2024
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np

def vertices_(self, method: str = 'convHull', *args, **kwargs):
    """
    Returns the vertices of an empty set
    
    Args:
        method: Method for vertex computation (unused, kept for interface compatibility)
        *args: Additional arguments (unused)
        **kwargs: Additional keyword arguments (unused)
    
    Returns:
        V: empty numpy array with shape (dimension, 0)
    """
    return np.zeros((self.dimension, 0)) 