"""
randPoint_ - generates random points within an empty set

Syntax:
   p = randPoint_(O)
   p = randPoint_(O,N)
   p = randPoint_(O,N,type)
   p = randPoint_(O,'all','extreme')

Inputs:
   O - emptySet object
   N - number of random points
   type - type of the random point ('standard', 'extreme')

Outputs:
   p - random point in empty set

Example: 
   O = emptySet(2);
   p = randPoint(O);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/randPoint

Authors:       Mark Wetzlinger
Written:       05-April-2023
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np

def randPoint_(self, N=None, type=None, *args):
    """
    Generates random points within an empty set
    
    Args:
        N: number of random points (ignored for empty set)
        type: type of the random point (ignored for empty set)
        *args: additional arguments (ignored for empty set)
        
    Returns:
        p: empty numpy array with shape (dimension, 0)
    """
    # unfortunately, we cannot concatenate empty column vectors...
    # Return empty numpy array with correct dimensions
    return np.empty((self.dimension, 0)) 