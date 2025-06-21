"""
copy - copies the polytope object (used for dynamic dispatch)

Syntax:
    P_out = copy(P)

Inputs:
    P - polytope object

Outputs:
    P_out - copied polytope object

Example:
    P = Polytope(np.array([[1, 0], [-1, 1], [-1, -1]]), np.array([[1], [1], [1]]))
    P_out = copy(P)

Authors:       Mark Wetzlinger
Written:       30-September-2024
Last update:   ---
Last revision: ---
"""

from .polytope import Polytope


def copy(P):
    """
    Creates a copy of the polytope object.
    
    Args:
        P: Polytope object to copy
        
    Returns:
        P_out: Copied polytope object
    """
    # Call copy constructor
    P_out = Polytope(P)
    
    return P_out 