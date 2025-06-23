"""
uminus - overloaded unary '-' operator for simResult objects

Syntax:
    simRes = -simRes1

Inputs:
    simRes1 - simResult object

Outputs:
    simRes - resulting simResult object

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 29-May-2020 (MATLAB)
Last update: ---
Python translation: 2025
"""

import numpy as np
from .simResult import SimResult

def uminus(simRes1):
    """
    Overloaded unary '-' operator for simResult objects
    
    Args:
        simRes1: simResult object
        
    Returns:
        Resulting simResult object with negated states
    """
    
    # Negate all states
    new_x = [-x for x in simRes1.x]
    new_y = [-y for y in simRes1.y] if simRes1.y else []
    new_a = [-a for a in simRes1.a] if simRes1.a else []
    
    return SimResult(new_x, simRes1.t.copy(), simRes1.loc, new_y, new_a) 