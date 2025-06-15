"""
minus - overloaded '-' operator for simResult objects

Syntax:
    simRes = simRes1 - simRes2

Inputs:
    simRes1 - simResult object
    simRes2 - simResult object or numeric value

Outputs:
    simRes - resulting simResult object

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 29-May-2020 (MATLAB)
Last update: ---
Python translation: 2025
"""

import numpy as np


def minus(simRes1, simRes2):
    """
    Overloaded '-' operator for simResult objects
    
    Args:
        simRes1: First simResult object
        simRes2: Second simResult object or numeric value
        
    Returns:
        Resulting simResult object
    """
    from .simResult import SimResult
    
    # If simRes2 is numeric, subtract from all states
    if isinstance(simRes2, (int, float, np.ndarray)):
        new_x = [x - simRes2 for x in simRes1.x]
        new_y = [y - simRes2 for y in simRes1.y] if simRes1.y else []
        new_a = [a - simRes2 for a in simRes1.a] if simRes1.a else []
        
        return SimResult(new_x, simRes1.t.copy(), simRes1.loc, new_y, new_a)
    
    # If simRes2 is another simResult, this is more complex (not commonly used)
    elif hasattr(simRes2, 'x') and hasattr(simRes2, 't'):
        raise NotImplementedError("Subtraction between simResult objects not implemented")
    
    else:
        raise TypeError(f"Unsupported operand type for -: simResult and {type(simRes2)}") 