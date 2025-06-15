"""
uplus - overloaded unary '+' operator for simResult objects

Syntax:
    simRes = +simRes1

Inputs:
    simRes1 - simResult object

Outputs:
    simRes - resulting simResult object (unchanged)

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 29-May-2020 (MATLAB)
Last update: ---
Python translation: 2025
"""


def uplus(simRes1):
    """
    Overloaded unary '+' operator for simResult objects
    
    Args:
        simRes1: simResult object
        
    Returns:
        Resulting simResult object (unchanged copy)
    """
    from .simResult import SimResult
    
    # Create a copy of the simResult (unary plus doesn't change anything)
    new_x = [x.copy() for x in simRes1.x]
    new_y = [y.copy() for y in simRes1.y] if simRes1.y else []
    new_a = [a.copy() for a in simRes1.a] if simRes1.a else []
    new_t = [t.copy() for t in simRes1.t]
    
    return SimResult(new_x, new_t, simRes1.loc, new_y, new_a) 