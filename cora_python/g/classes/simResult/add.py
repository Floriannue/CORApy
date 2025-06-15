"""
add - add simResult objects

Syntax:
    simRes = add(simRes1, simRes2)

Inputs:
    simRes1 - simResult object
    simRes2 - simResult object

Outputs:
    simRes - combined simResult object

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 29-May-2020 (MATLAB)
Last update: ---
Python translation: 2025
"""

import numpy as np


def add(simRes1, simRes2):
    """
    Add simResult objects
    
    Args:
        simRes1: First simResult object
        simRes2: Second simResult object
        
    Returns:
        Combined simResult object
    """
    from .simResult import SimResult
    
    # Validate inputs
    if not hasattr(simRes1, 'x') or not hasattr(simRes1, 't'):
        raise ValueError("simRes1 must be a simResult object")
    if not hasattr(simRes2, 'x') or not hasattr(simRes2, 't'):
        raise ValueError("simRes2 must be a simResult object")
    
    # Combine trajectories
    new_x = simRes1.x + simRes2.x
    new_t = simRes1.t + simRes2.t
    new_y = simRes1.y + simRes2.y
    new_a = simRes1.a + simRes2.a
    
    # Handle locations
    if isinstance(simRes1.loc, list) and isinstance(simRes2.loc, list):
        new_loc = simRes1.loc + simRes2.loc
    elif isinstance(simRes1.loc, list):
        new_loc = simRes1.loc + [simRes2.loc]
    elif isinstance(simRes2.loc, list):
        new_loc = [simRes1.loc] + simRes2.loc
    else:
        new_loc = [simRes1.loc, simRes2.loc]
    
    return SimResult(new_x, new_t, new_loc, new_y, new_a) 