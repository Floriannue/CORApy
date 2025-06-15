"""
isemptyobject - check if simResult object is empty

Syntax:
    res = isemptyobject(simRes)

Inputs:
    simRes - simResult object

Outputs:
    res - true if simResult is empty, false otherwise

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 16-May-2023 (MATLAB)
Last update: ---
Python translation: 2025
"""


def isemptyobject(simRes) -> bool:
    """
    Check if simResult object is empty
    
    Args:
        simRes: simResult object
        
    Returns:
        True if simResult is empty, False otherwise
    """
    # Check if x (states) is empty - this is the primary indicator
    if not hasattr(simRes, 'x') or not simRes.x or len(simRes.x) == 0:
        return True
    
    # Check if all trajectories in x are empty
    for x_traj in simRes.x:
        if x_traj is not None and len(x_traj) > 0:
            return False
    
    return True 