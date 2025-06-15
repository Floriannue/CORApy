"""
isemptyobject - check if reachSet object is empty

Syntax:
    res = isemptyobject(R)

Inputs:
    R - reachSet object

Outputs:
    res - true if reachSet is empty, false otherwise

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 16-May-2023 (MATLAB)
Last update: ---
Python translation: 2025
"""


def isemptyobject(R) -> bool:
    """
    Check if reachSet object is empty
    
    Args:
        R: reachSet object
        
    Returns:
        True if reachSet is empty, False otherwise
    """
    # Check if both timePoint and timeInterval are empty
    timePoint_empty = (not hasattr(R, 'timePoint') or 
                      not R.timePoint or 
                      'set' not in R.timePoint or 
                      len(R.timePoint['set']) == 0)
    
    timeInterval_empty = (not hasattr(R, 'timeInterval') or 
                         not R.timeInterval or 
                         'set' not in R.timeInterval or 
                         len(R.timeInterval['set']) == 0)
    
    return timePoint_empty and timeInterval_empty 