"""
ne - overloaded '~=' operator for reachSet objects

Syntax:
    res = R1 ~= R2

Inputs:
    R1 - reachSet object
    R2 - reachSet object

Outputs:
    res - true if reachSet objects are not equal, false otherwise

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 02-June-2020 (MATLAB)
Last update: ---
Python translation: 2025
"""


def ne(R1, R2) -> bool:
    """
    Overloaded '!=' operator for reachSet objects
    
    Args:
        R1: First reachSet object
        R2: Second reachSet object
        
    Returns:
        True if reachSet objects are not equal, False otherwise
    """
    return not R1.eq(R2) 