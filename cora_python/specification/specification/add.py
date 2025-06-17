"""
add - joins two specification objects

This function combines two specification objects into a list/array
of specifications that can be checked together.

Syntax:
    spec = add(spec1, spec2)

Inputs:
    spec1 - specification object or list of specifications
    spec2 - specification object or list of specifications

Outputs:
    spec - resulting list of specification objects

Example:
    spec1 = Specification(safe_set, 'safeSet')
    spec2 = Specification(unsafe_set, 'unsafeSet')
    combined = add(spec1, spec2)

Authors: Niklas Kochdumper, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 29-May-2020 (MATLAB)
Last update: 30-April-2023 (MW, massive simplification) (MATLAB)
Python translation: 2025
"""

from typing import Union, List
from .specification import Specification


def add(spec1: Union[Specification, List[Specification]], 
        spec2: Union[Specification, List[Specification]]) -> List[Specification]:
    """
    Join two specification objects or lists of specifications
    
    Args:
        spec1: First specification(s) 
        spec2: Second specification(s)
        
    Returns:
        List of combined specifications
    """
    
    # Convert to lists if they aren't already
    if isinstance(spec1, Specification):
        spec1_list = [spec1]
    else:
        spec1_list = list(spec1)
    
    if isinstance(spec2, Specification):
        spec2_list = [spec2]
    else:
        spec2_list = list(spec2)
    
    # Combine the lists
    return spec1_list + spec2_list 