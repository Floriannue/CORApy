"""
isempty - checks if a specification object is empty

Syntax:
    res = isempty(spec)

Inputs:
    spec - specification object

Outputs:
    res - true/false

Example:
    spec = specification();
    res = isempty(spec)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       02-May-2023 (MATLAB)
Last update:   ---
Last revision: ---
Python translation: 2025
"""

import numpy as np
from typing import Union, List
from .specification import Specification

def isempty(spec) -> bool:
    """
    Checks if a specification object is empty
    
    Args:
        spec: Specification object or list of specifications
        
    Returns:
        bool: True if specification is empty, False otherwise
    """
    
    # Import here to avoid circular imports
    
    # Handle single specification case
    if isinstance(spec, Specification):
        spec = [spec]
    
    # Check if spec list is empty
    if len(spec) == 0:
        return True
    
    # Check if any specification has an empty set
    for s in spec:
        if hasattr(s, 'set'):
            # Check if set is numeric and empty
            if isinstance(s.set, (list, tuple, np.ndarray)):
                if len(s.set) == 0 or (isinstance(s.set, np.ndarray) and s.set.size == 0):
                    return True
            # Check if set has an isemptyobject method
            elif hasattr(s.set, 'isemptyobject') and callable(s.set.isemptyobject):
                if s.set.isemptyobject():
                    return True
            # Check if set is None
            elif s.set is None:
                return True
    
    return False 