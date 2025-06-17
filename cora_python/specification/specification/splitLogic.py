"""
splitLogic - split into temporal logic and non-logic specifications

Syntax:
    [specNonLogic, specLogic] = splitLogic(spec)

Inputs:
    spec - specification object

Outputs:
    specNonLogic - non-temporal logic specifications
    specLogic - temporal logic specifications

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: specification

Authors:       Niklas Kochdumper (MATLAB)
               Python translation by AI Assistant
Written:       17-November-2022 (MATLAB)
Last update:   ---
Last revision: ---
Python translation: 2025
"""

from typing import Tuple, List, Union


def splitLogic(spec) -> Tuple[List, List]:
    """
    Split into temporal logic and non-logic specifications
    
    Args:
        spec: Specification object or list of specifications
        
    Returns:
        Tuple of (specNonLogic, specLogic) where:
        - specNonLogic: List of non-temporal logic specifications
        - specLogic: List of temporal logic specifications
    """
    
    # Import here to avoid circular imports
    from .specification import Specification
    from .add import add
    
    # Handle single specification case
    if isinstance(spec, Specification):
        spec_list = [spec]
    else:
        spec_list = spec
    
    spec_non_logic = []
    spec_logic = []
    
    for s in spec_list:
        if s.type == 'logic':
            # Temporal logic specifications
            spec_logic = add(spec_logic, s)
        else:
            # Non-temporal logic specifications
            spec_non_logic = add(spec_non_logic, s)
    
    return spec_non_logic, spec_logic 