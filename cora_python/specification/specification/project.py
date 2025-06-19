"""
project - projects the set of a specification onto a subspace

Syntax:
    spec = project(spec, dims)

Inputs:
    spec - specification object
    dims - dimensions for projection

Outputs:
    spec - projected specification object

Example:
    Z = zonotope([1;-1;0],[1 3 -2; 0 -1 1; 1 2 0]);
    spec = specification(Z,'safeSet');
    spec_ = project(spec,[1,3]);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       26-June-2022 (MATLAB)
Last update:   30-April-2023 (MW, bug fix for arrays, add 'logic') (MATLAB)
Last revision: ---
Python translation: 2025
"""

from typing import Union, List
from .specification import Specification
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def project(spec, dims):
    """
    Projects the set of a specification onto a subspace
    
    Args:
        spec: Specification object or list of specifications
        dims: Dimensions for projection (list or array)
        
    Returns:
        Projected specification object(s)
        
    Raises:
        CORAError: If projection is not supported for the specification type
    """
    
    # Import here to avoid circular imports
    
    # Handle single specification case
    if isinstance(spec, Specification):
        spec_list = [spec]
        return_single = True
    else:
        spec_list = spec
        return_single = False
    
    # Loop over array of specifications
    for i, s in enumerate(spec_list):
        # Check type
        if s.type in ['invariant', 'safeSet', 'unsafeSet']:
            # Project set
            if hasattr(s.set, 'project') and callable(s.set.project):
                s.set = s.set.project(dims)
            else:
                raise CORAError('CORA:notSupported',
                              f"Set type {type(s.set).__name__} does not support projection")
        
        elif s.type in ['custom', 'logic']:
            raise CORAError('CORA:notSupported',
                          "Projection of a specification of types 'custom' or "
                          "'logic' is not yet supported.")
        else:
            raise CORAError('CORA:notSupported',
                          f"Projection not supported for specification type '{s.type}'")
    
    # Return single spec or list based on input
    if return_single:
        return spec_list[0]
    else:
        return spec_list 