"""
and - overloads '&' operator, computes the intersection of two sets

This function computes the intersection of two sets:
{s | s ∈ S₁, s ∈ S₂}

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 18-August-2022 (MATLAB)
Last update: 27-March-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Union

from cora_python.g.functions.matlab.validate.check import input_args_check
from cora_python.g.functions.matlab.validate.preprocessing import set_default_values
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def and_op(S1: 'ContSet', S2: Union['ContSet', np.ndarray], *varargin) -> 'ContSet':
    """
    and - overloads '&' operator, computes the intersection of two sets
    
    Description:
        computes the set { s | s ∈ S₁, s ∈ S₂ }
    
    Syntax:
        res = S1 & S2
        res = and(S1, S2)  
        res = and(S1, S2, type)
    
    Inputs:
        S1, S2 - ContSet object
        type - type of computation ('exact', 'inner', 'outer')
    
    Outputs:
        res - intersection set
    
    Other m-files required: none
    Subfunctions: none
    MAT-files required: none
    
    See also: none
    
    Authors:       Mark Wetzlinger
    Written:       18-August-2022
    Last update:   23-November-2022 (MW, add classname as input argument)
    Last revision: 27-March-2023 (MW, restructure relation to subclass)
    """
    
    # check number of input arguments (note: andAveraging submethod of
    # zonotope/and has up to 6 input arguments, leave that to calling and_)
    if len(varargin) > 1:
        raise CORAerror('CORA:tooManyInputArgs', 3)
    
    # check input arguments
    input_args_check([
        [S1, 'att', ['ContSet', 'numeric']],
        [S2, 'att', ['ContSet', 'numeric', 'cell']]
    ])
    
    # order input arguments according to their precedence
    S1, S2 = S1.reorder(S2)
    
    # handle different default types based on class
    if S1.__class__.__name__ == 'Ellipsoid':
        # parse input arguments
        type_ = set_default_values(['outer'], varargin)[0]
        # check additional input arguments
        input_args_check([[type_, 'str', ['inner', 'outer']]])
    elif S1.__class__.__name__ == 'Zonotope':
        # parse input arguments  
        type_ = set_default_values(['conZonotope'], varargin)[0]
        # check additional input arguments
        input_args_check([[type_, 'str', ['conZonotope', 'averaging']]])
    else:
        type_ = set_default_values(['exact'], varargin)[0]
    
    # check dimension mismatch
    from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check
    equal_dim_check(S1, S2)
    
    # call subclass method
    try:
        return S1.and_(S2, type_)
    except Exception as ME:
        # Handle empty set cases
        if (hasattr(S1, '__class__') and hasattr(S1, 'representsa_') and 
            S1.representsa_('emptySet', 1e-15, linearize=0, verbose=1)):
            return S1
        elif isinstance(S1, np.ndarray) and S1.size == 0:
            return np.array([])
        elif (hasattr(S2, '__class__') and hasattr(S2, 'representsa_') and 
              S2.representsa_('emptySet', 1e-15, linearize=0, verbose=1)):
            return S2
        elif isinstance(S2, np.ndarray) and S2.size == 0:
            return np.array([])
        else:
            raise ME 