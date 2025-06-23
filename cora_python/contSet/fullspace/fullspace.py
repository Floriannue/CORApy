"""
fullspace - object constructor for n-dimensional spaces; setting n=0 is
   only permitted for some functions, because the 0-dimensional 0 vector
   cannot be represented in MATLAB (different from the 1-dimensional
   '0'). Still, the results of the remaining set operations are
   described in the respective function headers.

Description:
    This class represents objects defined as {x ∈ R^n}.

Syntax:
    obj = fullspace(n)

Inputs:
    n - dimension

Outputs:
    obj - generated fullspace object

Example: 
    n = 2
    fs = fullspace(n)
    plot(fs)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: emptySet

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       22-March-2023 (MATLAB)
Last update:   25-April-2023 (MW, disallow R^0, MATLAB)
Last revision: 10-January-2024 (MW, reformat, MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Union, Optional

from cora_python.contSet.contSet.contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.assertNarginConstructor import assertNarginConstructor
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck

if TYPE_CHECKING:
    pass


class Fullspace(ContSet):
    """
    Class for representing n-dimensional spaces
    
    Properties (SetAccess = {?contSet, ?matrixSet}, GetAccess = public):
        dimension: dimension of space
    """
    
    def __init__(self, *varargin):
        """
        Class constructor for fullspace objects
        """
        # 0. avoid empty instantiation
        if len(varargin) == 0:
            raise CORAerror('CORA:noInputInSetConstructor')
        assertNarginConstructor([1], len(varargin))

        # 1. copy constructor
        if len(varargin) == 1 and isinstance(varargin[0], Fullspace):
            # Direct assignment like MATLAB
            other = varargin[0]
            self.dimension = other.dimension
            super().__init__()
            self.precedence = 10
            return

        # 2. parse input arguments
        n = varargin[0]

        # 3. check correctness of input arguments
        inputArgsCheck([[n, 'att', 'numeric', ['scalar', 'nonnegative', 'integer']]])

        # 4. assign properties
        self.dimension = n

        # 5. set precedence (fixed) and initialize parent
        super().__init__()
        self.precedence = 10
        
    def __repr__(self) -> str:
        """Official string representation for programmers"""
        return f"Fullspace({self.dimension})"
    