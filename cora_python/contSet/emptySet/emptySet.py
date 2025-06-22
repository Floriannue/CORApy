import numpy as np
from typing import Union, Optional, Tuple, Any

from cora_python.contSet.contSet.contSet import ContSet
from cora_python.g.functions.matlab.validate.check import inputArgsCheck
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check import assertNarginConstructor


class EmptySet(ContSet):
    """
    emptySet - object constructor for empty sets

    Description:
       This class represents empty sets.

    Syntax:
       obj = EmptySet(n)

    Inputs:
       n - dimension

    Outputs:
       obj - generated EmptySet object

    Example:
       n = 2;
       O = EmptySet(n);
       # plot(O); # Plotting functionality to be implemented later

    Other m-files required: none
    Subfunctions: none
    MAT-files required: none

    See also: none

    Authors:       Mark Wetzlinger
    Written:       22-March-2023
    Last update:   ---
    Last revision: 10-January-2024 (MW, reformat)
    """

    def __init__(self, *varargin):
        """
        Constructor for EmptySet objects
        
        Args:
            *varargin: Variable arguments - can be (n) or (emptySet)
        """
        # 0. avoid empty instantiation
        if len(varargin) == 0:
            raise CORAerror('CORA:noInputInSetConstructor')
        assertNarginConstructor(1, len(varargin))

        # 1. copy constructor
        if len(varargin) == 1 and isinstance(varargin[0], EmptySet):
            other = varargin[0]
            self.dimension = other.dimension
            super().__init__()
            return

        # 2. parse input arguments
        n = varargin[0]

        # 3. check correctness of input arguments
        inputArgsCheck([[n, 'att', 'numeric', ['scalar', 'nonnegative', 'integer']]])
        
        # 4. assign properties
        self.dimension = n

        # Initialize parent class
        super().__init__()

        # 5. set precedence (fixed)
        self.precedence = 0

    # Abstract methods implementation (required by ContSet)
    def dim(self) -> int:
        """Get dimension of the empty set"""
        return self.dimension
    
    def is_empty(self) -> bool:
        """Empty set is always empty"""
        return True

    def __repr__(self) -> str:
        """Return programmer-friendly representation."""
        return f"EmptySet({self.dimension})"

    def __str__(self) -> str:
        """Return user-friendly representation."""
        if hasattr(self, 'display') and callable(getattr(self, 'display')):
            return self.display()
        return self.__repr__() 