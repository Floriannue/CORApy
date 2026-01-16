"""
contSet - Base class for all continuous sets

This module contains the abstract base class ContSet which defines the interface
for all continuous set representations in CORA.

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 02-May-2007 (MATLAB)
Python translation: 2025
"""

import numpy as np
from abc import ABC, abstractmethod



class ContSet(ABC):
    """
    Abstract base class for all continuous sets
    
    This class defines the common interface and functionality for all
    continuous set representations in CORA.
    """
    
    def __init__(self):
        """Initialize the base ContSet"""
        self.precedence = 50  # Default precedence for operator overloading
        # Ensure numpy prefers contSet's __r*__ operators for mixed ops
        self.__array_priority__ = 1000

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Allow numpy ufuncs (notably matmul) to dispatch to contSet operators.
        """
        import numpy as np
        other = inputs[1] if len(inputs) == 2 and inputs[0] is self else inputs[0] if len(inputs) == 2 else None
        if other is not None and hasattr(other, 'precedence'):
            if other.precedence > self.precedence:
                return NotImplemented

        if method == '__call__' and ufunc == np.matmul and len(inputs) == 2:
            if inputs[0] is self and hasattr(self, '__matmul__'):
                return self.__matmul__(inputs[1])
            if inputs[1] is self and hasattr(self, '__rmatmul__'):
                return self.__rmatmul__(inputs[0])

        return NotImplemented
    
    @abstractmethod
    def __repr__(self) -> str:
        """
        Official string representation for programmers.
        Should be unambiguous and ideally allow object reconstruction.
        """
        pass
    
    
    # Operator overloading with proper polymorphic dispatch
    def subsasgn(self, *args):
        """Assignment to index - not supported for arrays"""
        from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
        raise CORAerror('CORA:notSupported',
                       'Given subclass of contSet does not support class arrays.')
    
    