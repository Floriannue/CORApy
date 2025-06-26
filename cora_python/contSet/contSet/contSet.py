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
    
    @abstractmethod
    def __repr__(self) -> str:
        """
        Official string representation for programmers.
        Should be unambiguous and ideally allow object reconstruction.
        """
        pass
    
    def __str__(self) -> str:
        """
        Informal string representation for users.
        Should be readable and user-friendly.
        """
        # For most contSet objects, use the display method if available
        if hasattr(self, 'display') and callable(getattr(self, 'display')):
            return self.display()
        
        # Fallback to repr if display is not available or fails
        return self.__repr__()
    
    
    # Operator overloading with proper polymorphic dispatch
    def subsasgn(self, *args):
        """Assignment to index - not supported for arrays"""
        from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
        raise CORAerror('CORA:notSupported',
                       'Given subclass of contSet does not support class arrays.')
    
    