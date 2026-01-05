"""
contDynamics - basic class for continuous dynamics

This class serves as the base class for all continuous dynamical systems in CORA.
It provides the fundamental structure and properties that all continuous systems share.

Authors: Florian Nüssel (Python implementation)
Date: 2025-06-08
"""

from typing import Optional, Union
import numpy as np
from abc import ABC, abstractmethod
import warnings


class ContDynamics(ABC):
    """
    Basic class for continuous dynamics
    
    This class represents the base for all continuous dynamical systems.
    It stores fundamental properties like system dimensions and provides
    common functionality for all derived system classes.
    
    Attributes:
        name (str): Name of the system
        nr_of_dims (int): State dimension (number of state variables)
        nr_of_inputs (int): Input dimension (number of inputs)
        nr_of_outputs (int): Output dimension (number of outputs)
        nr_of_disturbances (int): Disturbance dimension (number of disturbances)
        nr_of_noises (int): Noise dimension (number of output noises)
    """
    
    def __init__(self, 
                 name: str = '',
                 states: int = 0,
                 inputs: int = 0,
                 outputs: int = 0,
                 dists: int = 0,
                 noises: int = 0):
        """
        Constructor for contDynamics
        
        Args:
            name: System name
            states: Number of states
            inputs: Number of inputs  
            outputs: Number of outputs
            dists: Number of disturbances
            noises: Number of disturbances on output
        """
        # Input validation
        if not isinstance(name, str):
            raise TypeError("Name must be a string")
        
        for param, value in [('states', states), ('inputs', inputs), 
                           ('outputs', outputs), ('dists', dists), ('noises', noises)]:
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{param} must be a non-negative integer")
        
        # Assign properties
        self.name = name
        self.nr_of_dims = states
        self.nr_of_inputs = inputs
        self.nr_of_outputs = outputs
        self.nr_of_disturbances = dists
        self.nr_of_noises = noises
    
    @abstractmethod
    def __repr__(self) -> str:
        """
        Detailed string representation
        """
        pass
    
    
    # Legacy property aliases for backward compatibility
    @property
    def dim(self) -> int:
        """Legacy property: use nr_of_dims instead"""
        warnings.warn("Property 'dim' is deprecated, use 'nr_of_dims' instead", 
                     DeprecationWarning, stacklevel=2)
        return self.nr_of_dims
    
    @dim.setter
    def dim(self, value: int):
        """Legacy property setter: use nr_of_dims instead"""
        warnings.warn("Property 'dim' is deprecated, use 'nr_of_dims' instead", 
                     DeprecationWarning, stacklevel=2)
        self.nr_of_dims = value
    
    @property
    def nr_of_states(self) -> int:
        """Legacy property: use nr_of_dims instead"""
        warnings.warn("Property 'nr_of_states' is deprecated, use 'nr_of_dims' instead", 
                     DeprecationWarning, stacklevel=2)
        return self.nr_of_dims
    
    @nr_of_states.setter
    def nr_of_states(self, value: int):
        """Legacy property setter: use nr_of_dims instead"""
        warnings.warn("Property 'nr_of_states' is deprecated, use 'nr_of_dims' instead", 
                     DeprecationWarning, stacklevel=2)
        self.nr_of_dims = value 
    