"""
specification - class for specifications for reachability analysis

This class implements specifications that can be used to verify
properties of reachable sets in reachability analysis.

Syntax:
    spec = Specification(set, type='safeSet', time=None)

Inputs:
    set - safe/unsafe set (contSet object)
    type - specification type ('safeSet', 'unsafeSet', 'invariant')
    time - time interval for specification (VerifyTime object, optional)

Outputs:
    spec - specification object

Example:
    # Safety specification
    safe_set = Interval([-5, -5], [5, 5])
    spec = Specification(safe_set, 'safeSet')
    
    # Invariant specification with time
    inv_set = Interval([-10, -10], [10, 10])
    time_int = VerifyTime([0, 2])
    spec = Specification(inv_set, 'invariant', time_int)

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 17-June-2016 (MATLAB)
Last update: 25-July-2016 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Optional, Union, Any
from cora_python.g.classes.verifyTime import VerifyTime


class Specification:
    """
    Specification class for temporal logic specifications
    
    This class represents specifications that can be verified
    during reachability analysis, such as safety properties,
    invariants, and temporal logic formulas.
    
    Properties:
        set: The set defining the specification (contSet object)
        type: Type of specification ('safeSet', 'unsafeSet', 'invariant')
        time: Time interval for specification (VerifyTime object)
    """
    
    def __init__(self, set_obj: Any, type_: str = 'safeSet', 
                 time: Optional[Union[VerifyTime, list, np.ndarray]] = None):
        """
        Constructor for specification objects
        
        Args:
            set_obj: Safe/unsafe set (contSet object)
            type_: Specification type ('safeSet', 'unsafeSet', 'invariant')
            time: Time interval for specification (VerifyTime object or list/array)
        """
        
        # Validate specification type
        valid_types = ['safeSet', 'unsafeSet', 'invariant']
        if type_ not in valid_types:
            raise ValueError(f"Invalid specification type '{type_}'. Must be one of {valid_types}")
        
        # Store properties
        self.set = set_obj
        self.type = type_
        
        # Handle time specification
        if time is not None:
            if isinstance(time, VerifyTime):
                self.time = time
            elif isinstance(time, (list, np.ndarray)):
                # Convert to VerifyTime object
                self.time = VerifyTime(time)
            else:
                raise TypeError("Time must be a VerifyTime object, list, or numpy array")
        else:
            self.time = None
    
    def __str__(self) -> str:
        """String representation of specification"""
        if self.time is not None:
            return f"Specification: {self.type} over time {self.time}"
        else:
            return f"Specification: {self.type}"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__()
    
    def is_active(self, time: float) -> bool:
        """
        Check if specification is active at given time
        
        Args:
            time: Time to check
            
        Returns:
            bool: True if specification is active at given time
        """
        if self.time is None:
            return True  # Always active if no time constraint
        
        return self.time.contains(time)
    
    def get_time_until_switch(self, time: float) -> Optional[float]:
        """
        Get time until specification switches (becomes active/inactive)
        
        Args:
            time: Current time
            
        Returns:
            float or None: Time until switch, or None if no switch
        """
        if self.time is None:
            return None  # No time constraint
        
        return self.time.timeUntilSwitch(time)
    
    def copy(self) -> 'Specification':
        """
        Create a copy of the specification
        
        Returns:
            Specification: Copy of this specification
        """
        return Specification(self.set, self.type, self.time)
    
    def __eq__(self, other) -> bool:
        """Equality operator"""
        from .eq import eq
        return eq(self, other)
    
    def __ne__(self, other) -> bool:
        """Not equal operator"""
        return not self.__eq__(other)
    
    def isequal(self, other, tol=None) -> bool:
        """Check equality with tolerance"""
        from .isequal import isequal
        return isequal(self, other, tol)
    
    def inverse(self):
        """Invert specification"""
        from .inverse import inverse
        return inverse(self)
    
    def isempty(self) -> bool:
        """Check if specification is empty"""
        from .isempty import isempty
        return isempty(self)
    
    def project(self, dims):
        """Project specification onto subspace"""
        from .project import project
        return project(self, dims)
    
    def check(self, reach_set, time=None) -> bool:
        """Check if specification is satisfied"""
        from .check import check
        return check(self, reach_set, time)
    
    def add(self, other):
        """Add specifications together"""
        from .add import add
        return add(self, other)


def create_safety_specification(safe_set: Any, time: Optional[Any] = None) -> Specification:
    """
    Convenience function to create a safety specification
    
    Args:
        safe_set: Safe set (contSet object)
        time: Time interval (optional)
        
    Returns:
        Specification: Safety specification
    """
    return Specification(safe_set, 'safeSet', time)


def create_invariant_specification(invariant_set: Any, time: Optional[Any] = None) -> Specification:
    """
    Convenience function to create an invariant specification
    
    Args:
        invariant_set: Invariant set (contSet object)
        time: Time interval (optional)
        
    Returns:
        Specification: Invariant specification
    """
    return Specification(invariant_set, 'invariant', time)


def create_unsafe_specification(unsafe_set: Any, time: Optional[Any] = None) -> Specification:
    """
    Convenience function to create an unsafe set specification
    
    Args:
        unsafe_set: Unsafe set (contSet object)
        time: Time interval (optional)
        
    Returns:
        Specification: Unsafe set specification
    """
    return Specification(unsafe_set, 'unsafeSet', time) 