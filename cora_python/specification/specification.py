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
from ..g.classes.verifyTime import VerifyTime


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
    
    def check(self, reach_set: Any, time: Optional[float] = None) -> bool:
        """
        Check if specification is satisfied by reachable set
        
        Args:
            reach_set: Reachable set to check (contSet object)
            time: Current time (optional)
            
        Returns:
            bool: True if specification is satisfied, False otherwise
        """
        
        # Check time constraints
        if self.time is not None and time is not None:
            if not self.time.contains(time):
                return True  # Specification not active at this time
        
        # Check specification based on type
        if self.type == 'safeSet':
            # Safe set: reachable set must be contained in safe set
            return self.set.contains(reach_set)
        
        elif self.type == 'unsafeSet':
            # Unsafe set: reachable set must not intersect unsafe set
            return not reach_set.isIntersecting(self.set)
        
        elif self.type == 'invariant':
            # Invariant: reachable set must always be in invariant set
            return self.set.contains(reach_set)
        
        else:
            raise ValueError(f"Unknown specification type: {self.type}")
    
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