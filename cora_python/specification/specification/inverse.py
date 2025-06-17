"""
inverse - inverts a specification object

Syntax:
    spec = inverse(spec)

Inputs:
    spec - specification object

Outputs:
    spec - inverted specification object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: specification

Authors:       Niklas Kochdumper (MATLAB)
               Python translation by AI Assistant
Written:       26-November-2021 (MATLAB)            
Last update:   ---
Last revision: ---
Python translation: 2025
"""

from typing import Union, List
import numpy as np


def inverse(spec):
    """
    Inverts a specification object
    
    Args:
        spec: Specification object or list of specifications
        
    Returns:
        Inverted specification object(s)
        
    Raises:
        CORAError: If operation is not supported
    """
    
    # Import here to avoid circular imports
    from .specification import Specification
    from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError
    from cora_python.contSet.polytope import Polytope
    from .add import add
    
    # Handle single specification
    if isinstance(spec, Specification):
        spec = [spec]
    
    # Divide into safe and unsafe sets
    safe = []
    unsafe = []
    
    for i, s in enumerate(spec):
        # Check if time specification is empty (not supported for timed specs)
        if hasattr(s, 'time') and s.time is not None:
            if hasattr(s.time, 'representsa_') and not s.time.representsa_('emptySet', np.finfo(float).eps):
                raise CORAError('CORA:notSupported',
                              'Computing the inverse is not yet supported for timed specifications!')
            elif not _is_empty_time(s.time):
                raise CORAError('CORA:notSupported',
                              'Computing the inverse is not yet supported for timed specifications!')
        
        if s.type == 'safeSet':
            safe.append(i)
        elif s.type == 'unsafeSet':
            unsafe.append(i)
        else:
            raise CORAError('CORA:notSupported',
                          f"Computing the inverse is not yet supported for "
                          f"other types than 'safeSet' and 'unsafeSet'.")
    
    if safe and unsafe:
        raise CORAError('CORA:notSupported',
                      "Computing the inverse is not yet supported for "
                      "mixed types 'safeSet' and 'unsafeSet'.")
    
    elif safe:
        # Combine all safe set specifications to a single unsafe set
        set_obj = Polytope(spec[safe[0]].set)
        
        for i in range(1, len(safe)):
            set_obj = set_obj & Polytope(spec[safe[i]].set)
        
        return Specification(set_obj, 'unsafeSet')
    
    else:
        # Use different conversion depending on the number of sets
        if len(unsafe) == 1:
            return Specification(spec[unsafe[0]].set, 'safeSet')
        else:
            # Get all unsafe sets
            sets = [spec[unsafe[i]].set for i in range(len(unsafe))]
            
            # Convert union of safe sets to an equivalent union of unsafe
            # set representation
            sets = _safeSet2unsafeSet(sets)
            result_spec = []
            for s in sets:
                result_spec = add(result_spec, Specification(s, 'unsafeSet'))
            
            return result_spec


def _is_empty_time(time_obj) -> bool:
    """Check if time object represents empty set"""
    if time_obj is None:
        return True
    if hasattr(time_obj, 'isemptyobject') and callable(time_obj.isemptyobject):
        return time_obj.isemptyobject()
    if hasattr(time_obj, 'representsa_') and callable(time_obj.representsa_):
        return time_obj.representsa_('emptySet', np.finfo(float).eps)
    return False


def _safeSet2unsafeSet(sets: List) -> List:
    """
    Convert union of safe sets to equivalent union of unsafe sets
    This is a placeholder implementation - the full implementation would
    require more sophisticated set operations
    """
    # This is a simplified implementation
    # The full implementation would handle complex set operations
    # For now, just return the sets as-is (this needs proper implementation)
    
    # TODO: Implement proper safeSet2unsafeSet conversion
    # This involves computing the complement of the union of safe sets
    return sets 