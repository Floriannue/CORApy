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
from .specification import Specification
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.contSet.polytope import Polytope


def inverse(spec):
    """
    Inverts a specification object
    
    Args:
        spec: Specification object or list of specifications
        
    Returns:
        Inverted specification object(s)
        
    Raises:
        CORAerror: If operation is not supported
    """

    # Handle single specification
    if isinstance(spec, Specification):
        spec = [spec]
    
    # Divide into safe and unsafe sets
    safe = []
    unsafe = []
    
    for i, s in enumerate(spec):
        # Check if time specification is empty (not supported for timed specs)
        if hasattr(s, 'time') and s.time is not None:
            if hasattr(s, 'representsa_') and not s.time.representsa_('emptySet', np.finfo(float).eps):
                raise CORAerror('CORA:notSupported',
                              'Computing the inverse is not yet supported for timed specifications!')
            elif not _is_empty_time(s.time):
                raise CORAerror('CORA:notSupported',
                              'Computing the inverse is not yet supported for timed specifications!')
        
        if s.type == 'safeSet':
            safe.append(i)
        elif s.type == 'unsafeSet':
            unsafe.append(i)
        else:
            raise CORAerror('CORA:notSupported',
                          f"Computing the inverse is not yet supported for "
                          f"other types than 'safeSet' and 'unsafeSet'.")
    
    if safe and unsafe:
        raise CORAerror('CORA:notSupported',
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
                result_spec = Specification(s, 'unsafeSet').add(result_spec)
            
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


def _safeSet2unsafeSet(sets: List[Polytope]) -> List[Polytope]:
    """
    Convert union of safe sets to an equivalent representation as the union of unsafe sets.

    Args:
        sets: A list of safe sets (as Polytope objects).

    Returns:
        A list of unsafe sets (as Polytope objects) whose union is equivalent
        to the complement of the union of the input safe sets.
    """
    if not sets:
        return []

    # Represent the first safe set by the union of unsafe sets
    F = _aux_get_unsafe_sets(sets[0])

    # Loop over all remaining safe sets
    for i in range(1, len(sets)):
        # Represent current safe set by the union of unsafe sets
        F_i = _aux_get_unsafe_sets(sets[i])
        
        # Compute the intersection with the previous unsafe sets
        F_new = []
        for j in range(len(F)):
            for k in range(len(F_i)):
                # The '&' operator for polytopes computes the intersection
                intersection_poly = F[j] & F_i[k]
                if not intersection_poly.is_empty():
                    F_new.append(intersection_poly)
        
        F = F_new

    return F

def _aux_get_unsafe_sets(s: Polytope) -> List[Polytope]:
    """
    Represent the safe set S as a union of unsafe sets (its complement).
    """
    # Convert to polytope to ensure halfspace representation is available
    p = Polytope(s.A, s.b, s.Ae, s.be)
    
    # Loop over all polytope halfspaces and invert them
    A, b = p.A, p.b
    
    if A is None or b is None:
        return []

    nr_con = A.shape[0]
    F = []
    
    for i in range(nr_con):
        # Invert the halfspace constraint: A[i]x <= b[i]  --->  -A[i]x <= -b[i]
        unsafe_poly = Polytope(-A[i, :], -b[i])
        F.append(unsafe_poly)
        
    return F 