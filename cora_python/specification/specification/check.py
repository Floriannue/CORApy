"""
check - check if reachable set satisfies specification

Syntax:
    res = check(spec, S, time)
    [res, indSpec, indObj] = check(spec, S, time)

Inputs:
    spec - specification object  
    S - contSet object, reachSet object, or simResult object
    time - time interval (interval object, default: interval.empty())

Outputs:
    res - true if specification satisfied, false otherwise
    indSpec - index of violated specification (if multiple specs)
    indObj - index of violated object in S (for simulations/reachSets)

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 18-June-2023 (MATLAB)
Python translation: 2025
"""

from typing import Union, Tuple, Optional, Any
import numpy as np
from cora_python.contSet.interval.interval import Interval

def check(spec, S, time: Optional[Interval] = None) -> Union[bool, Tuple[bool, int, Any]]:
    """
    Check if reachable set satisfies specification
    
    Checks whether a given set, simulation result, or reachable set
    satisfies the given specification according to MATLAB logic.
    
    Args:
        spec: specification object or list of specifications
        S: contSet object, simResult object, or reachSet object  
        time: time interval for timed specifications (default: empty interval)
        
    Returns:
        res: True if specification satisfied, False otherwise
        indSpec: index of violated specification (if violated)
        indObj: index of violated object (for arrays/cells)
        
    Example:
        >>> from cora_python.specification.specification.specification import Specification
        >>> from cora_python.contSet.polytope.polytope import Polytope
        >>> set_spec = Polytope(np.array([[1, 1]]), np.array([1]))
        >>> spec = Specification(set_spec, 'safeSet')
        >>> # Test with some set S
        >>> res = check(spec, S)
    """
    # Set default time if not provided
    if time is None:
        time = Interval.empty()
    
    # Initialize return values
    res = True
    indSpec = 0
    indObj = 1
    
    eps = 1e-8
    
    # Handle different input types for S
    if hasattr(S, '__class__') and S.__class__.__name__ == 'simResult':
        # Simulation result case - not implemented yet
        raise NotImplementedError("simResult checking not yet implemented")
        
    elif hasattr(S, '__class__') and S.__class__.__name__ == 'reachSet':
        # Reachable set case - not implemented yet  
        raise NotImplementedError("reachSet checking not yet implemented")
        
    else:
        # contSet case
        
        # Handle list/array of specifications
        if isinstance(spec, list):
            specs = spec
        else:
            specs = [spec]
            
        # Loop over all specifications
        for i, spec_i in enumerate(specs):
            
            # Check if time frames overlap
            if (hasattr(time, 'representsa_') and time.representsa_('emptySet', eps) and 
                hasattr(spec_i, 'time') and spec_i.time is not None and not spec_i.time.isemptyobject()):
                from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
                raise CORAerror('CORA:specialError',
                               'Timed specifications require a time interval.')
            
            # Check if specification is active at this time
            if (not hasattr(spec_i, 'time') or spec_i.time is None or spec_i.time.isemptyobject() or
                (hasattr(spec_i, 'time') and spec_i.time is not None and hasattr(time, 'isIntersecting_') and 
                 spec_i.time.isIntersecting_(time, 'exact', 1e-8))):
                
                # Check different types of specifications
                if spec_i.type == 'invariant':
                    res = _aux_checkInvariant(spec_i.set, S)
                elif spec_i.type == 'unsafeSet':
                    res = _aux_checkUnsafeSet(spec_i.set, S)
                elif spec_i.type == 'safeSet':
                    res = _aux_checkSafeSet(spec_i.set, S)
                elif spec_i.type == 'custom':
                    res = _aux_checkCustom(spec_i.set, S)
                else:
                    from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
                    raise CORAerror('CORA:wrongValue', f'Unknown specification type: {spec_i.type}')
                
                # Return as soon as one specification is violated
                if not res:
                    indSpec = i + 1  # MATLAB uses 1-based indexing
                    indObj = 1
                    return res, indSpec, indObj
    
    return res, indSpec, indObj


def _aux_checkUnsafeSet(set_spec, S) -> bool:
    """
    Check if reachable set intersects the unsafe sets
    
    According to MATLAB: res = ~isIntersecting_(set,S,'exact',1e-8)
    Returns True if S does NOT intersect with unsafe set (safe)
    Returns False if S DOES intersect with unsafe set (unsafe)
    """
    # Check if cell array is given
    wasCell = isinstance(S, list)
    if not wasCell:
        S = [S]
    
    # S is cell, check each
    res = True
    for i, S_i in enumerate(S):
        try:
            # Try exact intersection check first
            intersects = set_spec.isIntersecting_(S_i, 'exact', 1e-8)
            res = not intersects  # Safe if NO intersection
        except:
            try:
                # Fall back to approximate check
                intersects = set_spec.isIntersecting_(S_i, 'approx', 1e-8)
                res = not intersects  # Safe if NO intersection  
            except:
                # If both fail, be conservative and assume intersection (unsafe)
                res = False
        
        # Early exit if unsafe
        if not res:
            return res
    
    return res


def _aux_checkSafeSet(set_spec, S) -> bool:
    """
    Check if reachable set is inside the safe set
    
    According to MATLAB: res = contains(set,S,'approx')
    Returns True if safe set CONTAINS S
    Returns False if safe set does NOT contain S
    """
    if isinstance(S, list):
        res = True
        for i, S_i in enumerate(S):
            try:
                res = set_spec.contains(S_i, 'approx')
            except:
                # If contains check fails, be conservative and assume not contained
                res = False
            if not res:
                return res
    else:
        try:
            res = set_spec.contains(S, 'approx')
        except:
            # If contains check fails, be conservative and assume not contained
            res = False
    
    return res


def _aux_checkCustom(func, S) -> bool:
    """
    Check if the reachable set satisfies a user provided specification
    """
    if isinstance(S, list):
        res = False
        for i, S_i in enumerate(S):
            res = func(S_i)
            if res:
                return res
    else:
        res = func(S)
    
    return res


def _aux_checkInvariant(set_spec, S) -> bool:
    """
    Check if reachable set intersects the invariant
    
    According to MATLAB: res = isIntersecting_(set,S,'approx',1e-8)
    Returns True if S DOES intersect with invariant set
    Returns False if S does NOT intersect with invariant set
    """
    if isinstance(S, list):
        res = False
        for i, S_i in enumerate(S):
            res = set_spec.isIntersecting_(S_i, 'approx', 1e-8)

    else:
        res = set_spec.isIntersecting_(S, 'approx', 1e-8)

    
    return res 