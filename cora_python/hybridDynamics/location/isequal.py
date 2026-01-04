"""
isequal - checks if two locations are equal by comparing the invariants,
   transitions, flow equations, and names

Syntax:
    res = isequal(loc1,loc2)
    res = isequal(loc1,loc2,tol)

Inputs:
    loc1 - location object
    loc2 - location object
    tol - tolerance (optional)

Outputs:
    res - true/false

Example:
    % invariant
    inv = polytope([-1,0],0);
    
    % transition
    guard = polytope([-1,0],0,[0,1],0);

    % reset function
    reset = linearReset([1,0;0,-0.75]);

    % transition
    trans = transition(guard,reset,2);

    % flow equation
    dynamics = linearSys([0,1;0,0],[0;0],[0;-9.81]);

    % define locations
    loc1 = location('S1',inv,trans,dynamics);
    loc2 = location('S2',inv,trans,dynamics);

    % comparison
    res = isequal(loc1,loc1);
    res = isequal(loc1,loc2);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       26-November-2022
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, Optional
import numpy as np
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.preprocessing.set_default_values import set_default_values as setDefaultValues
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def isequal(loc1: Any, loc2: Any, *varargin) -> bool:
    """
    Check if two locations are equal
    
    Args:
        loc1: First location object
        loc2: Second location object
        *varargin: Optional tolerance (default: eps)
    
    Returns:
        bool: True if locations are equal, False otherwise
    """
    # MATLAB: narginchk(2,3);
    if len(varargin) > 1:
        raise CORAerror('CORA:tooManyInputArgs', 'isequal accepts at most 3 arguments (loc1, loc2, tol)')
    
    # MATLAB: tol = setDefaultValues({eps},varargin);
    tol_list = setDefaultValues([np.finfo(float).eps], *varargin)
    tol = tol_list[0]
    
    # MATLAB: inputArgsCheck({{loc1,'att','location'};...
    #                        {loc2,'att','location'};...
    #                        {tol,'att','numeric',{'scalar','nonnegative','nonnan'}}});
    inputArgsCheck([
        [loc1, 'att', 'location'],
        [loc2, 'att', 'location'],
        [tol, 'att', 'numeric', ['scalar', 'nonnegative', 'nonnan']]
    ])
    
    # MATLAB: check array length
    # For now, handle single objects (not arrays)
    if not isinstance(loc1, type(loc2)):
        return False
    
    # MATLAB: loop over all individual location objects
    # For now, handle single objects
    return _aux_isequal(loc1, loc2, tol)


def _aux_isequal(loc1: Any, loc2: Any, tol: float) -> bool:
    """
    Auxiliary function to check if two individual locations are equal
    
    Args:
        loc1: First location object
        loc2: Second location object
        tol: Tolerance for comparison
    
    Returns:
        bool: True if locations are equal, False otherwise
    """
    # MATLAB: res = true;
    res = True
    
    # MATLAB: compare invariants
    # if any([isnumeric(loc1.invariant),isnumeric(loc2.invariant)])
    loc1_inv_numeric = isinstance(loc1.invariant, (int, float, np.number)) or isinstance(loc1.invariant, np.ndarray)
    loc2_inv_numeric = isinstance(loc2.invariant, (int, float, np.number)) or isinstance(loc2.invariant, np.ndarray)
    
    # MATLAB: empty location object may have .invariant = []
    if loc1_inv_numeric or loc2_inv_numeric:
        if loc1_inv_numeric != loc2_inv_numeric:
            return False
        # Both are numeric (including empty arrays)
        if isinstance(loc1.invariant, np.ndarray) and isinstance(loc2.invariant, np.ndarray):
            if loc1.invariant.size == 0 and loc2.invariant.size == 0:
                pass  # Both empty, equal
            elif loc1.invariant.size > 0 and loc2.invariant.size > 0:
                if loc1.invariant.shape != loc2.invariant.shape or not np.array_equal(loc1.invariant, loc2.invariant):
                    return False
            else:
                return False
    else:
        # Both are contSet objects
        if hasattr(loc1.invariant, 'isequal'):
            if not loc1.invariant.isequal(loc2.invariant, tol):
                return False
        else:
            if loc1.invariant != loc2.invariant:
                return False
    
    # MATLAB: compare flow equations
    # if ~isequal(loc1.contDynamics,loc2.contDynamics,tol)
    if hasattr(loc1.contDynamics, 'isequal'):
        if not loc1.contDynamics.isequal(loc2.contDynamics, tol):
            return False
    else:
        if loc1.contDynamics != loc2.contDynamics:
            return False
    
    # MATLAB: compare transitions
    # same number of outgoing transitions
    # if length(loc1.transition) ~= length(loc2.transition)
    if len(loc1.transition) != len(loc2.transition):
        return False
    
    # MATLAB: try to find match between transitions
    # idxInLoc2 = false(length(loc1.transition));
    idxInLoc2 = [False] * len(loc1.transition)
    
    # MATLAB: for i=1:length(loc1.transition)
    for i in range(len(loc1.transition)):
        # MATLAB: found = false;
        found = False
        
        # MATLAB: loop over all transitions of second location
        for j in range(len(loc2.transition)):
            # MATLAB: skip transitions that have already been matched
            if not idxInLoc2[j]:
                # MATLAB: check for equality
                if hasattr(loc1.transition[i], 'isequal'):
                    if loc1.transition[i].isequal(loc2.transition[j], tol):
                        # MATLAB: matching transition found
                        found = True
                        idxInLoc2[j] = True
                        break
                else:
                    if loc1.transition[i] == loc2.transition[j]:
                        found = True
                        idxInLoc2[j] = True
                        break
        
        if not found:
            # MATLAB: i-th transition in loc1 has no match in loc2
            return False
    
    return True

