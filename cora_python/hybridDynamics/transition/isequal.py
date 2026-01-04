"""
isequal - checks if two transitions are equal by comparing the guard
   sets, reset functions, target locations, and synchronization labels

Syntax:
    res = isequal(trans1,trans2)
    res = isequal(trans1,trans2,tol)

Inputs:
    trans1 - transition object
    trans2 - transition object
    tol - tolerance (optional)

Outputs:
    res - true/false

Example: 
    % guard set
    guard = polytope([0 1],0,[-1 0],0);

    % reset function
    reset1 = linearReset([1,0;0,-0.75],[1;0],[0;0]);
    reset2 = linearReset([1,0;0,-0.75],[1;0],[1;0]);

    % transition
    trans1 = transition(guard,reset1,1);
    trans2 = transition(guard,reset2,1);

    % comparison
    res = isequal(trans1,trans2);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       26-November-2022
Last update:   21-May-2023 (MW, extend to arrays)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, Optional
import numpy as np
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.preprocessing.set_default_values import set_default_values as setDefaultValues
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def isequal(trans1: Any, trans2: Any, *varargin) -> bool:
    """
    Check if two transitions are equal
    
    Args:
        trans1: First transition object
        trans2: Second transition object
        *varargin: Optional tolerance (default: eps)
    
    Returns:
        bool: True if transitions are equal, False otherwise
    """
    # MATLAB: narginchk(2,3);
    # Python: varargin can be empty or have 1 element (tol)
    if len(varargin) > 1:
        raise CORAerror('CORA:tooManyInputArgs', 'isequal accepts at most 3 arguments (trans1, trans2, tol)')
    
    # MATLAB: tol = setDefaultValues({eps},varargin);
    # MATLAB: varargout = [givenValues(1:n_given), defaultValues(n_given+1:n_default)];
    # So if varargin is empty, we get [defaultValues(1)] = {eps}
    # Extract first element from returned list
    tol_list = setDefaultValues([np.finfo(float).eps], *varargin)
    tol = tol_list[0]
    
    # MATLAB: inputArgsCheck({{trans1,'att','transition'};...
    #                        {trans2,'att','transition'};...
    #                        {tol,'att','numeric',{'scalar','nonnegative','nonnan'}}});
    inputArgsCheck([
        [trans1, 'att', 'transition'],
        [trans2, 'att', 'transition'],
        [tol, 'att', 'numeric', ['scalar', 'nonnegative', 'nonnan']]
    ])
    
    # MATLAB: if any(size(trans1) ~= size(trans2))
    # Python: For now, assume single objects (not arrays)
    # TODO: Handle arrays if needed
    if not isinstance(trans1, type(trans2)):
        return False
    
    # MATLAB: loop over all individual transition objects
    # For now, handle single objects
    return _aux_isequal(trans1, trans2, tol)


def _aux_isequal(trans1: Any, trans2: Any, tol: float) -> bool:
    """
    Auxiliary function to check if two individual transitions are equal
    
    Args:
        trans1: First transition object
        trans2: Second transition object
        tol: Tolerance for comparison
    
    Returns:
        bool: True if transitions are equal, False otherwise
    """
    # MATLAB: res = true;
    res = True
    
    # MATLAB: target location
    # if any(size(trans1.target) ~= size(trans2.target)) ...
    #         || ~all(trans1.target == trans2.target)
    if isinstance(trans1.target, np.ndarray) and isinstance(trans2.target, np.ndarray):
        if trans1.target.shape != trans2.target.shape or not np.all(trans1.target == trans2.target):
            return False
    elif trans1.target != trans2.target:
        return False
    
    # MATLAB: synchronization label
    # if ~strcmp(trans1.syncLabel,trans2.syncLabel)
    if trans1.syncLabel != trans2.syncLabel:
        return False
    
    # MATLAB: reset function: can be set to [] by transition constructor, hence isempty
    # if xor(isempty(trans1.reset),isempty(trans2.reset)) ...
    #         || (~isempty(trans1.reset) && ~isequal(trans1.reset,trans2.reset,tol))
    trans1_reset_empty = (isinstance(trans1.reset, np.ndarray) and trans1.reset.size == 0) or trans1.reset is None
    trans2_reset_empty = (isinstance(trans2.reset, np.ndarray) and trans2.reset.size == 0) or trans2.reset is None
    
    if trans1_reset_empty != trans2_reset_empty:
        return False
    
    if not trans1_reset_empty:
        # Both have reset functions, check equality
        if hasattr(trans1.reset, 'isequal'):
            if not trans1.reset.isequal(trans2.reset, tol):
                return False
        elif hasattr(trans1.reset, '__eq__'):
            if trans1.reset != trans2.reset:
                return False
        else:
            # Fallback: direct comparison
            if isinstance(trans1.reset, np.ndarray) and isinstance(trans2.reset, np.ndarray):
                if not np.allclose(trans1.reset, trans2.reset, atol=tol):
                    return False
            elif trans1.reset != trans2.reset:
                return False
    
    # MATLAB: guard set
    # if any([isnumeric(trans1.guard),isnumeric(trans2.guard)])
    # MATLAB: isnumeric([]) returns true, so empty arrays are considered numeric
    trans1_guard_numeric = isinstance(trans1.guard, (int, float, np.number)) or isinstance(trans1.guard, np.ndarray)
    trans2_guard_numeric = isinstance(trans2.guard, (int, float, np.number)) or isinstance(trans2.guard, np.ndarray)
    
    # MATLAB: empty transition object may have .guard = []
    if trans1_guard_numeric or trans2_guard_numeric:
        # MATLAB: if xor(isnumeric(trans1.guard),isnumeric(trans2.guard))
        if trans1_guard_numeric != trans2_guard_numeric:
            return False
        # Both are numeric (including empty arrays)
        # MATLAB doesn't compare empty arrays element-wise, they're just considered equal if both are empty
        if isinstance(trans1.guard, np.ndarray) and isinstance(trans2.guard, np.ndarray):
            # Both are numpy arrays
            if trans1.guard.size == 0 and trans2.guard.size == 0:
                # Both empty, equal (MATLAB behavior)
                pass
            elif trans1.guard.size > 0 and trans2.guard.size > 0:
                # Both non-empty, compare element-wise
                if trans1.guard.shape != trans2.guard.shape or not np.array_equal(trans1.guard, trans2.guard):
                    return False
            else:
                # One empty, one non-empty
                return False
        elif isinstance(trans1.guard, (int, float, np.number)) and isinstance(trans2.guard, (int, float, np.number)):
            # Both are scalar numerics
            if abs(trans1.guard - trans2.guard) > tol:
                return False
        else:
            # Mixed types (shouldn't happen if both are numeric, but handle gracefully)
            return False
    else:
        # Both are contSet objects (not numeric)
        # MATLAB: elseif ~isequal(trans1.guard,trans2.guard,tol)
        if hasattr(trans1.guard, 'isequal'):
            if not trans1.guard.isequal(trans2.guard, tol):
                return False
        else:
            # Fallback: use == operator (should work for contSet objects)
            if trans1.guard != trans2.guard:
                return False
    
    return True

