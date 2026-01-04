"""
isequal - checks if two hybrid automata are equal

Syntax:
    res = isequal(HA1,HA2)
    res = isequal(HA1,HA2,tol)

Inputs:
    HA1 - hybridAutomaton object
    HA2 - hybridAutomaton object
    tol - tolerance (optional)

Outputs:
    res - true/false

Example: 
    ---

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       10-January-2023
Last update:   21-May-2023 (MW, extend to arrays)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, Optional
import numpy as np
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.preprocessing.set_default_values import set_default_values as setDefaultValues
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def isequal(HA1: Any, HA2: Any, *varargin) -> bool:
    """
    Check if two hybrid automata are equal
    
    Args:
        HA1: First hybridAutomaton object
        HA2: Second hybridAutomaton object
        *varargin: Optional tolerance (default: eps)
    
    Returns:
        bool: True if hybrid automata are equal, False otherwise
    """
    # MATLAB: narginchk(2,3);
    if len(varargin) > 1:
        raise CORAerror('CORA:tooManyInputArgs', 'isequal accepts at most 3 arguments (HA1, HA2, tol)')
    
    # MATLAB: tol = setDefaultValues({eps},varargin);
    tol_list = setDefaultValues([np.finfo(float).eps], *varargin)
    tol = tol_list[0]
    
    # MATLAB: inputArgsCheck({{HA1,'att','hybridAutomaton'};...
    #                        {HA2,'att','hybridAutomaton'};...
    #                        {tol,'att','numeric',{'scalar','nonnegative','nonnan'}}});
    inputArgsCheck([
        [HA1, 'att', 'hybridAutomaton'],
        [HA2, 'att', 'hybridAutomaton'],
        [tol, 'att', 'numeric', ['scalar', 'nonnegative', 'nonnan']]
    ])
    
    # MATLAB: check array length
    # For now, handle single objects (not arrays)
    if not isinstance(HA1, type(HA2)):
        return False
    
    # MATLAB: loop over all individual hybridAutomaton objects
    # For now, handle single objects
    return _aux_isequal(HA1, HA2, tol)


def _aux_isequal(HA1: Any, HA2: Any, tol: float) -> bool:
    """
    Auxiliary function to check if two individual hybrid automata are equal
    
    Args:
        HA1: First hybridAutomaton object
        HA2: Second hybridAutomaton object
        tol: Tolerance for comparison
    
    Returns:
        bool: True if hybrid automata are equal, False otherwise
    """
    # MATLAB: check number of locations
    # if length(HA1.location) ~= length(HA2.location)
    if len(HA1.location) != len(HA2.location):
        return False
    
    # MATLAB: loop over locations
    # for i=1:length(HA1.location)
    for i in range(len(HA1.location)):
        # MATLAB: compare locations
        # if ~isequal(HA1.location(i),HA2.location(i),tol)
        if hasattr(HA1.location[i], 'isequal'):
            if not HA1.location[i].isequal(HA2.location[i], tol):
                return False
        else:
            if HA1.location[i] != HA2.location[i]:
                return False
    
    # MATLAB: all checks ok
    return True

