"""
transition - constructor of class transition

Syntax:
    trans = transition()
    trans = transition(guard,reset,target)
    trans = transition(other_transition)

Inputs:
    guard - guard set (contSet)
    reset - reset function (reset object)
    target - target location index

Outputs:
    trans - generated transition object

Example:
    guard = polytope([0,1],0);
    reset = linearReset([1,0;0,-0.75]);
    trans = transition(guard,reset,1);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: location, hybridAutomaton, linearReset, nonlinearReset

Authors:       Matthias Althoff
Written:       02-May-2007 
Last update:   ---
Last revision: 14-October-2024 (MW, update to current constructor structure)
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, Optional
import numpy as np
from cora_python.g.functions.matlab.validate.check.assertNarginConstructor import assertNarginConstructor
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.macros.CHECKS_ENABLED import CHECKS_ENABLED


class Transition:
    """
    Transition class for hybrid automata
    
    A transition represents an edge in a hybrid automaton with:
    - A guard set (where the transition can be taken)
    - A reset function (how the state is transformed)
    - A target location (where the system goes after the transition)
    
    Properties:
        guard: Guard set (contSet)
        reset: Reset function (reset object)
        target: Target location index (int)
    """
    
    def __init__(self, *args):
        """
        Constructor for transition
        
        Args:
            *args: Variable arguments:
                - transition(): Empty transition
                - transition(guard, reset, target): Transition with guard, reset, and target
                - transition(other_transition): Copy constructor
        """
        # 0. empty
        assertNarginConstructor([0, 1, 3, 4], len(args))
        if len(args) == 0:
            # MATLAB: guard = []; reset = []; target = [];
            # MATLAB [] is a 2D empty array (0x0), not None
            self.guard = np.empty((0, 0))  # MATLAB: guard = [] (empty array)
            self.reset = np.empty((0, 0))  # MATLAB: reset = [] (empty array)
            self.target = np.empty((0, 0))  # MATLAB: target = [] (empty array)
            self.syncLabel = ''
            return
        
        # 1. copy constructor
        if len(args) == 1 and isinstance(args[0], Transition):
            other = args[0]
            self.guard = other.guard
            self.reset = other.reset
            self.target = other.target
            self.syncLabel = other.syncLabel
            return
        
        # 2. parse input arguments
        guard, reset, target, syncLabel = _aux_parseInputArgs(*args)
        
        # 3. check correctness of input arguments
        if CHECKS_ENABLED:
            _aux_checkInputArgs(guard, reset, target, syncLabel, len(args))
        
        # 4. compute dependent properties
        guard, reset, target, syncLabel = _aux_computeProperties(guard, reset, target, syncLabel)
        
        # 5. assign properties
        self.guard = guard
        self.reset = reset
        self.target = target
        self.syncLabel = syncLabel
    
    def __repr__(self) -> str:
        return f"Transition(guard={self.guard}, reset={self.reset}, target={self.target})"


def _aux_parseInputArgs(*args):
    """Parse input arguments"""
    from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
    
    # default properties (MATLAB uses [] not None)
    # MATLAB [] is a 2D matrix (0x0), not 1D array
    guard = np.empty((0, 0))
    reset = np.empty((0, 0))
    target = np.empty((0, 0))
    syncLabel = ''
    
    # parse arguments
    # MATLAB: [guard,reset,target,syncLabel] = setDefaultValues({guard,reset,target,syncLabel},varargin);
    guard, reset, target, syncLabel = setDefaultValues([guard, reset, target, syncLabel], list(args))
    
    # Convert None to empty arrays (MATLAB uses [] not None)
    # This handles cases where None is explicitly passed
    # MATLAB [] is a 2D matrix (0x0), not 1D array
    if guard is None:
        guard = np.empty((0, 0))
    if reset is None:
        reset = np.empty((0, 0))
    if target is None:
        target = np.empty((0, 0))
    
    return guard, reset, target, syncLabel


def _aux_checkInputArgs(guard: Any, reset: Any, target: Any, syncLabel: str, n_in: int) -> None:
    """Check correctness of input arguments"""
    if CHECKS_ENABLED and n_in > 0:
        inputArgsCheck([
            [guard, 'att', ('interval', 'polytope', 'levelSet', 'fullspace')],
            [reset, 'att', ('abstractReset', 'struct'), 'scalar'],
            [target, 'att', 'numeric', ('column', 'nonempty', 'integer', 'positive')],
            [syncLabel, 'att', 'char']
        ])
        
        # pre-state dimension of reset function must match dimension of guard
        if hasattr(reset, 'preStateDim') and hasattr(guard, 'dim'):
            if guard.dim() != reset.preStateDim:
                raise CORAerror('CORA:wrongInputInConstructor',
                               'Dimension of guard set must match pre-state dimension of reset function.')


def _aux_computeProperties(guard: Any, reset: Any, target: Any, syncLabel: str) -> tuple:
    """Compute dependent properties"""
    # backward compatibility for structs
    if isinstance(reset, dict):
        # MATLAB: CORAwarning("CORA:deprecated","property","reset","CORA v2025",...)
        # try to convert the given struct to a linearReset/nonlinearReset object
        try:
            if 'A' in reset:
                # linear reset function, has .A and .c, potentially also .B
                from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
                if 'B' in reset:
                    reset = LinearReset(reset['A'], reset['B'], reset.get('c', None))
                else:
                    reset = LinearReset(reset['A'], None, reset.get('c', None))
            elif 'f' in reset:
                # nonlinear reset function
                from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
                reset = NonlinearReset(reset['f'])
            else:
                raise CORAerror('CORA:wrongInputInConstructor',
                               'Reset function must be a linearReset or nonlinearReset object.')
        except Exception:
            # conversion not successful... immediately direct away from struct
            # usage and toward classes
            raise CORAerror('CORA:wrongInputInConstructor',
                           'Reset function must be a linearReset or nonlinearReset object.')
    
    return guard, reset, target, syncLabel

