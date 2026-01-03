"""
abstractReset - abstract superclass for reset functions

Syntax:
    reset = abstractReset(preStateDim,inputDim,postStateDim)

Inputs:
    preStateDim - dimension of state before reset
    inputDim - dimension of input before reset
    postStateDim - dimension of state after reset

Outputs:
    reset - generated abstractReset object

Example:
    reset = abstractReset(2,1,2);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       07-September-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from abc import ABC
from typing import Any
import numpy as np
from cora_python.g.functions.matlab.validate.check.assertNarginConstructor import assertNarginConstructor
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.macros.CHECKS_ENABLED import CHECKS_ENABLED


class AbstractReset(ABC):
    """
    AbstractReset - abstract superclass for reset functions
    
    This is the base class for all reset functions (linearReset, nonlinearReset).
    It defines the common properties: preStateDim, inputDim, postStateDim.
    """
    
    def __init__(self, *args):
        """
        Constructor for abstractReset
        
        Args:
            *args: Variable arguments:
                - abstractReset(): Empty reset (all dimensions 0)
                - abstractReset(preStateDim, inputDim, postStateDim): Reset with specified dimensions
                - abstractReset(other_reset): Copy constructor
        """
        # 0. check number of input arguments
        assertNarginConstructor([1, 3], len(args))
        
        # 1. copy constructor
        if len(args) == 1 and isinstance(args[0], AbstractReset):
            other = args[0]
            self.preStateDim = other.preStateDim
            self.inputDim = other.inputDim
            self.postStateDim = other.postStateDim
            return
        
        # 2. parse input arguments: varargin -> vars
        from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
        preStateDim, inputDim, postStateDim = setDefaultValues([0, 0, 0], list(args))
        
        # 3. check correctness of input arguments
        if CHECKS_ENABLED and len(args) > 0:
            inputArgsCheck([
                [preStateDim, 'att', 'numeric', ['integer', 'nonnegative', 'scalar']],
                [inputDim, 'att', 'numeric', ['integer', 'nonnegative', 'scalar']],
                [postStateDim, 'att', 'numeric', ['integer', 'nonnegative', 'scalar']]
            ])
        
        # 4. assign properties
        self.preStateDim = preStateDim
        self.inputDim = inputDim
        self.postStateDim = postStateDim

