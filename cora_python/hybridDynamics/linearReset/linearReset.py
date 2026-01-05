"""
linearReset - constructor of class linearReset

Syntax:
    reset = linearReset()
    reset = linearReset(A)
    reset = linearReset(A,c)
    reset = linearReset(A,c,const_input)

Inputs:
    A - state reset matrix
    c - constant offset vector
    const_input - constant input vector

Outputs:
    reset - generated linearReset object

Example:
    reset = linearReset([1,0;0,-0.75]);
    reset = linearReset([1,0;0,-0.75],[0;0],[0;0]);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: transition, nonlinearReset

Authors:       Matthias Althoff
Written:       02-May-2007 
Last update:   ---
Last revision: 14-October-2024 (MW, update to current constructor structure)
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, Optional
import numpy as np
from cora_python.hybridDynamics.abstractReset.abstractReset import AbstractReset
from cora_python.g.functions.matlab.validate.check.assertNarginConstructor import assertNarginConstructor
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.macros.CHECKS_ENABLED import CHECKS_ENABLED


class LinearReset(AbstractReset):
    """
    LinearReset class for hybrid automata
    
    A linear reset function transforms the state after a transition:
    x_new = A * x_old + c + B * u
    
    Properties:
        A: State reset matrix
        c: Constant offset vector
        const_input: Constant input vector (for input-dependent resets)
    """
    
    def __init__(self, *args):
        """
        Constructor for linearReset
        
        Args:
            *args: Variable arguments:
                - linearReset(): Empty reset
                - linearReset(A): Reset with matrix A
                - linearReset(A, c): Reset with matrix A and offset c
                - linearReset(A, c, const_input): Reset with matrix A, offset c, and constant input
                - linearReset(other_reset): Copy constructor
        """
        # 0. empty
        assertNarginConstructor([0, 1, 2, 3], len(args))
        if len(args) == 0:
            # Empty constructor: call parent with (0, 1, 0) to match MATLAB behavior
            # MATLAB: empty linearReset has inputDim = 1 by default
            # MATLAB [] is a 2D matrix (0x0), not 1D array
            super().__init__(0, 1, 0)
            self.A = np.empty((0, 0))  # 2D empty matrix (0x0) like MATLAB []
            self.B = np.empty((0, 0))  # 2D empty matrix (0x0) like MATLAB []
            self.c = np.empty((0, 0))  # 2D empty matrix (0x0) like MATLAB []
            return
        
        # 1. copy constructor
        if len(args) == 1 and isinstance(args[0], LinearReset):
            other = args[0]
            # Call parent copy constructor
            super().__init__(other)
            self.A = other.A
            self.B = other.B
            self.c = other.c
            return
        
        # 2. parse input arguments
        A, B, c = _aux_parseInputArgs(*args)
        
        # 3. check correctness of input arguments
        if CHECKS_ENABLED:
            _aux_checkInputArgs(A, B, c, len(args))
        
        # 4. compute properties (dimensions, defaults)
        A, B, c, preStateDim, inputDim, postStateDim = _aux_computeProperties(A, B, c)
        
        # 5. instantiate parent class, assign properties
        # MATLAB: linReset@abstractReset(preStateDim,inputDim,postStateDim);
        super().__init__(preStateDim, inputDim, postStateDim)
        self.A = A
        self.B = B
        self.c = c
    
    def __repr__(self) -> str:
        return f"LinearReset(A={self.A}, B={self.B}, c={self.c})"


def _aux_parseInputArgs(*args):
    """Parse input arguments"""
    from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
    
    # Init properties (MATLAB uses empty arrays [], not None)
    # MATLAB [] is a 2D matrix (0x0), not 1D array
    A = np.empty((0, 0))  # 2D empty matrix (0x0) like MATLAB []
    B = np.empty((0, 0))  # 2D empty matrix (0x0) like MATLAB []
    c = np.empty((0, 0))  # 2D empty matrix (0x0) like MATLAB []
    
    # No input arguments
    if len(args) == 0:
        return A, B, c
    
    # Set defaults
    A, B, c = setDefaultValues([A, B, c], list(args))
    
    # Convert None to empty arrays (MATLAB uses [] not None)
    # This handles cases where None is explicitly passed (e.g., LinearReset(None, None, None))
    # MATLAB [] is a 2D matrix (0x0), not 1D array
    if A is None:
        A = np.empty((0, 0))  # 2D empty matrix (0x0) like MATLAB []
    if B is None:
        B = np.empty((0, 0))  # 2D empty matrix (0x0) like MATLAB []
    if c is None:
        c = np.empty((0, 0))  # 2D empty matrix (0x0) like MATLAB []
    
    return A, B, c


def _aux_checkInputArgs(A: Any, B: Any, c: Any, n_in: int) -> None:
    """Check correctness of input arguments"""
    if CHECKS_ENABLED and n_in > 0:
        inputArgsCheck([
            [A, 'att', 'numeric', 'matrix'],
            [B, 'att', 'numeric'],
            [c, 'att', 'numeric']
        ])
        
        # Check size (MATLAB: ~isempty(c) means size > 0)
        if c.size > 0:
            if c.ndim > 1 and c.shape[1] > 1:
                raise CORAerror('CORA:wrongInputInConstructor',
                               'Offset c must be a vector.')
            if A.size > 0:
                c_vec = c.flatten() if c.ndim > 1 else c
                if len(c_vec) != A.shape[0]:
                    raise CORAerror('CORA:wrongInputInConstructor',
                                   'Length of offset c must match row dimension of state mapping matrix A.')
            if B.size > 0:
                if A.size > 0:
                    if A.shape[0] != B.shape[0]:
                        raise CORAerror('CORA:wrongInputInConstructor',
                                       'Row dimension of input mapping matrix B must match row dimension of state mapping matrix A.')


def _aux_computeProperties(A: Any, B: Any, c: Any) -> tuple:
    """Compute properties (dimensions, defaults)"""
    # Convert None to empty arrays (MATLAB uses [] not None)
    # MATLAB [] is a 2D matrix (0x0), not 1D array
    if A is None:
        A = np.empty((0, 0))  # 2D empty matrix (0x0) like MATLAB []
    if B is None:
        B = np.empty((0, 0))  # 2D empty matrix (0x0) like MATLAB []
    if c is None:
        c = np.empty((0, 0))  # 2D empty matrix (0x0) like MATLAB []
    
    # Instantiate A as zeros if empty (MATLAB: if isempty(A) means size == 0)
    if A.size == 0:
        if B.size > 0:
            A = np.zeros((B.shape[0], 0))
        elif c.size > 0:
            c_vec = c.flatten() if c.ndim > 1 else c
            A = np.zeros((len(c_vec), 0))
        else:
            A = np.zeros((0, 0))
    
    A = np.asarray(A)
    
    # Compute dimensions
    postStateDim = A.shape[0]
    preStateDim = A.shape[1] if A.shape[1] > 0 else 0
    
    # Instantiate B as zeros (MATLAB: if isempty(B) means size == 0)
    if B.size == 0:
        B = np.zeros((postStateDim, 1))
        inputDim = 1
    else:
        B = np.asarray(B)
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        inputDim = B.shape[1]
    
    # Instantiate c as zeros (MATLAB: if isempty(c) means size == 0)
    if c.size == 0:
        c = np.zeros((postStateDim, 1))
    else:
        c = np.asarray(c)
        if c.ndim == 1:
            c = c.reshape(-1, 1)
    
    return A, B, c, preStateDim, inputDim, postStateDim

