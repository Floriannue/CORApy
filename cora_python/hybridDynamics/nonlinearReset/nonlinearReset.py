"""
nonlinearReset - constructor for nonlinear reset functions

Description:
    This class represents nonlinear reset functions
        x_ = f(x,u)
    where x and u are the state and input before transition, and x_ is the
    state after transition

Syntax:
    nonlinReset = nonlinearReset()
    nonlinReset = nonlinearReset(f)
    nonlinReset = nonlinearReset(f,preStateDim,inputDim,postStateDim)

Inputs:
    f - function handle
    preStateDim - length of vector x
    inputDim - length of vector u
    postStateDim - length of vector x_

Outputs:
    nonlinReset - generated nonlinearReset object

Example:
    f = @(x,u) [x(1) + 2*u(1); x(2)];
    nonlinReset = nonlinearReset(f);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: reset, linearReset

Authors:       Mark Wetzlinger
Written:       07-September-2024
Last update:   15-October-2024 (MW, add dimensions to input arguments)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, Optional, Callable
import numpy as np
import inspect
from cora_python.hybridDynamics.abstractReset.abstractReset import AbstractReset
from cora_python.g.functions.matlab.validate.check.assertNarginConstructor import assertNarginConstructor
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.macros.CHECKS_ENABLED import CHECKS_ENABLED


class NonlinearReset(AbstractReset):
    """
    NonlinearReset class for hybrid automata
    
    A nonlinear reset function transforms the state after a transition:
    x_new = f(x_old, u)
    
    Properties:
        f: Function handle to mapping function
        tensorOrder: Tensor order for evaluation (computed by derivatives)
        J: Function handle to Jacobian matrix (computed by derivatives)
        H: Function handle to Hessian matrix (computed by derivatives)
        T: Function handle to third-order tensor (computed by derivatives)
    """
    
    def __init__(self, *args):
        """
        Constructor for nonlinearReset
        
        Args:
            *args: Variable arguments:
                - nonlinearReset(): Empty reset
                - nonlinearReset(f): Reset with function handle f
                - nonlinearReset(f, preStateDim, inputDim, postStateDim): Reset with function and dimensions
        """
        # 0. check number of input arguments
        assertNarginConstructor([0, 1, 4], len(args))
        
        # 1. copy constructor: not allowed due to obj@abstractReset below...
        # (commented out in MATLAB)
        
        # 2. parse input arguments: varargin -> vars
        f, preStateDim, inputDim, postStateDim = _aux_parseInputArgs(*args)
        
        # 3. check correctness of input arguments
        _aux_checkInputArgs(f, preStateDim, inputDim, postStateDim, len(args))
        
        # 4. compute number of states, inputs, and outputs
        f, preStateDim, inputDim, postStateDim = _aux_computeProperties(f, preStateDim, inputDim, postStateDim)
        
        # 5. instantiate parent class, assign properties
        # MATLAB: nonlinReset@abstractReset(preStateDim,inputDim,postStateDim);
        super().__init__(preStateDim, inputDim, postStateDim)
        self.f = f
        # Properties computed by derivatives method
        self.tensorOrder = None
        self.J = None
        self.H = None
        self.T = None
    
    def __repr__(self) -> str:
        return f"NonlinearReset(f={self.f}, preStateDim={self.preStateDim}, inputDim={self.inputDim}, postStateDim={self.postStateDim})"


def _aux_parseInputArgs(*args):
    """Parse input arguments from user and assign to variables"""
    from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
    
    # Init properties
    f = lambda x, u: np.array([]).reshape(0, 1)
    preStateDim = None
    inputDim = None
    postStateDim = None
    
    # No input arguments
    if len(args) == 0:
        return f, preStateDim, inputDim, postStateDim
    
    # Only function handle given
    if len(args) == 1:
        f = args[0]
        return f, preStateDim, inputDim, postStateDim
    
    # All dimensions must be given
    if len(args) != 4:
        raise CORAerror('CORA:numInputArgsConstructor',
                       'nonlinearReset requires 0, 1, or 4 input arguments.')
    
    # Set defaults
    f, preStateDim, inputDim, postStateDim = setDefaultValues([f, preStateDim, inputDim, postStateDim], list(args))
    
    return f, preStateDim, inputDim, postStateDim


def _aux_checkInputArgs(f: Any, preStateDim: Any, inputDim: Any, postStateDim: Any, n_in: int) -> None:
    """Ensure that nonlinear reset function x_ = f(x,u) is properly defined"""
    if CHECKS_ENABLED and n_in > 0:
        # Only check dimensions if they are provided (not None)
        # MATLAB uses [] (empty array) which is different from None, but in Python
        # we use None to represent "not provided" and only validate when provided
        check_list = [[f, 'att', 'function_handle']]
        if preStateDim is not None:
            check_list.append([preStateDim, 'att', 'numeric'])
        if inputDim is not None:
            check_list.append([inputDim, 'att', 'numeric'])
        if postStateDim is not None:
            check_list.append([postStateDim, 'att', 'numeric'])
        
        inputArgsCheck(check_list)
        
        # If preStateDim and inputDim provided, check if they are plausible
        # by inserting vector of given size into the function handle
        if preStateDim is not None and inputDim is not None:
            try:
                f_out = f(np.zeros((preStateDim, 1)), np.zeros((inputDim, 1)))
            except Exception:
                raise CORAerror('CORA:wrongInputInConstructor',
                               'Sizes for pre-state and input do not match the given function handle.')
            # Check if output has appropriate size (must be vector)
            if postStateDim is not None:
                f_out = np.asarray(f_out)
                if f_out.ndim > 1 and f_out.shape[1] > 1:
                    raise CORAerror('CORA:wrongInputInConstructor',
                                   'Function handle must return a column vector.')
                elif len(f_out.flatten()) != postStateDim:
                    raise CORAerror('CORA:wrongInputInConstructor',
                                   'Size for post-state does not match the given function handle.')


def _aux_computeProperties(f: Callable, preStateDim: Any, inputDim: Any, postStateDim: Any) -> tuple:
    """Compute number of states, inputs, and outputs"""
    from cora_python.g.functions.matlab.function_handle.inputArgsLength import inputArgsLength
    
    # As long as aux_parseInputArgs enforces either 1 or 4 input arguments,
    # we must only check 'preStateDim' and can then compute all values
    if preStateDim is None:
        count, postStateDim_tuple = inputArgsLength(f, 2)
        preStateDim = count[0]
        # For ease of code, we always have at least one input dimension
        inputDim = count[1] if len(count) > 1 else 1
        # postStateDim is computed by inputArgsLength as a tuple, extract the value
        # MATLAB: postStateDim is a scalar, Python: it's a tuple, take first element
        if isinstance(postStateDim_tuple, (tuple, list)) and len(postStateDim_tuple) > 0:
            postStateDim = postStateDim_tuple[0]
        elif isinstance(postStateDim_tuple, (int, np.integer)):
            postStateDim = int(postStateDim_tuple)
        else:
            # If inputArgsLength didn't compute it, evaluate function to determine output size
            try:
                # Create dummy inputs based on computed dimensions
                x_dummy = np.zeros((preStateDim, 1))
                u_dummy = np.zeros((inputDim, 1))
                f_out = f(x_dummy, u_dummy)
                f_out = np.asarray(f_out)
                postStateDim = len(f_out.flatten())
            except Exception:
                raise CORAerror('CORA:wrongInputInConstructor',
                              'Could not determine post-state dimension from function handle.')
    
    # Input dimension must also be >= 1
    inputDim = max(inputDim, 1) if inputDim is not None else 1
    
    # Ensure postStateDim is not None and is an integer
    if postStateDim is None:
        raise CORAerror('CORA:wrongInputInConstructor',
                      'Post-state dimension must be provided or computable from function handle.')
    postStateDim = int(postStateDim)  # Ensure it's an integer
    
    return f, preStateDim, inputDim, postStateDim

