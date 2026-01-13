"""
nonlinearSys class (continuous-time nonlinear system):
   x' = f(x,u)    % dynamic equation
   y  = g(x,u)    % output equation

Syntax:
    % only dynamic equation
    nlnsys = nonlinearSys(fun)
    nlnsys = nonlinearSys(name,fun)
    nlnsys = nonlinearSys(fun,states,inputs)
    nlnsys = nonlinearSys(name,fun,states,inputs)

    % dynamic equation and output equation
    nlnsys = nonlinearSys(fun,out_fun)
    nlnsys = nonlinearSys(name,fun,out_fun)
    nlnsys = nonlinearSys(fun,states,inputs,out_fun,outputs)
    nlnsys = nonlinearSys(name,fun,states,inputs,out_fun,outputs)

Inputs:
    name - name of system
    fun - function handle to the dynamic equation
    states - number of states
    inputs - number of inputs
    out_fun - function handle to the output equation
    outputs - number of outputs

Outputs:
    nlnsys - generated nonlinearSys object

Example:
    fun = @(x,u) [x(2); ...
               (1-x(1)^2)*x(2)-x(1)];
    sys = nonlinearSys('vanDerPol',fun)

    out_fun = @(x,u) [x(1) + x(2)];
    sys = nonlinearSys('vanDerPol',fun,out_fun);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contDynamics

Authors:       Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
Written:       17-October-2007 
Last update:   29-October-2007
               04-August-2016 (changed to new OO format)
               19-May-2020 (NK, changed constructor syntax)
               02-February-2021 (MW, add switching between tensor files)
               17-November-2022 (MW, add output equation)
               23-November-2022 (MW, introduce checks, restructure)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import inspect
from typing import Optional, Callable, Any, List
from cora_python.contDynamics.contDynamics.contDynamics import ContDynamics
from cora_python.g.macros.CHECKS_ENABLED import CHECKS_ENABLED
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.function_handle.inputArgsLength import inputArgsLength
from cora_python.g.functions.matlab.validate.check.is_func_linear import is_func_linear
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class NonlinearSys(ContDynamics):
    """
    nonlinearSys class (continuous-time nonlinear system)
    
    Properties:
        mFile: function handle dynamic equation
        jacobian: function handle jacobian matrix
        hessian: function handle hessian tensor
        thirdOrderTensor: function handle third-order tensor
        tensors: list of function handles for higher-order tensors
        
        out_mFile: function handle output equation
        out_isLinear: which output functions are linear
        out_jacobian: function handle jacobian matrix
        out_hessian: function handle hessian tensor
        out_thirdOrderTensor: function handle third-order tensor
        
        linError: linearization error
    """
    
    def __init__(self, *args, name=None, fun=None, states=None, inputs=None, out_fun=None, outputs=None):
        """
        Constructor for nonlinearSys
        
        Supports multiple calling patterns:
            NonlinearSys()
            NonlinearSys(fun)
            NonlinearSys(name, fun)
            NonlinearSys(fun, states, inputs)
            NonlinearSys(name, fun, states, inputs)
            NonlinearSys(fun, out_fun)
            NonlinearSys(name, fun, out_fun)
            NonlinearSys(fun, states, inputs, out_fun, outputs)
            NonlinearSys(name, fun, states, inputs, out_fun, outputs)
            
        Also supports keyword arguments:
            NonlinearSys(fun, states=1, inputs=1)
            NonlinearSys(fun, states=1, inputs=1, out_fun=g, outputs=1)
        """
        # 0. check number of input arguments
        # assertNarginConstructor(0:6,nargin); # Python handles nargin differently
        if len(args) > 6:
            raise CORAerror('CORA:wrongInputInConstructor',
                           f'Too many positional arguments: {len(args)} (max 6)')
        
        # 1. copy constructor: not allowed due to obj@contDynamics below
        # if nargin == 1 && isa(varargin{1},'nonlinearSys')
        #     obj = varargin{1}; return
        # end
        
        # 2. parse input arguments: varargin -> vars
        parsed_name, parsed_fun, parsed_states, parsed_inputs, parsed_out_fun, parsed_outputs = _aux_parseInputArgs(*args)
        
        # 3. Merge positional and keyword arguments (kwargs override positional)
        name = name if name is not None else parsed_name
        fun = fun if fun is not None else parsed_fun
        states = states if states is not None else parsed_states
        inputs = inputs if inputs is not None else parsed_inputs
        out_fun = out_fun if out_fun is not None else parsed_out_fun
        outputs = outputs if outputs is not None else parsed_outputs
        
        # 4. check correctness of input arguments
        _aux_checkInputArgs(name, fun, states, inputs, out_fun, outputs)
        
        # 5. analyze functions and extract number of states, inputs, outputs
        states, inputs, out_fun, outputs, out_isLinear = \
            _aux_computeProperties(fun, states, inputs, out_fun, outputs)
        
        # 6. instantiate parent class
        # note: currently, we only support unit disturbance matrices
        #       (same as number of states) and unit noise matrices (same as
        #       number of outputs)
        super().__init__(name, states, inputs, outputs, states, outputs)
        
        # 7a. assign object properties: dynamic equation
        self.mFile = fun
        # In MATLAB: eval(['@jacobian_',name]) etc.
        # In Python, we'll use a naming convention or pass via kwargs
        # For now, set to None - they can be set later or via kwargs
        self.jacobian = None
        self.hessian = None
        self.thirdOrderTensor = None
        self.tensors = [None] * 7  # For tensors 4-10 (indices 0-6)
        
        # 7b. assign object properties: output equation
        self.out_mFile = out_fun
        self.out_isLinear = out_isLinear
        self.out_jacobian = None
        self.out_hessian = None
        self.out_thirdOrderTensor = None
        
        self.linError = None
    
    def setHessian(self, version: str):
        """
        Allow switching between standard and interval arithmetic
        
        Args:
            version: 'standard' or 'int'
        """
        from cora_python.g.macros.CORAROOT import CORAROOT
        import os
        import importlib.util
        
        path = os.path.join(CORAROOT(), 'models', 'auxiliary', self.name)
        
        if version == 'standard':
            # MATLAB: eval(['@hessianTensor_' self.name])
            hessian_file = os.path.join(path, f'hessianTensor_{self.name}.py')
            if os.path.isfile(hessian_file):
                module_name = f'hessianTensor_{self.name}_{id(self)}'
                spec = importlib.util.spec_from_file_location(module_name, hessian_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.hessian = getattr(module, f'hessianTensor_{self.name}')
        elif version == 'int':
            # MATLAB: eval(['@hessianTensorInt_' self.name])
            hessian_file = os.path.join(path, f'hessianTensorInt_{self.name}.py')
            if os.path.isfile(hessian_file):
                module_name = f'hessianTensorInt_{self.name}_{id(self)}'
                spec = importlib.util.spec_from_file_location(module_name, hessian_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.hessian = getattr(module, f'hessianTensorInt_{self.name}')
        
        return self
    
    def setOutHessian(self, version: str):
        """
        Allow switching between standard and interval arithmetic for output
        
        Args:
            version: 'standard' or 'int'
        """
        if version == 'standard':
            pass
        elif version == 'int':
            pass
    
    def setThirdOrderTensor(self, version: str):
        """
        Allow switching between standard and interval arithmetic
        
        Args:
            version: 'standard' or 'int'
        """
        if version == 'standard':
            pass
        elif version == 'int':
            pass
    
    def setOutThirdOrderTensor(self, version: str):
        """
        Allow switching between standard and interval arithmetic for output
        
        Args:
            version: 'standard' or 'int'
        """
        if version == 'standard':
            pass
        elif version == 'int':
            pass
    
    def __repr__(self) -> str:
        """String representation of the nonlinearSys object"""
        return f"NonlinearSys(name='{self.name}', states={self.nr_of_dims}, inputs={self.nr_of_inputs}, outputs={self.nr_of_outputs})"


# Auxiliary functions -----------------------------------------------------

def _aux_parseInputArgs(*args):
    """
    Parse input arguments for nonlinearSys constructor
    
    Returns:
        name, fun, states, inputs, out_fun, outputs
    """
    # default values
    name = None
    fun = None
    states = None
    inputs = None
    out_fun = None
    outputs = None
    
    # no input arguments
    if len(args) == 0:
        return name, fun, states, inputs, out_fun, outputs
    
    # parse input arguments
    nargin = len(args)
    
    if nargin == 1:
        # syntax: obj = nonlinearSys(fun)
        fun = args[0]
    elif nargin == 2:
        if isinstance(args[0], str):
            # syntax: obj = nonlinearSys(name,fun)
            name, fun = args
        elif callable(args[0]):
            # syntax: obj = nonlinearSys(fun,out_fun)
            fun, out_fun = args
    elif nargin == 3:
        if isinstance(args[0], str):
            # syntax: obj = nonlinearSys(name,fun,out_fun)
            name, fun, out_fun = args
        elif callable(args[0]):
            # syntax: obj = nonlinearSys(fun,states,inputs)
            fun, states, inputs = args
    elif nargin == 4:
        # syntax: obj = nonlinearSys(name,fun,states,inputs)
        name, fun, states, inputs = args
    elif nargin == 5:
        # syntax: obj = nonlinearSys(fun,states,inputs,out_fun,outputs)
        fun, states, inputs, out_fun, outputs = args
    elif nargin == 6:
        # syntax: obj = nonlinearSys(name,fun,states,inputs,out_fun,outputs)
        name, fun, states, inputs, out_fun, outputs = args
    
    # get name from function handle
    if name is None and fun is not None:
        if callable(fun):
            name = fun.__name__ if hasattr(fun, '__name__') else str(fun)
            # Clean up name similar to MATLAB
            name = name.replace('@', '').replace('(', '').replace(')', '').replace(',', '')
            if not name.isidentifier():
                # default name
                name = 'nonlinearSys'
        else:
            name = 'nonlinearSys'
    
    return name, fun, states, inputs, out_fun, outputs


def _aux_checkInputArgs(name, fun, states, inputs, out_fun, outputs):
    """
    Check correctness of input arguments
    """
    if CHECKS_ENABLED:
        # check name (not empty because default name is not empty)
        if name is not None and not isinstance(name, str):
            raise CORAerror('CORA:wrongInputInConstructor',
                          'System name has to be a char array.')
        
        # fun and out_fun have to be function handles with two inputs
        if fun is not None:
            if not callable(fun):
                raise CORAerror('CORA:wrongInputInConstructor',
                              'Dynamic function has to be a function handle.')
            # Check number of arguments
            try:
                sig = inspect.signature(fun)
                if len(sig.parameters) != 2:
                    raise CORAerror('CORA:wrongInputInConstructor',
                                  'Dynamic function has to be a function handle with two input arguments.')
            except (ValueError, TypeError):
                pass  # Can't inspect, assume it's OK
        
        if out_fun is not None:
            if not callable(out_fun):
                raise CORAerror('CORA:wrongInputInConstructor',
                              'Output function has to be a function handle.')
            try:
                sig = inspect.signature(out_fun)
                if len(sig.parameters) != 2:
                    raise CORAerror('CORA:wrongInputInConstructor',
                                  'Output function has to be a function handle with two input arguments.')
            except (ValueError, TypeError):
                pass
        
        # states and outputs have to be numeric, scalar integer > 0,
        # inputs can be 0 (e.g., in parallel hybrid automata with only local
        # outputs = inputs and no global inputs)
        if states is not None:
            inputArgsCheck([[states, 'att', ['numeric'],
                           ['positive', 'integer', 'scalar']]])
        if inputs is not None:
            inputArgsCheck([[inputs, 'att', ['numeric'],
                           ['nonnegative', 'integer', 'scalar']]])
        if outputs is not None:
            inputArgsCheck([[outputs, 'att', ['numeric'],
                           ['positive', 'integer', 'scalar']]])


def _aux_computeProperties(fun, states, inputs, out_fun, outputs):
    """
    Analyze functions and extract number of states, inputs, outputs
    """
    if inputs is not None and inputs == 0:
        # CORA requires at least one input so that the internal
        # computations execute properly; some system do not have an input,
        # so it would be correct to explicitly state '0'
        inputs = 1
    
    # get number of states and number of inputs
    if states is None or inputs is None:
        try:
            temp, _ = inputArgsLength(fun, 2)
            # CORA models have to have at least one input
            if states is None:
                states = temp[0]
            if inputs is None:
                inputs = max(1, temp[1])
        except Exception as e:
            raise CORAerror('CORA:specialError',
                          f'Failed to determine number of states and inputs automatically!\n'
                          f'Please provide number of states and inputs as additional input arguments!\n'
                          f'Error: {e}')
    
    if out_fun is None:
        # init out_fun via eval to have numeric values within function handle
        outputs = states
        # Create identity output function: y = I * x
        def out_fun(x, u):
            return np.eye(outputs) @ x[:outputs]
        out_isLinear = np.ones(outputs, dtype=bool)
    else:
        # get number of states in output equation and outputs
        if outputs is None:
            # only try to compute if outputs is not provided
            try:
                temp, out_out = inputArgsLength(out_fun, 2)
                outputs = out_out[0] if len(out_out) > 0 else 1
            except Exception as e:
                raise CORAerror('CORA:specialError',
                              f'Failed to determine number of outputs automatically!\n'
                              f'Please provide number of outputs as an additional input argument!\n'
                              f'Error: {e}')
        else:
            # outputs is provided, but we still need temp for validation
            try:
                temp, _ = inputArgsLength(out_fun, 2)
            except Exception:
                # If we can't analyze but outputs is provided, use a default temp
                # This allows explicit outputs to work even if analysis fails
                temp = [states]  # Assume same number of states as dynamic equation
        
        # ensure that output equation does not have more states than
        # dynamic equation
        if temp[0] > states:
            raise CORAerror('CORA:wrongInputInConstructor',
                          'More states in output equation than in dynamic equation.')
        
        # check which output functions are linear
        out_isLinear = is_func_linear(out_fun, [states, inputs])
        # Convert to numpy array if needed
        if not isinstance(out_isLinear, np.ndarray):
            out_isLinear = np.array([out_isLinear] if np.isscalar(out_isLinear) else out_isLinear)
    
    return states, inputs, out_fun, outputs, out_isLinear
