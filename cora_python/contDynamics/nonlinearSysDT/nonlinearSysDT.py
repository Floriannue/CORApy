import numpy as np
import sympy
from cora_python.contDynamics.contDynamics import ContDynamics
from cora_python.g.macros.CHECKS_ENABLED import CHECKS_ENABLED
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.function_handle.input_args_length import input_args_length
from cora_python.g.functions.matlab.validate.check.is_func_linear import is_func_linear
import inspect
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

class NonlinearSysDT(ContDynamics):
    """
    nonlinearSysDT class (time-discrete nonlinear system)
       x_k+1 = f(x_k,u_k)
       y_k   = g(x_k,u_k)

    Syntax:
        nlnsysDT = NonlinearSysDT(fun, dt, states, inputs)
        nlnsysDT = NonlinearSysDT(fun, dt, states, inputs, name="mySys")
        nlnsysDT = NonlinearSysDT(fun, dt, states, inputs, out_fun, outputs)
        ... and other combinations

    Inputs:
        fun (callable): function handle to the dynamic equation
        dt (float): sampling time
        states (int): number of states
        inputs (int): number of inputs
        name (str, optional): name of dynamics
        out_fun (callable, optional): function handle to the output equation
        outputs (int, optional): number of outputs
        ... other optional tensor functions
    """

    def __init__(self, *args, **kwargs):
        # 1. Parse input arguments
        name, fun, dt, states, inputs, out_fun, outputs = self._parse_input_args(args)

        # Update with keyword arguments
        name = kwargs.get('name', name)
        fun = kwargs.get('fun', fun)
        dt = kwargs.get('dt', dt)
        states = kwargs.get('states', states)
        inputs = kwargs.get('inputs', inputs)
        out_fun = kwargs.get('out_fun', out_fun)
        outputs = kwargs.get('outputs', outputs)

        # 2. Check correctness of input arguments
        if CHECKS_ENABLED:
            self._check_input_args(name, fun, dt, states, inputs, out_fun, outputs)

        # 3. Compute properties
        states, inputs, out_fun, outputs, out_is_linear, rewrite_as_C, C = \
            self._compute_properties(fun, states, inputs, out_fun, outputs)
        
        # 4. Instantiate parent class
        super().__init__(name, states, inputs, outputs, states, outputs)
        
        # 5a. Assign object properties: dynamic equation
        self.dt = dt
        self.mFile = fun
        is_dynamic_linear = is_func_linear(fun, [states, inputs]) if fun else False
        self.jacobian = kwargs.get('jacobian') # In MATLAB this is handled by eval
        self.hessian = kwargs.get('hessian')
        self.thirdOrderTensor = kwargs.get('thirdOrderTensor')
        self.linError = None

        # 5b. Assign object properties: output equation
        self.out_mFile = out_fun
        self.out_isLinear = out_is_linear
        
        self.isLinear = is_dynamic_linear and self.out_isLinear

        if np.all(out_is_linear) and rewrite_as_C:
             C = self._rewrite_out_fun_as_matrix(out_fun, states, outputs)

        self.C = C
        self.out_jacobian = kwargs.get('out_jacobian')
        self.out_hessian = kwargs.get('out_hessian')
        self.out_thirdOrderTensor = kwargs.get('out_thirdOrderTensor')

    def __repr__(self):
        return self.display()

    def _parse_input_args(self, args):
        # Default values
        name, fun, dt, states, inputs, out_fun, outputs = None, None, 0, None, None, None, None

        if not args:
            return name, fun, dt, states, inputs, out_fun, outputs

        nargin = len(args)
        
        if nargin == 2:
            fun, dt = args
        elif nargin == 3:
            if isinstance(args[0], str):
                name, fun, dt = args
            elif callable(args[0]):
                fun, dt, out_fun = args
        elif nargin == 4:
            if isinstance(args[0], str):
                name, fun, dt, out_fun = args
            elif callable(args[0]):
                fun, dt, states, inputs = args
        elif nargin == 5:
            name, fun, dt, states, inputs = args
        elif nargin == 6:
            fun, dt, states, inputs, out_fun, outputs = args
        elif nargin == 7:
            name, fun, dt, states, inputs, out_fun, outputs = args
        else:
             # Using a custom error class would be better
            raise CORAerror("Invalid number of arguments for NonlinearSysDT constructor")


        if name is None and fun is not None:
            name = fun.__name__.replace('@','').replace('(','').replace(')','').replace(',','')
            if not name.isidentifier():
                name = 'nonlinearSysDT'
        
        return name, fun, dt, states, inputs, out_fun, outputs

    def _check_input_args(self, name, fun, dt, states, inputs, out_fun, outputs):
        if name is not None and not isinstance(name, str):
            raise TypeError("System name has to be a string.")
        if dt is not None and (not isinstance(dt, (int, float)) or dt <= 0):
             raise ValueError("Sampling time has to be a positive scalar.")
        if fun is not None and not callable(fun):
            raise TypeError("Dynamic function has to be a function handle.")
        if out_fun is not None and not callable(out_fun):
            raise TypeError("Output function has to be a function handle.")
        if states is not None:
            inputArgsCheck([[states, 'att', ['numeric'], ['positive', 'integer', 'scalar']]])
        if inputs is not None:
            inputArgsCheck([[inputs, 'att', ['numeric'], ['positive', 'integer', 'scalar']]])
        if outputs is not None:
            inputArgsCheck([[outputs, 'att', ['numeric'], ['positive', 'integer', 'scalar']]])

    def _compute_properties(self, fun, states, inputs, out_fun, outputs):
        
        C = None
        rewrite_as_C = False

        # If dimensions are not specified, try to infer them
        if states is None or inputs is None:
            try:
                # MATLAB's `nargin` for function handles is tricky. 
                # Python's `inspect` is a good equivalent.
                temp, _ = input_args_length(fun)
                if states is None: states = temp[0]
                # In MATLAB, if a function is defined as f(x), nargin is 1. We need to handle this.
                if inputs is None: inputs = temp[1] if len(temp) > 1 else 0
            except Exception as e:
                # Can't determine dimensions, raise an error
                raise RuntimeError(f"Could not determine system dimensions from the dynamic function. "
                                   f"Please specify 'states' and 'inputs' explicitly. Details: {e}")

        # Default output equation
        if out_fun is None:
            # y = x
            out_fun = lambda x, u: x
            if outputs is None:
                outputs = states
            out_is_linear = True
            rewrite_as_C = True
        else:
            # custom output equation
            try:
                temp_out, out_dims = input_args_length(out_fun)
                auto_outputs = out_dims[0] if out_dims else 0
                out_inputs = temp_out[1] if len(temp_out) > 1 else 0
                
                if outputs is None:
                    outputs = auto_outputs
                
                out_is_linear = is_func_linear(out_fun, [states, out_inputs])

                # Check if all outputs are linear functions of the state variables
                # (and not of the input variables)
                if out_is_linear and out_inputs == 0:
                    rewrite_as_C = True

            except Exception as e:
                raise RuntimeError(f"Could not process output function: {e}")
        
        return states, inputs, out_fun, outputs, out_is_linear, rewrite_as_C, C

    def _rewrite_out_fun_as_matrix(self, out_fun, states, outputs):
        x = sympy.symbols(f'x_1:{states+1}')
        C = np.zeros((outputs, states))

        for j in range(states):
            x_temp_list = [0] * states
            x_temp_list[j] = x[j]
            x_temp = sympy.Matrix(x_temp_list)
            
            try:
                # Assume u is not used for linear output eq, so pass None or empty list
                out_lhs = out_fun(x_temp, []) 
            except Exception as e:
                print(f"Warning: Could not symbolically evaluate out_fun. C matrix will be zero. Error: {e}")
                return np.zeros((outputs, states))

            for i in range(outputs):
                if out_lhs[i].has(x[j]):
                    # a bit of a trick to get the coefficient of x[j]
                    C[i,j] = float(sympy.poly(out_lhs[i], x[j]).all_coeffs()[0])
        return C 