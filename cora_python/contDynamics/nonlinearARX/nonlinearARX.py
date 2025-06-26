import numpy as np
from cora_python.contDynamics.contDynamics import ContDynamics
from cora_python.g.macros.CHECKS_ENABLED import CHECKS_ENABLED
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck

class NonlinearARX(ContDynamics):
    """
    nonlinearARX class (time-discrete nonlinear ARX model)

    Generates a discrete-time nonlinear ARX object (NARX) according
    to the following equation:
       y(k) = f(y(k-1),...,y(k-n_y),u(k),...,u(k-n_u),e(k-1),...,e(k-n_e))
               + e(k)

    Syntax:
        nlnARX = NonlinearARX(fun, dt, dim_y, dim_u, n_p)
        nlnARX = NonlinearARX(fun, dt, dim_y, dim_u, n_p, name="myNARX")

    Inputs:
        fun (callable): function handle for the NARX equation with arguments (y,u)
                        y = [y(k-1); ...; y(k-n_p)]: array dim_y x n_p
                        u = [u(k); ...; u(k-n_p)]: array dim_u x (n_p+1)
        dt (float): sampling time
        dim_y (int): dimension of the output
        dim_u (int): dimension of the input
        n_p (int): number of past time steps which are considered
        name (str, optional): name of the model
        jacobian (callable, optional): function for the jacobian matrix
        hessian (callable, optional): function for the hessian tensor
        thirdOrderTensor (callable, optional): function for the third-order tensor

    Outputs:
        nlnARX - generated NonlinearARX object
    """

    def __init__(self, *args, **kwargs):
        # 1. parse input arguments
        name, fun, dt, dim_y, dim_u, n_p = self._parse_input_args(*args)

        # 2. check correctness of input arguments
        if CHECKS_ENABLED:
            self._check_input_args(name, dt, dim_y, dim_u, n_p)

        # 3. compute properties
        name = str(name)

        # 4a. instantiate parent class
        super().__init__(name, 0, dim_u, dim_y)

        # 4b. assign object properties
        self.dt = dt
        self.n_p = n_p
        self.mFile = fun
        
        # A more Pythonic way to handle jacobian, etc. is to pass them in
        self.jacobian = kwargs.get('jacobian')
        self.hessian = kwargs.get('hessian')
        self.thirdOrderTensor = kwargs.get('thirdOrderTensor')
        
        self.prev_ID = 10
        
    def __repr__(self):
        return self.display()

    def _parse_input_args(self, *args):
        if isinstance(args[0], str):
            if len(args) != 6:
                raise TypeError(f"Expected 6 arguments when providing a name, but got {len(args)}")
            name, fun, dt, dim_y, dim_u, n_p = args
        else:
            if len(args) != 5:
                raise TypeError(f"Expected 5 arguments when not providing a name, but got {len(args)}")
            name = 'nonlinearARX'
            fun, dt, dim_y, dim_u, n_p = args
            
        return name, fun, dt, dim_y, dim_u, n_p

    def _check_input_args(self, name, dt, dim_y, dim_u, n_p):
        if name == 'nonlinearARX':
            inputArgsCheck([
                (name, 'att', ['char', 'string']),
                (dt, 'att', 'numeric', 'scalar'),
                (dim_y, 'att', 'numeric', ['scalar', 'nonnegative']),
                (dim_u, 'att', 'numeric', ['scalar', 'nonnegative']),
                (n_p, 'att', 'numeric', ['scalar', 'nonnegative'])
            ])
        else:
            inputArgsCheck([
                (dt, 'att', 'numeric', 'scalar'),
                (dim_y, 'att', 'numeric', ['scalar', 'nonnegative']),
                (dim_u, 'att', 'numeric', ['scalar', 'nonnegative']),
                (n_p, 'att', 'numeric', ['scalar', 'nonnegative'])
            ]) 