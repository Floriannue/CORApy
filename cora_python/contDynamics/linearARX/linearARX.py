import numpy as np
from cora_python.contDynamics.contDynamics import ContDynamics
from cora_python.g.macros.CHECKS_ENABLED import CHECKS_ENABLED
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.check.assertNarginConstructor import assertNarginConstructor

class LinearARX(ContDynamics):
    """
    linearARX - object constructor for linear discrete-time ARX systems

    Generates a discrete-time linear ARX object according to the
    following equation:
       y(k) =  sum_{i=1}^p A_bar{i} y(k-i) +
               sum_{i=1}^{p+1} B_bar{i} u(k-i+1)

    Syntax:
        linARX = LinearARX(A_bar, B_bar, dt)
        linARX = LinearARX(A_bar, B_bar, dt, name) # name is optional

    Inputs:
        A_bar (list of np.ndarray): output parameters
        B_bar (list of np.ndarray): input parameters
        dt (float): sampling time
        name (str, optional): name of system

    Outputs:
        linARX - LinearARX object
    """

    def __init__(self, *args):
        
        # parse input arguments
        name, A_bar, B_bar, dt = self._parse_input_args(*args)

        # check correctness of input arguments
        if CHECKS_ENABLED:
            self._check_input_args(name, A_bar, B_bar, dt, len(args))

        # compute properties
        n_p, outputs, inputs = self._compute_properties(A_bar, B_bar)

        # instantiate parent class
        super().__init__(name, 0, inputs, outputs)

        # assign object properties
        self.A_bar = A_bar
        self.B_bar = B_bar
        self.dt = dt
        self.tvp = False
        self.n_p = n_p
        self.conv_tvp = None
        self.A_tilde = None
        self.B_tilde = None

    def __repr__(self):
        return self.display()

    def _parse_input_args(self, *args):
        
        assertNarginConstructor([3, 4], len(args))

        if len(args) == 4 and isinstance(args[0], str):
            name = args[0]
            A_bar = args[1]
            B_bar = args[2]
            dt = args[3]
        elif len(args) == 3 and isinstance(args[0], list):
            name = 'linearARX'
            A_bar = args[0]
            B_bar = args[1]
            dt = args[2]
        else:
            # This case should not be reachable due to assertNarginConstructor
            # and the preceding checks, but as a safeguard:
            raise ValueError("Invalid combination of input arguments.")

        return name, A_bar, B_bar, dt

    def _check_input_args(self, name, A_bar, B_bar, dt, n_in):
        # No checks needed if name is not default, assuming properties were checked before
        if name == 'linearARX':
            inputArgsCheck([
                (name, 'att', ['char', 'string']),
                (A_bar, 'att', 'cell', 'nonempty'),
                (B_bar, 'att', 'cell', 'nonempty'),
                (dt, 'att', 'numeric', 'scalar')
            ])
        else:
            inputArgsCheck([
                (A_bar, 'att', 'cell', 'nonempty'),
                (B_bar, 'att', 'cell', 'nonempty'),
                (dt, 'att', 'numeric', 'scalar')
            ])

    def _compute_properties(self, A_bar, B_bar):
        n_p = len(A_bar)
        
        if n_p == 0:
            # This case is handled by inputArgsCheck, but as a fallback:
            outputs = 0
            inputs = B_bar[0].shape[1] if B_bar else 0
            return n_p, outputs, inputs
        
        # all matrices in A_bar must be square and have the same dimensions
        dim_y = A_bar[0].shape[0]
        if A_bar[0].shape[1] != dim_y:
            raise ValueError("Matrices in A_bar must be square.")
        
        for A in A_bar:
            if A.shape[0] != dim_y or A.shape[1] != dim_y:
                raise ValueError("All matrices in A_bar must have the same dimensions.")
        
        outputs = dim_y
        
        # check B_bar
        dim_u = B_bar[0].shape[1]
        for B in B_bar:
            if B.shape[0] != dim_y:
                raise ValueError("Row dimension of matrices in B_bar must match dimension of matrices in A_bar.")
            if B.shape[1] != dim_u:
                raise ValueError("All matrices in B_bar must have the same number of columns.")

        if len(B_bar) != n_p + 1:
            raise ValueError("Length of B_bar must be length of A_bar + 1.")
        
        inputs = dim_u
        
        return n_p, outputs, inputs 