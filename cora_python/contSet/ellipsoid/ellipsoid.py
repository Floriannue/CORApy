import numpy as np
from typing import Union, List, Any

from cora_python.contSet.contSet.contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.assertNarginConstructor import assertNarginConstructor
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.macros.CHECKS_ENABLED import CHECKS_ENABLED
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.check.isApproxSymmetric import isApproxSymmetric


class Ellipsoid(ContSet):
    """
    ellipsoid - object constructor for ellipsoids
    Some ideas for this class have been extracted from [1].

    Description:
       This class represents ellipsoid objects defined as
       {x | (x - q)' * Q^(-1) (x - q) <= 1}
       in the non-degenerate case, and are defined using the support function
       of ellipsoids (see ellipsoid/supportFunc.m for details) in the general
       case.

    Syntax:
       Ellipsoid(E)
       Ellipsoid(Q)
       Ellipsoid(Q,q)
       Ellipsoid(Q,q,TOL)

    Inputs:
       E - ellipsoid object
       Q - square, positive semi-definite shape matrix
       q - center vector
       TOL - tolerance

    Outputs:
       obj - generated Ellipsoid object

    Example:
       Q = np.array([[2.7, -0.2], [-0.2, 2.4]]);
       q = np.array([[1], [2]]);
       E = Ellipsoid(Q, q);

    Other m-files required: none
    Subfunctions: none
    MAT-files required: none

    See also: none

    Authors:       Victor Gassmann, Matthias Althoff
    Written:       13-March-2019
    Last update:   16-October-2019
                   02-May-2020 (MW, add property validation)
                   29-March-2021 (MA, faster eigenvalue computation)
                   14-December-2022 (TL, property check in inputArgsCheck)
                   16-June-2023 (MW, restructure using auxiliary functions)
    Automatic python translation: Florian Nüssel BA 2025
    """

    def __init__(self, *args, **kwargs):

        # 0. avoid empty instantiation
        if not args:
            raise CORAerror('CORA:noInputInSetConstructor')

        # Handle Ellipsoid.empty(n) case specifically (with or without TOL)
        if (len(args) in [2, 3]) and isinstance(args[0], np.ndarray) and args[0].size == 0 and \
           isinstance(args[1], np.ndarray) and args[1].shape[1] == 0:
            self.Q = np.zeros((0, 0))
            self.q = args[1] # This is already n x 0
            self.TOL = args[2] if len(args) == 3 else 1e-6  # Use provided TOL or default
            self.precedence = 50
            return

        assertNarginConstructor([1, 2, 3], len(args))

        # 1. copy constructor
        if len(args) == 1 and isinstance(args[0], Ellipsoid):
            obj = args[0]
            self.Q = obj.Q
            self.q = obj.q
            self.TOL = obj.TOL
            self.precedence = obj.precedence
            return

        # 1.5. contSet constructor - if first argument is a contSet object, call its ellipsoid method
        if len(args) == 1 and isinstance(args[0], ContSet) and not isinstance(args[0], Ellipsoid):
            # Call the ellipsoid method of the contSet object
            E = args[0].ellipsoid()
            self.Q = E.Q
            self.q = E.q
            self.TOL = E.TOL
            self.precedence = E.precedence
            return

        # 2. parse input arguments: varargin -> vars
        Q, q, TOL = self._aux_parseInputArgs(*args, **kwargs)
        # Allow empty TOL ([]) and default it as in MATLAB behavior
        if isinstance(TOL, np.ndarray) and TOL.size == 0:
            TOL = 1e-6
        
        # Call input argument check
        self._aux_checkInputArgs(Q, q, TOL)

        # 4. compute properties
        Q, q = self._aux_computeProperties(Q, q)

        # 5. assign properties
        self.Q = Q
        self.q = q
        self.TOL = TOL

        # 6. set precedence (fixed)
        self.precedence = 50

    # Auxiliary functions (will be moved to separate files)
    def _aux_parseInputArgs(self, *varargin, **kwargs):
        """
        _aux_parseInputArgs - helper function to parse input arguments and set default values

        Syntax:
            [Q,q,TOL] = _aux_parseInputArgs(varargin)

        Inputs:
            varargin - arguments of constructor

        Outputs:
            Q - shape matrix
            q - center
            TOL - tolerance

        Other m-files required: none
        Subfunctions: none
        MAT-files required: none

        See also: setDefaultValues.m

        Author:         Matthias Althoff, S. Rakovic
        Written:        05-May-2016
        Last update:    26-February-2019
        Last revision:  11-August-2021
        Automatic python translation: Florian Nüssel BA 2025
        """

        # Extract main arguments: Q, q, TOL
        # Assume Q is the first argument, q the second, TOL the third if provided
        Q = varargin[0]
        q_passed = None
        TOL = 1e-6 # Default tolerance

        if len(varargin) > 1:
            q_passed = varargin[1]
        if len(varargin) > 2:
            TOL = varargin[2]

        # Ensure q_passed is a NumPy array and a column vector if it's provided
        if q_passed is not None:
            if isinstance(q_passed, list):
                q_passed = np.array(q_passed).reshape(-1, 1)
            elif isinstance(q_passed, np.ndarray) and q_passed.ndim == 1:
                q_passed = q_passed.reshape(-1, 1)
            elif not isinstance(q_passed, np.ndarray):
                # If it's not a list or ndarray, convert to ndarray. Assume scalar leads to 1x1.
                q_passed = np.array(q_passed).reshape(-1, 1)

        # Initialize default_q based on Q's dimension or provided q_passed dimension.
        if isinstance(Q, np.ndarray) and Q.size == 0:
            if q_passed is not None and isinstance(q_passed, np.ndarray) and q_passed.shape[1] == 0:
                default_q = q_passed # Use the passed n x 0 vector
            else:
                dim_from_q = q_passed.shape[0] if q_passed is not None and q_passed.size > 0 else 0
                default_q = np.zeros((dim_from_q, 1))
        elif isinstance(Q, np.ndarray):
            default_q = np.zeros((Q.shape[0], 1))
        else:
            default_q = np.array([]).reshape(0,0)

        # Pass q_passed directly to setDefaultValues; let it handle None or empty arrays.
        # This removes the complex logic for q_args_for_set_default.
        parsed_q = setDefaultValues([default_q], [q_passed] if q_passed is not None else [])
        q = parsed_q[0]

        # Ensure q is a column vector (d x 1) before returning (redundant with earlier check, but harmless)
        if isinstance(q, np.ndarray) and q.ndim == 1:
            q = q.reshape(-1, 1)
        elif isinstance(q, list):
            q = np.array(q).reshape(-1, 1)

        return Q, q, TOL

    def _aux_checkInputArgs(self, Q, q, TOL):
        """
        check correctness of input arguments
        """
        if CHECKS_ENABLED(): # CHECKS_ENABLED is a function in Python
            # allow empty Q matrix for ellipsoid.empty
            if Q.size == 0 and q.shape[1] == 0: # This case is for an empty ellipsoid
                # Removed 'return' statement to allow further checks
                pass
            elif Q.size == 0 and q.size != 0: # This case is for a point ellipsoid
                # For a point ellipsoid, q determines the dimension. Q must be 0x0 but consistent with q.
                if Q.shape != (0,0):
                    raise CORAerror('CORA:wrongInputInConstructor', 'For a point ellipsoid, Q must be an empty 0x0 matrix.')
                # The remaining checks still apply for point ellipsoids if q is non-empty.
                # However, many checks below rely on Q.shape[0] > 0 which is not true here.
                # We need specific checks for point ellipsoids, or modify existing ones.
                # For now, let's allow it to proceed and see if later checks in _aux_checkInputArgs handle it.
                pass

            # inputArgsCheck expects a list of lists, where each inner list describes an argument.
            # Example: [arg_value, 'attribute', 'type', ['constraint1', 'constraint2']]
            inputArgsCheck([
                [Q, 'att', 'numeric', ['finite', 'matrix']],
                [q, 'att', 'numeric', ['finite', 'column']],
                [TOL, 'att', 'numeric', ['nonnegative', 'scalar']],
            ])

            # shape matrix needs to be square (only if Q is not empty 0x0)
            if Q.size > 0 and (Q.shape[0] != Q.shape[1]):
                raise CORAerror('CORA:wrongInputInConstructor', \
                                'The shape matrix needs to be a square matrix.')

            # check dimensions (only if Q is not empty 0x0 and q is not empty n x 0)
            if Q.size > 0 and q.size != 0 and Q.shape[0] != q.shape[0]:
                raise CORAerror('CORA:wrongInputInConstructor', \
                                'Dimensions of the shape matrix and center are different.')
            
            # Check for positive semi-definite and symmetric (only if Q is not empty 0x0)
            if Q.size > 0:
                mev = np.min(np.linalg.eigvalsh(Q))
                if not isApproxSymmetric(Q, TOL) or mev < -TOL:
                    raise CORAerror('CORA:wrongInputInConstructor', \
                                    'The shape matrix needs to be positive semidefinite/symmetric.')

    def _aux_computeProperties(self, Q, q):
        """
        returns zero values in case Q or q are empty
        """
        # If Q is 0x0 and q is n x 0, it indicates an empty ellipsoid.
        if isinstance(Q, np.ndarray) and Q.size == 0 and isinstance(q, np.ndarray) and q.shape[1] == 0:
            return np.array([]).reshape(0,0), q # Ensure Q remains 0x0
        # If Q is 0x0 but q is non-empty (point ellipsoid).
        elif isinstance(Q, np.ndarray) and Q.size == 0 and isinstance(q, np.ndarray) and q.size != 0: 
            return np.array([]).reshape(0,0), q # Ensure Q remains 0x0
        return Q, q
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Handle numpy ufuncs on ellipsoids.
        This ensures that operations like @ (matmul) are handled by the ellipsoid's own methods.
        """
        if ufunc == np.matmul and method == '__call__':
            # Handle matrix multiplication
            if len(inputs) == 2:
                left, right = inputs
                if left is self:
                    # self @ other
                    return self.__matmul__(right)
                elif right is self:
                    # other @ self
                    return self.__rmatmul__(left)
        return NotImplemented

    def __repr__(self) -> str:
        """
        Official string representation for programmers.
        Should be unambiguous and allow object reconstruction.
        """
        try:
            if self.is_empty():
                return f"Ellipsoid.empty({self.dim()})"
            else:
                # For small ellipsoids, show the actual values
                if self.Q.size <= 9 and self.q.size <= 3:
                    if np.allclose(self.q, 0):
                        return f"Ellipsoid({self.Q.tolist()})"
                    else:
                        return f"Ellipsoid({self.Q.tolist()}, {self.q.flatten().tolist()})"
                else:
                    return f"Ellipsoid(dim={self.dim()})"
        except:
            return "Ellipsoid()"

