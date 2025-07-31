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
        q_passed = varargin[1] if len(varargin) > 1 else None
        TOL = varargin[2] if len(varargin) > 2 else None

        # Remaining arguments (if any) are passed to setDefaultValues for q
        q_args = []
        if len(varargin) > 1:
            q_args = varargin[1:]

        # Default values for q
        # The MATLAB default for TOL is 1e-6, for q it's zeros(size(Q,1),1)
        # We need to handle this dynamically as q's default depends on Q.
        # setDefaultValues handles arbitrary number of arguments for q.
        # The structure is [default_q]

        # Initialize default_q based on Q's size.
        # If Q is empty (0x0), the q should be an n x 0 zero vector (from ellipsoid.empty)
        # Otherwise, if Q is a matrix, then q should be a column vector of its row dimension
        if Q.size == 0:
            # If Q is empty, and q_passed is also empty (n x 0 from empty() constructor),
            # then default_q should reflect that dimension (n x 0).
            # Otherwise, it's a 0x0 empty matrix for a default.
            if q_passed is not None and q_passed.shape[1] == 0: # Check if it's an n x 0 empty vector
                default_q = q_passed # Use the passed n x 0 vector
            else:
                default_q = np.zeros((0, 0)) # Default to 0x0 for q if Q is 0x0 and q is not n x 0
        else:
            default_q = np.zeros((Q.shape[0], 1))

        # Pass q_args to setDefaultValues for processing.
        # If q_passed was provided (not None), we include it in q_args
        # so setDefaultValues can handle it.
        if q_passed is not None:
            # If q_passed is already handled by our specific empty logic, don't pass it again
            # as a general argument, otherwise setDefaultValues might misinterpret it.
            # Only pass if it's not the case of an n x 0 empty vector which is our default.
            if not (Q.size == 0 and q_passed.shape[1] == 0):
                q_args_for_set_default = [q_passed] + list(varargin[2:]) # varargin[2:] are the actual q args if q_passed is varargin[1]
            else:
                q_args_for_set_default = list(varargin[2:])
        else:
            q_args_for_set_default = list(varargin[1:]) # if q_passed is None, then varargin[1:] are actual q args

        parsed_q, _ = setDefaultValues([default_q], q_args_for_set_default)
        q = parsed_q[0]

        # Ensure q is a column vector (d x 1) before returning
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
            if Q.size == 0:
                return # If Q is empty, we trust _aux_computeProperties to handle it.

            # inputArgsCheck expects a list of lists, where each inner list describes an argument.
            # Example: [arg_value, 'attribute', 'type', ['constraint1', 'constraint2']]
            inputArgsCheck([
                [Q, 'att', 'numeric', ['finite', 'matrix']],
                [q, 'att', 'numeric', ['finite', 'column']],
                [TOL, 'att', 'numeric', ['nonnegative', 'scalar']],
            ])

            # shape matrix needs to be square
            if Q.shape[0] != Q.shape[1]:
                raise CORAerror('CORA:wrongInputInConstructor', \
                                'The shape matrix needs to be a square matrix.')

            # check dimensions
            if q.size != 0 and Q.shape[0] != q.shape[0]:
                raise CORAerror('CORA:wrongInputInConstructor', \
                                'Dimensions of the shape matrix and center are different.')
            
            # Check for positive semi-definite and symmetric
            # Using np.linalg.eigvalsh for symmetric matrices for better stability
            # Only check if Q is not empty (e.g. for Ellipsoid.empty() cases where Q is 0x0)
            if Q.size > 0:
                mev = np.min(np.linalg.eigvalsh(Q))
                if not isApproxSymmetric(Q, TOL) or mev < -TOL:
                    raise CORAerror('CORA:wrongInputInConstructor', \
                                    'The shape matrix needs to be positive semidefinite/symmetric.')

    def _aux_computeProperties(self, Q, q):
        """
        returns zero values in case Q or q are empty
        """
        # If Q is 0x0, it indicates an empty ellipsoid (or a point ellipsoid if q is non-empty)
        # We should not force q to be empty if Q is 0x0 and q is provided and non-empty.
        # Instead, the dimension should be handled.
        # The dimensions of Q and q are checked in _aux_checkInputArgs.
        # The only thing left here is to ensure if Q is 0x0 and q is also 0x0,
        # they are correctly represented as empty arrays in Python.
        if Q.size == 0 and q.shape[1] == 0:
            # If Q is 0x0 and q is n x 0, it's an empty ellipsoid. 
            # We ensure Q remains 0x0 and q remains n x 0 (empty in MATLAB sense).
            Q = np.zeros((0, 0))
            pass # q remains n x 0, which is empty in MATLAB sense
        elif Q.size == 0 and q.size != 0: 
            # Q is 0x0 but q is non-empty (point ellipsoid).
            pass # Keep Q as 0x0, q as is
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

