import numpy as np
from typing import Union, Optional, Tuple, Any

from cora_python.contSet.contSet.contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check import assertNarginConstructor
from cora_python.g.functions.matlab.validate.check import inputArgsCheck
from .dim import dim
from .isemptyobject import isemptyobject


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
    Last revision: 16-June-2023 (MW, restructure using auxiliary functions)
    """

    def __init__(self, *args):

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
        Q, q, TOL = self._aux_parseInputArgs(*args)

        # 3. check correctness of input arguments
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
    def _aux_parseInputArgs(self, *varargin):
        # parse input arguments from user and assign to variables
        Q = varargin[0]
        # setDefaultValues needs to be translated first
        # For now, let's manually handle the defaults.
        q = np.zeros((Q.shape[0], 1)) if len(varargin) < 2 else varargin[1]
        TOL = 1e-6 if len(varargin) < 3 else varargin[2]
        
        # Convert center vector to column vector if needed
        if q is not None and hasattr(q, 'shape'):
            q = np.asarray(q)
            if q.ndim == 1:
                # Convert 1D array to column vector
                q = q.reshape(-1, 1)
            elif q.ndim == 2 and q.shape[1] != 1:
                # If it's a row vector, transpose it (but not if it's empty like zeros(n,0))
                if q.shape[0] == 1 and q.shape[1] > 0:
                    q = q.T
        
        return Q, q, TOL

    def _aux_checkInputArgs(self, Q, q, TOL):
        # check correctness of input arguments
        CHECKS_ENABLED = True # This needs to be correctly handled from a global config

        if CHECKS_ENABLED:
            # allow empty Q matrix for ellipsoid.empty
            if Q.size == 0:
                # For empty ellipsoids, q should be zeros(n,0) which has size 0 in MATLAB terms
                # but preserves dimension information
                if q.size != 0:
                    raise CORAerror('CORA:wrongInputInConstructor',
                                    'Shape matrix is empty, but center is not.')
                return

            inputArgsCheck([
                [Q, 'att', ['numpy.ndarray'], ['finite', 'matrix']],
                [q, 'att', ['numpy.ndarray'], ['finite', 'column']],
                [TOL, 'att', ['numeric'], ['nonnegative', 'scalar']],
            ])

            # shape matrix needs to be square
            if Q.shape[0] != Q.shape[1]:
                raise CORAerror('CORA:wrongInputInConstructor',
                                'The shape matrix needs to be a square matrix.')

            # check dimensions
            if q.size != 0 and Q.shape[0] != q.shape[0]:
                raise CORAerror('CORA:wrongInputInConstructor',
                                'Dimensions of the shape matrix and center are different.')

            # check for positive semidefinite/symmetric (isApproxSymmetric)
            # This needs to be translated
            # For now, a placeholder or simple check
            mev = np.linalg.eigvalsh(Q)[0]
            if not np.allclose(Q, Q.T, atol=TOL) or mev < -TOL:
                raise CORAerror('CORA:wrongInputInConstructor',
                                'The shape matrix needs to be positive semidefinite/symmetric.')


    def _aux_computeProperties(self, Q, q):
        # returns zero values in case Q or q are empty
        if Q.size == 0:
            # For empty ellipsoids, Q becomes zeros(0,0) but q should preserve its dimensions
            # In MATLAB: Q=zeros(0,0), q=zeros(n,0) where n is the dimension
            Q = np.zeros((0, 0))
            # Don't modify q - keep it as zeros(n,0) to preserve dimension information
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

