import numpy as np
from typing import Union, Optional, Tuple, Any

from cora_python.contSet.contSet.contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check import assertNarginConstructor
from cora_python.g.functions.matlab.validate.check import inputArgsCheck


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
        return Q, q, TOL

    def _aux_checkInputArgs(self, Q, q, TOL):
        # check correctness of input arguments
        CHECKS_ENABLED = True # This needs to be correctly handled from a global config

        if CHECKS_ENABLED:
            # allow empty Q matrix for ellipsoid.empty
            if Q.size == 0:
                # only ensure that q is also empty
                if q.size != 0:
                    raise CORAerror('CORA:wrongInputInConstructor',
                                    'Shape matrix is empty, but center is not.')
                return

            inputArgsCheck([
                [Q, 'att', ['numpy.ndarray'], {'finite', 'matrix'}],
                [q, 'att', ['numpy.ndarray'], {'finite', 'column'}],
                [TOL, 'att', ['numeric'], {'nonnegative', 'scalar'}],
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
            Q = np.zeros((0, 0))
            if q.size != 0:
                q = np.zeros((0, 0))
        return Q, q 