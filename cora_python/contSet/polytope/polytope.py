"""
polytope - object constructor for polytope objects

Description:
    This class represents polytope objects defined as (halfspace
    representation)
      { x | A*x <= b }. 
    For convenience, equality constraints
      { x | A*x <= b, Ae*x == be }
    can be added, too.
    Alternatively, polytopes can be defined as (vertex representation)
      { sum_i a_i v_i | sum_i a_i = 1, a_i >= 0 }
    Note: A polytope without any constraints represents R^n.
    Note: A polytope instantiated without input arguments is the empty set.

Syntax:
    P = polytope(V)
    P = polytope(A,b)
    P = polytope(A,b,Ae,be)

Inputs:
    V - (n x p) array of vertices
    A - (n x m) matrix for the inequality representation
    b - (n x 1) vector for the inequality representation
    Ae - (k x l) matrix for the equality representation
    be - (k x 1) vector for the equality representation

Outputs:
    obj - generated polytope object

Example: 
    A = np.array([[1, 0, -1, 0, 1], [0, 1, 0, -1, 1]]).T
    b = np.array([3, 2, 3, 2, 1])
    P = Polytope(A, b)

Other m-files required: none
Subfunctions: none
MAT-files required: none

Authors:       Viktor Kotsev, Mark Wetzlinger, Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       25-April-2022 (MATLAB)
Last update:   16-July-2024 (MATLAB)
Python translation: 2025
"""

# This file is part of the CORA project.
# Copyright (c) 2024, System Control and Robotics Group, TU Graz.
# All rights reserved.

import numpy as np
from typing import Union, List, Tuple, TYPE_CHECKING, Optional

from cora_python.contSet.contSet.contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


class Polytope(ContSet):
    """
    Class for representing polytopes.
    A polytope can be represented by its vertices (V-representation)
    or by half-space constraints (H-representation).

    H-representation: {x in R^n | A*x <= b, Ae*x = be}
    V-representation: conv(V)
    """

    # Give higher priority than numpy arrays for @ operator
    __array_priority__ = 1000

    def __init__(self, *args, **kwargs):
        """
        Constructor for the Polytope class.

        Args:
            *args: Variable arguments for different construction modes:
                   - Polytope(): Empty polytope (not allowed, throws error)
                   - Polytope(V): Vertex representation
                   - Polytope(A, b): Halfspace representation
                   - Polytope(A, b, Ae, be): Halfspace with equality constraints
                   - Polytope(other_polytope): Copy constructor
            **kwargs: Keyword arguments:
                   - A, b: Inequality constraints (A*x <= b)
                   - Ae, be: Equality constraints (Ae*x = be)
                   - A_eq, b_eq: Aliases for Ae, be (for convenience)
                   - V: Vertices (alternative to positional)
        """
        super().__init__()
        
        # Handle keyword arguments as aliases
        if 'A_eq' in kwargs:
            kwargs['Ae'] = kwargs.pop('A_eq')
        if 'b_eq' in kwargs:
            kwargs['be'] = kwargs.pop('b_eq')
        
        # Convert keyword arguments to positional if provided
        if kwargs and not args:
            # Pure keyword argument construction
            V = kwargs.get('V', None)
            A = kwargs.get('A', None)
            b = kwargs.get('b', None)
            Ae = kwargs.get('Ae', None)
            be = kwargs.get('be', None)
            
            if V is not None:
                args = (V,)
            elif A is not None or b is not None or Ae is not None or be is not None:
                # Build positional args from keywords
                A = A if A is not None else np.array([])
                b = b if b is not None else np.array([])
                Ae = Ae if Ae is not None else np.array([])
                be = be if be is not None else np.array([])
                args = (A, b, Ae, be)
        elif kwargs and args:
            # Mixed positional and keyword - extend args with keywords
            args = list(args)
            if len(args) == 2 and ('Ae' in kwargs or 'be' in kwargs):
                # A, b provided positionally, Ae, be as keywords
                Ae = kwargs.get('Ae', np.array([]))
                be = kwargs.get('be', np.array([]))
                args.extend([Ae, be])
            elif len(args) == 1 and any(k in kwargs for k in ['A', 'b', 'Ae', 'be']):
                # V provided positionally, but other args as keywords - this is invalid
                raise CORAerror('CORA:wrongInput', 'Cannot mix vertex representation with constraint keywords')
            args = tuple(args)
        
        # 0. avoid empty instantiation
        if len(args) == 0:
            raise CORAerror('CORA:noInputInSetConstructor')
        
        if len(args) > 4:
            raise CORAerror('CORA:wrongInput', 'Too many input arguments')

        # 1. copy constructor
        if len(args) == 1 and isinstance(args[0], Polytope):
            P = args[0]
            # Copy properties
            self._A = P._A.copy() if P._A is not None else None
            self._b = P._b.copy() if P._b is not None else None
            self._Ae = P._Ae.copy() if P._Ae is not None else None
            self._be = P._be.copy() if P._be is not None else None
            self._V = P._V.copy() if P._V is not None else None
            self.precedence = P.precedence
            
            # Copy set properties
            self._isHRep = P._isHRep
            self._isVRep = P._isVRep
            self._emptySet = P._emptySet
            self._fullDim = P._fullDim
            self._bounded = P._bounded
            self._minHRep = P._minHRep
            self._minVRep = P._minVRep
            return

        # 2. parse input arguments: varargin -> vars
        A, b, Ae, be, V = self._aux_parseInputArgs(*args)

        # 3. check correctness of input arguments
        self._aux_checkInputArgs(A, b, Ae, be, V, len(args))

        # 4. compute properties and hidden properties
        A, b, Ae, be, V, isHRep, isVRep = self._aux_computeProperties(A, b, Ae, be, V, len(args))
        empty, bounded, fullDim, minHRep, minVRep, V, isHRep, isVRep = \
            self._aux_computeHiddenProperties(A, b, Ae, be, V, isHRep, isVRep)

        # 4a. assign properties
        self._A = A
        self._b = b
        self._Ae = Ae
        self._be = be
        self._V = V

        self._isHRep = isHRep
        self._isVRep = isVRep
        self._emptySet = empty
        self._bounded = bounded
        self._fullDim = fullDim
        self._minHRep = minHRep
        self._minVRep = minVRep

        # 5. set precedence (fixed)
        self.precedence = 80

    def _aux_parseInputArgs(self, *varargin):
        """Parse input arguments from user and assign to variables"""
        # no input arguments
        if len(varargin) == 0:
            A = np.array([])
            b = np.array([])
            Ae = np.array([])
            be = np.array([])
            V = np.array([])
            return A, b, Ae, be, V

        # read out arguments
        if len(varargin) == 1:
            # vertices as input argument
            V = np.asarray(varargin[0])
            A = np.array([])
            b = np.array([])
            Ae = np.array([])
            be = np.array([])
        else:
            # halfspaces as input arguments
            A = np.asarray(varargin[0]) if varargin[0] is not None else np.array([])
            b = np.asarray(varargin[1]) if varargin[1] is not None else np.array([])
            Ae = np.asarray(varargin[2]) if len(varargin) > 2 and varargin[2] is not None else np.array([])
            be = np.asarray(varargin[3]) if len(varargin) > 3 and varargin[3] is not None else np.array([])
            V = np.array([])

        return A, b, Ae, be, V

    def _aux_checkInputArgs(self, A, b, Ae, be, V, n_in):
        """Check correctness of input arguments"""
        # Only check if macro set to true (simplified for Python)
        CHECKS_ENABLED = True
        
        if CHECKS_ENABLED and n_in > 0:
            # Check numeric type of V
            if V.size > 0:
                if np.any(np.isnan(V)):
                    raise CORAerror('CORA:wrongInputInConstructor',
                                  'Vertices have to be non-nan.')
                elif V.shape[0] > 1 and np.any(np.isinf(V)):
                    raise CORAerror('CORA:wrongInputInConstructor',
                                  'nD vertices for n > 1 have to be finite.')

            # Check types (all should be numeric arrays)
            for var, name in [(A, 'A'), (b, 'b'), (Ae, 'Ae'), (be, 'be')]:
                if var.size > 0 and not np.issubdtype(var.dtype, np.number):
                    raise CORAerror('CORA:wrongInputInConstructor', f'{name} has to be numeric.')

            # Check b, be (get later reshaped to column vector)
            if b.size > 0 and b.ndim > 1 and min(b.shape) > 1:
                raise CORAerror('CORA:wrongInputInConstructor',
                              'Argument "b" has to be column vector.')
            if be.size > 0 and be.ndim > 1 and min(be.shape) > 1:
                raise CORAerror('CORA:wrongInputInConstructor',
                              'Argument "be" has to be column vector.')

            # Check number of inequality constraints
            if A.size > 0 and A.shape[0] != len(b.flatten()):
                raise CORAerror('CORA:wrongInputInConstructor',
                              'Number of rows does not hold between arguments "A", "b".')
            
            # Check for empty argument
            if A.size == 0 and b.size > 0:
                raise CORAerror('CORA:wrongInputInConstructor',
                              'Number of rows does not hold between arguments "A", "b".')

            # Check number of equality constraints
            if Ae.size > 0 and Ae.shape[0] != len(be.flatten()):
                raise CORAerror('CORA:wrongInputInConstructor',
                              'Number of rows does not hold between arguments "Ae", "be".')

            # Same dimension if both equality and inequality constraints given
            if A.size > 0 and Ae.size > 0 and A.shape[1] != Ae.shape[1]:
                raise CORAerror('CORA:wrongInputInConstructor',
                              'Number of columns does not hold between arguments "A", "Ae".')

    def _aux_computeProperties(self, A, b, Ae, be, V, n_in):
        """Compute properties and fix dimensions"""
        # Determine dimension
        dims = []
        if A.size > 0:
            dims.append(A.shape[1])
        if Ae.size > 0:
            dims.append(Ae.shape[1])
        if V.size > 0:
            dims.append(V.shape[0])
        
        n = max(dims) if dims else 0

        # Offsets must be column vectors
        if b.size > 0:
            b = b.reshape(-1, 1)
        if be.size > 0:
            be = be.reshape(-1, 1)

        # Store which representation is given (constructor only allows one)
        isVRep = n_in == 1
        isHRep = not isVRep

        # In 1D, remove redundancies (otherwise keep V as is)
        if isVRep and n == 1 and V.size > 0:
            V_min = np.min(V)
            V_max = np.max(V)
            V = np.array([V_min, V_max])
            if withinTol(V[0], V[1], np.finfo(float).eps):
                V = np.array([V[0]])

        # Empty constraint matrices must have correct dimension
        if A.size == 0:
            A = np.zeros((0, n))
        if Ae.size == 0:
            Ae = np.zeros((0, n))

        # Remove inequality constraints with Inf in offset (trivially fulfilled)
        if b.size > 0:
            idxRemove = np.isinf(b.flatten()) & (np.sign(b.flatten()) == 1)
            if np.any(idxRemove):
                A = A[~idxRemove, :]
                b = b[~idxRemove]

        return A, b, Ae, be, V, isHRep, isVRep

    def _aux_computeHiddenProperties(self, A, b, Ae, be, V, isHRep, isVRep):
        """Compute hidden properties"""
        # Init hidden properties as unknown
        empty = None
        bounded = None
        fullDim = None
        minHRep = None
        minVRep = None

        # Determine dimension
        dims = []
        if A.size > 0:
            dims.append(A.shape[1])
        if Ae.size > 0:
            dims.append(Ae.shape[1])
        if V.size > 0:
            dims.append(V.shape[0])
        
        n = max(dims) if dims else 0

        # Check if instantiated via vertices
        if isVRep:
            # Check emptiness
            empty = V.size == 0 or (V.ndim == 2 and V.shape[1] == 0)

            # Max. 1 vertex -> minimal V-representation
            minVRep = V.size <= 1 or (V.ndim == 2 and V.shape[1] <= 1) or n == 1

            # Check if 1D
            if n == 1:
                # Inf values for vertices only supported for 1D
                if np.any(np.isinf(V)):
                    bounded = False
                    fullDim = True
                else:
                    bounded = True
                    fullDim = V.size > 1 if V.ndim == 1 else V.shape[1] > 1
            else:
                # nD -> has to be bounded
                bounded = True
                # Easy checks for degeneracy
                if V.ndim == 1:
                    num_vertices = 1 if V.size > 0 else 0
                else:
                    num_vertices = V.shape[1]
                
                if num_vertices <= n:
                    # Full-dimensionality requires at least n+1 vertices
                    fullDim = False
                else:
                    # Use SVD to check full dimensionality
                    try:
                        V_centered = V - np.mean(V, axis=1, keepdims=True)
                        _, S, _ = np.linalg.svd(V_centered, full_matrices=False)
                        fullDim = n == np.sum(~withinTol(S, 0, 1e-12))
                    except:
                        fullDim = None

        elif isHRep:
            if A.size == 0 and Ae.size == 0:
                # No constraints
                empty = False
                bounded = False
                fullDim = True
                minHRep = True
                # Do not compute -Inf/Inf vertices here...
                V = np.array([])
                minVRep = None
            else:
                # Equality constraint with -Inf or Inf in offset OR inequality
                # constraint with -Inf in offset -> empty polytope
                be_inf = be.size > 0 and np.any(np.isinf(be))
                b_neg_inf = b.size > 0 and np.any(np.isinf(b) & (np.sign(b) == -1))
                
                if be_inf or b_neg_inf:
                    empty = True
                    bounded = True
                    fullDim = False
                    # Only a single infeasible constraint required to represent an empty set
                    minHRep = (be.size + b.size) == 1
                    # Init no vertices (which is the minimal representation)
                    V = np.zeros((n, 0))
                    isVRep = True
                    minVRep = True

        return empty, bounded, fullDim, minHRep, minVRep, V, isHRep, isVRep

    # Property getters that match MATLAB behavior
    @property
    def V(self):
        """Get vertices (throws error if not available)"""
        if not self._isVRep:
            raise CORAerror('CORA:specialError',
                           "The vertex representation is not available. " +
                           "Call the function 'polytope/vertices'.")
        return self._V

    @property
    def A(self):
        """Get inequality constraint matrix"""
        if not self._isHRep:
            raise CORAerror('CORA:specialError',
                           "The halfspace representation is not available. " +
                           "Call the function 'polytope/constraints'.")
        return self._A

    @property
    def b(self):
        """Get inequality constraint vector"""
        if not self._isHRep:
            raise CORAerror('CORA:specialError',
                           "The halfspace representation is not available. " +
                           "Call the function 'polytope/constraints'.")
        return self._b

    @property
    def Ae(self):
        """Get equality constraint matrix"""
        if not self._isHRep:
            raise CORAerror('CORA:specialError',
                           "The halfspace representation is not available. " +
                           "Call the function 'polytope/constraints'.")
        return self._Ae

    @property
    def be(self):
        """Get equality constraint vector"""
        if not self._isHRep:
            raise CORAerror('CORA:specialError',
                           "The halfspace representation is not available. " +
                           "Call the function 'polytope/constraints'.")
        return self._be

    # Additional properties for compatibility
    @property
    def isHRep(self):
        """Check if halfspace representation is available"""
        return self._isHRep

    @property
    def isVRep(self):
        """Check if vertex representation is available"""
        return self._isVRep

    @property
    def emptySet(self):
        """Check if polytope is empty"""
        return self._emptySet

    @property
    def fullDim(self):
        """Check if polytope is full-dimensional"""
        return self._fullDim

    @property
    def bounded(self):
        """Check if polytope is bounded"""
        return self._bounded

    @property
    def minHRep(self):
        """Check if halfspace representation is minimal"""
        return self._minHRep

    @property
    def minVRep(self):
        """Check if vertex representation is minimal"""
        return self._minVRep


    def __repr__(self) -> str:
        """
        Official string representation for programmers.
        Should be unambiguous and allow object reconstruction.
        """
        try:
            if self.is_empty():
                return f"Polytope.empty({self.dim()})"
            elif self._isVRep and self._V is not None and self._V.size <= 12:
                # For small polytopes, show vertices
                return f"Polytope({self._V.tolist()})"
            elif self._isHRep and self._A is not None and self._A.size <= 12:
                # For small polytopes, show constraints
                if self._Ae is not None and self._be is not None and self._Ae.size > 0:
                    return f"Polytope({self._A.tolist()}, {self._b.flatten().tolist()}, {self._Ae.tolist()}, {self._be.flatten().tolist()})"
                else:
                    return f"Polytope({self._A.tolist()}, {self._b.flatten().tolist()})"
            else:
                return f"Polytope(dim={self.dim()})"
        except:
            return "Polytope()"
    
    def __str__(self) -> str:
        """
        Informal string representation for users.
        Uses the display method for MATLAB-style output.
        """
        try:
            from .display import display
            return display(self)
        except:
            return self.__repr__()



 