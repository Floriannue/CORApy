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
    A = [1 0 -1 0 1];
    b = [3; 2; 3; 2; 1];
    P = polytope(A,b);

Other m-files required: none
Subfunctions: none
MAT-files required: none

Authors:       Viktor Kotsev, Mark Wetzlinger, Tobias Ladner
Written:       25-April-2022
Last update:   01-December-2022 (MW, add CORAerrors, checks)
               12-June-2023 (MW, add hidden properties)
               08-December-2023 (MW, handle -Inf/Inf offsets)
               01-January-2024 (MW, different meaning of fully empty obj)
               13-March-2024 (TL, check if input is numeric)
               16-July-2024 (MW, allow separate usage of VRep/HRep)
Last revision: 25-July-2023 (MW, restructure constructor)
"""

# This file is part of the CORA project.
# Copyright (c) 2024, System Control and Robotics Group, TU Graz.
# All rights reserved.

import numpy as np
from typing import Union, List, Tuple

from cora_python.contSet.contSet.contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from .private.priv_V_to_H import priv_V_to_H
from .vertices_ import vertices_

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

    def __init__(self, *args):
        """
        Constructor for the Polytope class.

        Possible calls:
        P = Polytope()            # Empty polytope
        P = Polytope(V)           # From vertices V
        P = Polytope(contSetObj)  # From another set with vertices
        P = Polytope(A, b)        # From inequalities
        P = Polytope(A, b, Ae, be)# From inequalities and equalities
        """
        super().__init__()
        self.precedence = 80
        self.dimension = 0

        # Representations
        self._V = None
        self._A = None
        self._b = None
        self._Ae = None
        self._be = None
        
        # Internal flags
        self._has_v_rep = False
        self._has_h_rep = False

        # --- Constructor Logic ---
        if len(args) == 0:
            # Empty polytope
            self._has_v_rep = True # Empty V-rep
            self._V = np.array([[]])
            return

        # Copy constructor
        if len(args) == 1 and isinstance(args[0], Polytope):
            p_in = args[0]
            self._V = p_in._V.copy() if p_in._V is not None else None
            self._A = p_in._A.copy() if p_in._A is not None else None
            self._b = p_in._b.copy() if p_in._b is not None else None
            self._Ae = p_in._Ae.copy() if p_in._Ae is not None else None
            self._be = p_in._be.copy() if p_in._be is not None else None
            self._has_v_rep = p_in._has_v_rep
            self._has_h_rep = p_in._has_h_rep
            self.dimension = p_in.dimension
            return

        # From another set object that has vertices
        if len(args) == 1 and isinstance(args[0], ContSet):
            try:
                self._V = args[0].vertices()
                self._has_v_rep = True
            except (NotImplementedError, CORAerror):
                raise CORAerror('CORA:wrongInputInConstructor',
                              'The provided set object must have a vertices method.')
            if self._V is None or self._V.size == 0:
                 self._has_v_rep = False # Could not get vertices
            else:
                 self.dimension = self._V.shape[0]  # rows are spatial dimensions
            return

        # From numeric arrays
        # V-representation
        if len(args) == 1:
            self._V = np.asarray(args[0])
            if self._V.size > 0:
                self.dimension = self._V.shape[0]  # rows are spatial dimensions
            self._has_v_rep = True
        # H-representation
        elif len(args) in [2, 3, 4]:
            self._A = np.asarray(args[0])
            self._b = np.asarray(args[1]).reshape(-1, 1)
            if self._A.size > 0:
                self.dimension = self._A.shape[1]

            if len(args) >= 3 and args[2] is not None:
                self._Ae = np.asarray(args[2])
                if self.dimension == 0 and self._Ae.size > 0:
                    self.dimension = self._Ae.shape[1]
            if len(args) == 4 and args[3] is not None:
                self._be = np.asarray(args[3]).reshape(-1, 1)
            self._has_h_rep = True
        else:
            raise CORAerror('CORA:wrongInputInConstructor',
                          'Invalid number of arguments for Polytope constructor.')

    @property
    def V(self):
        if not self._has_v_rep:
            self._V = vertices_(self)
            self._has_v_rep = True
        return self._V

    @property
    def A(self):
        if not self._has_h_rep:
            self._convert_V_to_H()
        return self._A

    @property
    def b(self):
        if not self._has_h_rep:
            self._convert_V_to_H()
        return self._b

    @property
    def Ae(self):
        if not self._has_h_rep:
            self._convert_V_to_H()
        return self._Ae

    @property
    def be(self):
        if not self._has_h_rep:
            self._convert_V_to_H()
        return self._be

    def _convert_V_to_H(self):
        """Converts vertex representation to half-space representation."""
        if self._V is None or self._V.size == 0:
            # Cannot convert empty vertices
            self._A, self._b, self._Ae, self._be = np.array([]), np.array([]), np.array([]), np.array([])
        else:
            # priv_V_to_H expects vertices as columns (d x n_vertices), which is our storage format
            self._A, self._b, self._Ae, self._be = priv_V_to_H(self._V)
        self._has_h_rep = True

    def dim(self):
        return self.dimension

    def __repr__(self):
        # A simple representation for the polytope
        if self._V is not None:
            return f"Polytope(V={self._V.shape})"
        elif self._A is not None:
            if self._Ae is not None:
                return f"Polytope(H-rep: A({self._A.shape}), b({self._b.shape}), Ae({self._Ae.shape}), be({self._be.shape}))"
            else:
                return f"Polytope(H-rep: A({self._A.shape}), b({self._b.shape}))"
        else:
            return "Polytope(empty)"

    def __mul__(self, other):
        from .mtimes import mtimes
        return mtimes(self, other)

    def __rmul__(self, other):
        from .mtimes import mtimes
        return mtimes(other, self)

    def __matmul__(self, other):
        from .mtimes import mtimes
        return mtimes(self, other)

    def __rmatmul__(self, other):
        from .mtimes import mtimes
        return mtimes(other, self)

    def __add__(self, other):
        return self.plus(self, other)

    def constraints(self):
        """
        Computes the half-space representation (A, b) and equality 
        constraints (Ae, be) of the polytope if it is not already available.
        
        If the polytope is defined by vertices, it converts them to half-spaces.
        
        Returns:
            Polytope: The same polytope object with A, b, Ae, and be properties populated.
        """
        if self._A is None:
            if self._V is not None:
                from .private.priv_V_to_H import priv_V_to_H
                # priv_V_to_H expects vertices as columns (d x n_vertices), which is our storage format
                self._A, self._b, self._Ae, self._be = priv_V_to_H(self._V)
            else:
                # This is an empty polytope defined with no args
                pass
        return self

    # Static methods
    @staticmethod
    def generate_random(*args) -> 'Polytope':
        # P = generateRandom(varargin) % generate random polytope
        # This will be implemented in a separate file: generate_random.py
        raise NotImplementedError("This method will be implemented in a separate file.")

    @staticmethod
    def enclose_points(points: np.ndarray, *args) -> 'Polytope':
        # P = enclosePoints(points,varargin) % enclose point cloud with polytope
        # This will be implemented in a separate file: enclose_points.py
        raise NotImplementedError("This method will be implemented in a separate file.")

    @staticmethod
    def empty(n: int) -> 'Polytope':
        # P = empty(n) % instantiate empty polytope
        # This will be implemented in a separate file: empty.py
        raise NotImplementedError("This method will be implemented in a separate file.")

    @staticmethod
    def Inf(n: int) -> 'Polytope':
        # P = Inf(n) % instantiate polytope representing R^n
        # This will be implemented in a separate file: Inf.py
        raise NotImplementedError("This method will be implemented in a separate file.")

    @staticmethod
    def origin(n: int) -> 'Polytope':
        # P = origin(n) % instantiate polytope representing the origin in R^n
        # This will be implemented in a separate file: origin.py
        raise NotImplementedError("This method will be implemented in a separate file.")

    # Protected methods
    def get_print_set_info(self) -> Tuple[str, List[str]]:
        # [abbrev,printOrder] = getPrintSetInfo(S)
        # This will be implemented in a separate file: get_print_set_info.py
        raise NotImplementedError("This method will be implemented in a separate file.")

    def representsa_(self, set_type: str, tol: float = 1e-9, **kwargs) -> bool:
        """Internal check if set represents a specific type"""
        from .representsa_ import representsa_
        return representsa_(self, set_type, tol, **kwargs)

    def copy(self):
        """Create a copy of this polytope"""
        return Polytope(self) 