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

    # Halfspace representation and vertex representation (private storage)
    _A: np.ndarray
    _b: np.ndarray
    _Ae: np.ndarray
    _be: np.ndarray
    _V: np.ndarray

    # Private cached properties for lazy evaluation
    _emptySet_val: Optional[bool]
    _fullDim_val: Optional[bool]
    _bounded_val: Optional[bool]

    # Public flags (initially set directly in constructor)
    _isHRep: bool
    _isVRep: bool

    # minHRep and minVRep will be computed lazily if needed, but for now
    # they are not, so we define them as simple attributes.
    _minHRep_val: Optional[bool]
    _minVRep_val: Optional[bool]
    _dim_val: Optional[int] # Stores ambient dimension, initialized to None

    def __init__(self, *args, **kwargs):
        """
        Constructor for the Polytope class, supporting various input formats.

        Supports:
        - Polytope(V): Vertex representation
        - Polytope(A, b): Halfspace representation (inequalities only)
        - Polytope(A, b, Ae, be): Halfspace representation (inequalities + equalities)
        - Polytope(other_polytope): Copy constructor
        - Keyword arguments for flexibility (e.g., Polytope(V=V_array), Polytope(A=A_matrix, b=b_vector))
        """
        super().__init__() # Call parent constructor

        # Initialize cached properties flags
        self._emptySet_val = None
        self._fullDim_val = None
        self._bounded_val = None
        self._minHRep_val = None
        self._minVRep_val = None
        self._dim_val = None # Initialize _dim_val

        # Initialize core properties as empty NumPy arrays (standardized)
        # These are set to 0x0 or 0x1 arrays to ensure consistent shapes for empty sets
        # and to avoid NoneType errors in later operations.
        self._A = np.array([]).reshape(0, 0)
        self._b = np.array([]).reshape(0, 1)
        self._Ae = np.array([]).reshape(0, 0)
        self._be = np.array([]).reshape(0, 1)
        self._V = np.array([]).reshape(0, 0)

        self._isHRep = False # Default to false, determined by input
        self._isVRep = False # Default to false, determined by input

        # Parse inputs
        if len(args) == 1 and isinstance(args[0], Polytope):
            # Copy constructor
            self._copy_constructor(args[0])
            # The dim value is copied directly in _copy_constructor
            return
        

        # Handle Zonotope conversion
        if len(args) == 1 and hasattr(args[0], 'c') and hasattr(args[0], 'G'):
            # This is a Zonotope object, convert it to polytope
            Z = args[0]
            from ..zonotope import Zonotope
            if isinstance(Z, Zonotope):
                # Convert zonotope to polytope using the zonotope's polytope method
                P = Z.polytope()
                # Copy properties from the converted polytope

                self._copy_constructor(P)
                return
        else:
            # Handle general constructors
            self._general_constructor(*args, **kwargs)

        # Ensure _dim_val is set after construction,
        # in case it was not explicitly set by _general_constructor for edge cases.
        # This acts as a final safeguard to guarantee dimension is known.
        # if self._dim_val is None:
        #     # Call the *function* dim (from dim.py) to compute and cache the dimension.
        #     # This ensures the dimension is computed and stored in _dim_val if not already.
        #     # We use `_` to discard the direct return of `dim_func` as it also updates `self._dim_val`.
        #     _ = self.dim() # Call the attached method, which uses the external function

    def _aux_checkInputArgs(self, A, b, Ae, be, V, n_in):
        """Check correctness of input arguments"""
        # Only check if macro set to true (simplified for Python)
        CHECKS_ENABLED = True
        
        if CHECKS_ENABLED and n_in > 0:
            # Check numeric type of V
            if V.size > 0:
                # Check if V contains numeric data (not objects like Zonotope)
                if hasattr(V, 'dtype') and np.issubdtype(V.dtype, np.number):
                    if np.any(np.isnan(V)):
                        raise CORAerror('CORA:wrongInputInConstructor',
                                      'Vertices have to be non-nan.')
                    elif V.shape[0] > 1 and np.any(np.isinf(V)):
                        raise CORAerror('CORA:wrongInputInConstructor',
                                      'nD vertices for n > 1 have to be finite.')
                else:
                    # V contains non-numeric objects (like Zonotope)
                    # This will be handled by the conversion logic
                    pass

    def _copy_constructor(self, other: 'Polytope'):
        """Internal helper for copy constructor."""
        # Ensure that attributes are copied as numpy arrays, even if they are None or empty in the source.
        # This prevents AttributeError when .size or .copy() is called on None.
        self._A = other.A.copy() if other.A is not None and other.A.size > 0 else np.array([]).reshape(0,0)
        self._b = other.b.copy() if other.b is not None and other.b.size > 0 else np.array([]).reshape(0,1)
        self._Ae = other.Ae.copy() if other.Ae is not None and other.Ae.size > 0 else np.array([]).reshape(0,0)
        self._be = other.be.copy() if other.be is not None and other.be.size > 0 else np.array([]).reshape(0,1)
        self._V = other.V.copy() if other.V is not None and other.V.size > 0 else np.array([]).reshape(0,0)

        self._isHRep = other.isHRep
        self._isVRep = other.isVRep

        # Copy lazy evaluation flags and values
        self._emptySet_val = other._emptySet_val
        self._fullDim_val = other._fullDim_val
        self._bounded_val = other._bounded_val
        self._minHRep_val = other._minHRep_val
        self._minVRep_val = other._minVRep_val
        self._dim_val = other._dim_val # Copy the cached dimension


    def _general_constructor(self, *args, **kwargs):
        """Internal helper for general constructors (H-rep, V-rep, empty, fullspace)."""
        A_raw, b_raw, Ae_raw, be_raw, V_raw = None, None, None, None, None
        isHRep_flag, isVRep_flag = False, False

        # If keyword arguments are provided
        if kwargs:
            A_raw = kwargs.get('A')
            b_raw = kwargs.get('b')
            Ae_raw = kwargs.get('Ae')
            be_raw = kwargs.get('be')
            V_raw = kwargs.get('V')
            dim_from_kwargs = kwargs.get('dim') # Retrieve 'dim' keyword argument

            # If 'dim' is provided and no other representation is given,
            # it means we are constructing an empty or full-space polytope of specified dimension.
            if dim_from_kwargs is not None and V_raw is None and A_raw is None and Ae_raw is None:
                self._dim_val = dim_from_kwargs
                # Initialize with correct dimensions for empty arrays
                # This is important for cases like Polytope.empty(N) or Polytope.Inf(N)
                self._A = np.zeros((0, dim_from_kwargs))
                self._b = np.zeros((0, 1))
                self._Ae = np.zeros((0, dim_from_kwargs))
                self._be = np.zeros((0, 1))
                self._V = np.zeros((dim_from_kwargs, 0)) # V-rep is (dim x num_vertices)
                return # Skip further raw input processing for these specific cases

            # Determine representation based on provided keywords
            if V_raw is not None:
                isVRep_flag = True
            elif A_raw is not None or Ae_raw is not None:
                isHRep_flag = True

        # If positional arguments are provided
        elif args:
            if len(args) == 1 and isinstance(args[0], np.ndarray):
                # Assume V-representation if single numpy array
                V_raw = args[0]
                isVRep_flag = True
            elif len(args) == 2 and isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray):
                # Assume H-representation (A, b)
                A_raw = args[0]
                b_raw = args[1]
                isHRep_flag = True
            elif len(args) == 4 and all(isinstance(arg, np.ndarray) for arg in args):
                # Assume H-representation (A, b, Ae, be)
                A_raw = args[0]
                b_raw = args[1]
                Ae_raw = args[2]
                be_raw = args[3]
                isHRep_flag = True
            else:
                raise CORAerror('CORA:wrongInputInConstructor', 'Unsupported positional arguments.')

        # Validate and normalize inputs, and get dimension
        A_norm, b_norm, Ae_norm, be_norm, V_norm, n = _aux_validate_and_normalize_polytope_inputs(
            A_raw, b_raw, Ae_raw, be_raw, V_raw, isHRep_flag, isVRep_flag
        )

        # Set internal properties based on normalized inputs
        self._A = A_norm
        self._b = b_norm
        self._Ae = Ae_norm
        self._be = be_norm
        self._V = V_norm

        self._isHRep = isHRep_flag
        self._isVRep = isVRep_flag

        # Store the determined dimension
        self._dim_val = n
        
        # Compute minimal representation properties (like MATLAB's aux_computeHiddenProperties)
        self._compute_min_representation_properties(n)

    @property
    def A(self) -> np.ndarray:
        """Get the A matrix of the halfspace representation."""
        if not self._isHRep:
            self.constraints()
        return self._A

    @A.setter
    def A(self, val: np.ndarray):
        """Set the A matrix of the halfspace representation."""
        self._A = val
        self.isHRep = True # Use the setter
        self._reset_lazy_flags() # Reset all lazy flags when A is set

    @property
    def b(self) -> np.ndarray:
        """Get the b vector of the halfspace representation."""
        if not self._isHRep:
            self.constraints()
        return self._b

    @b.setter
    def b(self, val: np.ndarray):
        """Set the b vector of the halfspace representation."""
        self._b = val
        self.isHRep = True # Use the setter
        self._reset_lazy_flags() # Reset all lazy flags when b is set

    @property
    def Ae(self) -> np.ndarray:
        """Get the Ae matrix of the halfspace representation (equality constraints)."""
        if not self._isHRep:
            self.constraints()
        return self._Ae

    @Ae.setter
    def Ae(self, val: np.ndarray):
        """Set the Ae matrix of the halfspace representation (equality constraints)."""
        self._Ae = val
        self.isHRep = True # Use the setter
        self._reset_lazy_flags() # Reset all lazy flags when Ae is set

    @property
    def be(self) -> np.ndarray:
        """Get the be vector of the halfspace representation (equality constraints)."""
        if not self._isHRep:
            self.constraints()
        return self._be

    @be.setter
    def be(self, val: np.ndarray):
        """Set the be vector of the halfspace representation (equality constraints)."""
        self._be = val
        self.isHRep = True # Use the setter
        self._reset_lazy_flags() # Reset all lazy flags when be is set

    @property
    def V(self) -> np.ndarray:
        """Get the V matrix (vertices) of the vertex representation."""
        if not self._isVRep:
            self.vertices_()
        return self._V

    @V.setter
    def V(self, val: np.ndarray):
        """Set the V matrix (vertices) of the vertex representation."""
        self._V = val
        self.isVRep = True # Use the setter
        self._reset_lazy_flags() # Reset all lazy flags when V is set

    @property
    def isHRep(self) -> bool:
        """Check if the polytope is currently stored in H-representation."""
        return self._isHRep

    @isHRep.setter
    def isHRep(self, val: bool):
        """Set the H-representation status."""
        self._isHRep = val
        # When HRep status is explicitly set, other representation is implicitly not active
        if val:
            self._isVRep = False

    @property
    def isVRep(self) -> bool:
        """
        Check if the polytope is currently stored in V-representation.
        """
        return self._isVRep

    @isVRep.setter
    def isVRep(self, val: bool):
        """Set the V-representation status."""
        self._isVRep = val
        # When VRep status is explicitly set, other representation is implicitly not active
        if val:
            self._isHRep = False

    @property
    def minHRep(self) -> Optional[bool]:
        """
        Check if halfspace representation is minimal.
        Returns the cached value directly (MATLAB pattern: P.minHRep.val).
        Use interface methods or constraints() to compute if needed.
        """
        return self._minHRep_val

    @minHRep.setter
    def minHRep(self, val: Optional[bool]):
        """Set the minHRep cache value."""
        self._minHRep_val = val

    @property
    def minVRep(self) -> Optional[bool]:
        """
        Check if vertex representation is minimal.
        Returns the cached value directly (MATLAB pattern: P.minVRep.val).
        Use interface methods or vertices_() to compute if needed.
        """
        return self._minVRep_val

    @minVRep.setter
    def minVRep(self, val: Optional[bool]):
        """Set the minVRep cache value."""
        self._minVRep_val = val

    def _compute_min_representation_properties(self, n: int):
        """
        Compute minimal representation properties during construction.
        Mirrors MATLAB's aux_computeHiddenProperties logic.
        """
        # note: representations for 1D polytopes always minimal (MATLAB comment)
        
        if self._isVRep:
            # MATLAB: minVRep = size(V,2) <= 1 || n == 1;
            # max. 1 vertex -> minimal V-representation
            self._minVRep_val = self._V.shape[1] <= 1 or n == 1
            
        elif self._isHRep:
            if self._A.size == 0 and self._Ae.size == 0:
                # MATLAB: no constraints -> minHRep = true
                self._minHRep_val = True
            else:
                # Check for empty polytope cases
                # MATLAB: equality constraint with -Inf or Inf in offset OR 
                # inequality constraint with -Inf in offset -> empty polytope
                be_has_inf = np.any(np.isinf(self._be)) if self._be.size > 0 else False
                b_has_neg_inf = (np.any(np.isinf(self._b) & (self._b < 0)) 
                               if self._b.size > 0 else False)
                
                if be_has_inf or b_has_neg_inf:
                    # MATLAB: only a single infeasible constraint required to represent an empty set
                    # minHRep = length(be) + length(b) == 1;
                    total_constraints = (self._be.shape[0] if self._be.size > 0 else 0) + \
                                      (self._b.shape[0] if self._b.size > 0 else 0)
                    self._minHRep_val = total_constraints == 1

    def _reset_lazy_flags(self):
        """Resets all lazy evaluation cache values when the polytope's definition changes."""
        self._emptySet_val = None
        self._fullDim_val = None  
        self._bounded_val = None
        self._minHRep_val = None
        self._minVRep_val = None
        # Do NOT reset _dim_val here; the ambient dimension doesn't change with representation.
        # It's a fundamental property of the space the polytope lives in.


    def __repr__(self) -> str:
        """
        Official string representation for programmers.
        Should be unambiguous and allow object reconstruction.
        """
        try:
            # Call the external dim function via the attached method
            current_dim = self.dim() # Using the attached method P.dim()
            if self.isemptyobject():
                return f"Polytope.empty({current_dim})"
            elif self.isVRep and self.V.size > 0 and self.V.shape[1] <= 12: # Check V.size > 0 to avoid empty list
                # For small polytopes, show vertices
                return f"Polytope(V={self.V.tolist()})"
            elif self.isHRep and self.A.size > 0 and self.A.shape[0] <= 12: # Check A.size > 0
                # For small polytopes, show constraints
                if self.Ae.size > 0:
                    return f"Polytope(A={self.A.tolist()}, b={self.b.flatten().tolist()}, Ae={self.Ae.tolist()}, be={self.be.flatten().tolist()})"
                else:
                    return f"Polytope(A={self.A.tolist()}, b={self.b.flatten().tolist()})"
            else:
                return f"Polytope(dim={current_dim})"
        except Exception as e:
            return f"Polytope object (error in repr: {e})"

    def __str__(self) -> str:
        """
        Informal string representation for users.
        Uses the display method for MATLAB-style output.
        """
        try:
            from cora_python.contSet.polytope.display import display as display_func
            return display_func(self)
        except Exception as e:
            return f"Polytope object (error in str: {e})"


# Auxiliary functions (outside the class definition)

def _aux_validate_and_normalize_polytope_inputs(
    A_raw: Optional[np.ndarray], b_raw: Optional[np.ndarray],
    Ae_raw: Optional[np.ndarray], be_raw: Optional[np.ndarray],
    V_raw: Optional[np.ndarray],
    isHRep_flag: bool, isVRep_flag: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Validates and transforms raw input arrays for Polytope construction into a
    standardized format. Ensures correct types, shapes, and handles empty/None inputs.
    Returns: A, b, Ae, be, V (all as standardized NumPy arrays), and n (dimension).
    """
    A, b, Ae, be, V = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    n = 0 # Initialize dimension

    if isVRep_flag:
        if V_raw is None or not isinstance(V_raw, np.ndarray):
            raise CORAerror('CORA:wrongInputInConstructor', 'Vertices (V) must be a numpy array.')
        V = V_raw

        if V.ndim == 1:
            V = V.reshape(-1, 1) # Ensure column vector for 1D points/vertices (n x 1)
        elif V.ndim == 0: # Handle scalar input like np.array(5) -> should be (1,1)
            V = V.reshape(1,1)
        elif V.ndim > 2:
            raise CORAerror('CORA:wrongInputInConstructor', 'Vertices (V) must be a 1D or 2D array.')

        # Check for NaN/Inf in vertices
        if np.any(np.isnan(V)):
            raise CORAerror('CORA:wrongInputInConstructor', 'Vertices have to be non-nan.')
        
        # In MATLAB, 1D (rows=1) means dimension is 1, and each column is a vertex.
        # For Python (d x num_vertices), 1D is dim 1.
        # If V is (N, 1) for N points in 1D space, it means N vertices.
        # If V is (1, N) for 1 point in N-dimensional space, it means 1 vertex.
        # Assume V is (dim x num_vertices) or (num_vertices x dim) and we take dim as V.shape[0] or V.shape[1]
        
        # Convention: V is (dimension x number_of_vertices)
        # So V.shape[0] is the dimension.
        n = V.shape[0] if V.size > 0 else 0 # Determine dimension from V's first dimension

        # Handle special case for 1D polytopes to ensure vertices are sorted and unique min/max
        if n == 1 and V.size > 0:
            if V.shape[1] > 1: # If more than one vertex, simplify to min/max
                V = np.array([[np.min(V), np.max(V)]]) # Ensure 1x2 array (min, max)
                if withinTol(V[0,0], V[0,1], np.finfo(float).eps): # If min and max are same (within tol)
                    V = V[:,0].reshape(-1,1) # Make it a single column vector (1x1 point)

        # Initialize A, b, Ae, be for V-rep (empty but with correct dimensions if n > 0)
        A = np.zeros((0, n))
        b = np.zeros((0, 1))
        Ae = np.zeros((0, n))
        be = np.zeros((0, 1))

    elif isHRep_flag:
        A = A_raw if A_raw is not None else np.array([]).reshape(0,0)
        b = b_raw if b_raw is not None else np.array([]).reshape(0,1)
        Ae = Ae_raw if Ae_raw is not None else np.array([]).reshape(0,0)
        be = be_raw if be_raw is not None else np.array([]).reshape(0,1)
        V = np.array([]).reshape(0,0) # No vertices for HRep construction

        # Validate types
        for var, name in [(A, 'A'), (b, 'b'), (Ae, 'Ae'), (be, 'be')]:
            if not isinstance(var, np.ndarray):
                raise CORAerror('CORA:wrongInputInConstructor', f'{name} has to be a numpy array.')
            if var.size > 0 and not np.issubdtype(var.dtype, np.number):
                raise CORAerror('CORA:wrongInputInConstructor', f'{name} has to contain numeric values.')

        # Ensure b and be are column vectors
        if b.ndim == 1:
            b = b.reshape(-1, 1)
        elif b.ndim == 2 and b.shape[1] != 1 and b.shape[0] != 0:
             raise CORAerror('CORA:wrongInputInConstructor', 'Argument "b" has to be a column vector or 1D array.')

        if be.ndim == 1:
            be = be.reshape(-1, 1)
        elif be.ndim == 2 and be.shape[1] != 1 and be.shape[0] != 0:
            raise CORAerror('CORA:wrongInputInConstructor', 'Argument "be" has to be a column vector or 1D array.')

        # Determine dimension from A/Ae
        if A.size > 0:
            n = A.shape[1]
        elif Ae.size > 0:
            n = Ae.shape[1]
        # If A and Ae are empty, n remains 0 initially; it will be determined by context or default.

        # Ensure empty constraint matrices have correct column dimension (0 rows, n columns)
        # This is critical for `numpy.dot` operations later, which fail on 0x0 @ x if x is N-dim.
        if A.shape[1] != n: # If A is empty or has wrong dimension
             A = np.zeros((A.shape[0], n)) if A.shape[0] > 0 else np.zeros((0, n))
        if Ae.shape[1] != n: # If Ae is empty or has wrong dimension
             Ae = np.zeros((Ae.shape[0], n)) if Ae.shape[0] > 0 else np.zeros((0, n))

        # Check row consistency
        if A.shape[0] != b.shape[0]:
            raise CORAerror('CORA:wrongInputInConstructor', 'Number of rows does not match between A and b.')
        if Ae.shape[0] != be.shape[0]:
            raise CORAerror('CORA:wrongInputInConstructor', 'Number of rows does not match between Ae and be.')

        # Check dimension consistency between A and Ae
        if A.size > 0 and Ae.size > 0 and A.shape[1] != Ae.shape[1]:
            raise CORAerror('CORA:wrongInputInConstructor', 'Number of columns (dimensions) does not match between A and Ae.')

        # Remove inequality constraints with Inf in offset (trivially fulfilled, A*x <= inf)
        if b.size > 0:
            idxRemove = np.isinf(b) & (np.sign(b) == 1) # Positive infinity
            if np.any(idxRemove):
                A = A[~idxRemove.flatten(), :]
                b = b[~idxRemove.flatten(), :].reshape(-1, 1) # Ensure b remains column vector
    else:
        # If neither VRep nor HRep flags set, means no valid inputs were explicitly provided.
        # This typically implies an empty set or a default 0-dimensional case.
        # Initialize everything as empty 0-dimensional arrays.
        A = np.array([]).reshape(0, 0)
        b = np.array([]).reshape(0, 1)
        Ae = np.array([]).reshape(0, 0)
        be = np.array([]).reshape(0, 1)
        V = np.array([]).reshape(0, 0)
        n = 0 # Default to 0 dimension if no other information


    return A, b, Ae, be, V, n