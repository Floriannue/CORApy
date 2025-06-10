"""
interval - object constructor for real-valued intervals

Description:
    This class represents interval objects defined as
    {x | a_i <= x <= b_i, ∀ i = 1,...,n}.

Syntax:
    obj = Interval(I)
    obj = Interval(a)
    obj = Interval(a,b)

Inputs:
    I - interval object
    a - lower limit
    b - upper limit

Outputs:
    obj - generated interval object

Example:
    a = [1, -1]
    b = [2, 3]
    I = Interval(a, b)

Authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 19-June-2015 (MATLAB)
Last update: 08-December-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, Optional, Any
from ..contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


# Import auxiliary functions
from .aux_functions import _within_tol, _reorder_numeric, _equal_dim_check


class Interval(ContSet):
    """
    Interval class for real-valued intervals
    
    This class represents interval objects defined as
    {x | a_i <= x <= b_i, ∀ i = 1,...,n}.
    
    Properties:
        inf: Lower bound (numpy array)
        sup: Upper bound (numpy array)
        precedence: Set to 120 for intervals
    """
    
    def __init__(self, *args):
        """
        Constructor for interval objects
        
        Args:
            *args: Variable arguments for different construction modes:
                   - Interval(I): Copy constructor
                   - Interval(a): Point interval
                   - Interval(a, b): Interval with bounds
        """
        # Call parent constructor
        super().__init__()
        
        # Avoid empty instantiation
        if len(args) == 0:
            raise CORAError('CORA:noInputInSetConstructor', 
                           'No input arguments provided to interval constructor')
        
        if len(args) > 2:
            raise CORAError('CORA:wrongInput', 
                           'Too many input arguments for interval constructor')
        
        # Copy constructor
        if len(args) == 1 and isinstance(args[0], Interval):
            other = args[0]
            self.inf = other.inf.copy()
            self.sup = other.sup.copy()
            self.precedence = 120
            return
        
        # Parse input arguments
        lb, ub = self._parse_input_args(*args)
        
        # Check correctness of input arguments
        self._check_input_args(lb, ub, len(args))
        
        # Compute properties (deal with corner cases)
        lb, ub = self._compute_properties(lb, ub)
        
        # Assign properties
        self.inf = lb
        self.sup = ub
        
        # Set precedence (fixed for intervals)
        self.precedence = 120
    
    def _parse_input_args(self, *args):
        """Parse input arguments from user and assign to variables"""
        if len(args) == 1:
            lb = np.asarray(args[0], dtype=float)
            ub = lb.copy()
        elif len(args) == 2:
            lb = np.asarray(args[0], dtype=float) if args[0] is not None else np.array([])
            ub = np.asarray(args[1], dtype=float) if args[1] is not None else np.array([])
        else:
            raise CORAError('CORA:wrongInput', 'Invalid number of arguments')
        
        return lb, ub
    
    def _check_input_args(self, lb, ub, n_in):
        """Check correctness of input arguments"""
        # Check for NaN values
        if np.any(np.isnan(lb)) or np.any(np.isnan(ub)):
            raise CORAError('CORA:wrongInputInConstructor', 
                           'Input arguments contain NaN values')
        
        # Check dimension compatibility
        if lb.size > 0 and ub.size > 0:
            if lb.shape != ub.shape:
                raise CORAError('CORA:wrongInputInConstructor',
                               'Lower and upper bounds have different dimensions')
            
            # Check bound ordering with tolerance
            if not np.all(lb <= ub):
                if not np.all(_within_tol(lb, ub, 1e-6) | (lb <= ub)):
                    raise CORAError('CORA:wrongInputInConstructor',
                                   'Lower bound is larger than upper bound')
    
    def _compute_properties(self, lb, ub):
        """Compute properties and handle corner cases"""
        # Handle empty intervals - preserve shape for proper empty intervals
        if lb.size == 0 and ub.size == 0:
            # Keep the original shape if it was intentionally set (e.g., (n, 0))
            pass
        
        # Check for infinite bounds that indicate empty intervals
        if lb.size > 0 and ub.size > 0:
            # If any dimension has [-inf, -inf] or [inf, inf], it's empty
            inf_mask = np.isinf(lb) & np.isinf(ub) & (np.sign(lb) == np.sign(ub))
            if np.any(inf_mask):
                # For matrices, this is not allowed
                if lb.ndim > 1 and np.any(np.array(lb.shape) > 1):
                    raise CORAError('CORA:wrongInputInConstructor',
                                   'Empty interval matrix cannot be instantiated')
                # Create empty interval
                shape = list(lb.shape)
                shape = [s if s > 1 else 0 for s in shape]
                lb = np.zeros(shape)
                ub = np.zeros(shape)
        
        return lb, ub
    
    def end(self, k, n):
        """Overload the end operator for referencing elements"""
        return self.inf.shape[k-1] if k <= len(self.inf.shape) else 1
    
    # Method implementations (imported from separate files)
    def dim(self) -> int:
        """Get dimension of the interval"""
        from .dim import dim
        return dim(self)
    
    def is_empty(self) -> bool:
        """Check if interval is empty"""
        from .isemptyobject import isemptyobject
        return isemptyobject(self)
    
    def representsa_(self, set_type: str, tol: float = 1e-9) -> bool:
        """Check if interval represents a specific set type"""
        from .representsa_ import representsa_
        return representsa_(self, set_type, tol)
    
    def __eq__(self, other) -> bool:
        """Equality comparison"""
        from .isequal import isequal
        return isequal(self, other)
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if interval contains given point(s)"""
        from .contains_ import contains_
        return contains_(self, point)
    
    def center(self) -> np.ndarray:
        """Get center of the interval"""
        from .center import center
        return center(self)
    
    def rad(self) -> np.ndarray:
        """Get radius of the interval"""
        from .rad import rad
        return rad(self)
    
    def project(self, dims):
        """Project interval to lower-dimensional subspace"""
        from .project import project
        return project(self, dims)
    
    def is_bounded(self) -> bool:
        """Check if interval is bounded"""
        from .is_bounded import is_bounded
        return is_bounded(self)
    
    def vertices(self, *args):
        """Get vertices of the interval"""
        from .vertices import vertices
        return vertices(self)
    
    def vertices_(self):
        """Get vertices of the interval (internal version)"""
        from .vertices import vertices_
        return vertices_(self)
    
    def interval(self, *args):
        """Return self (already an Interval)"""
        return self
    
    def and_(self, other, method: str = 'exact'):
        """Intersection with another set"""
        from .and_ import and_
        return and_(self, other, method)
    
    # Operator overloads
    def __add__(self, other):
        """Addition operation"""
        from .plus import plus
        return plus(self, other)
    
    def __radd__(self, other):
        """Reverse addition operation"""
        from .plus import plus
        return plus(other, self)
    
    def __sub__(self, other):
        """Subtraction operation"""
        from .minus import minus
        return minus(self, other)
    
    def __rsub__(self, other):
        """Reverse subtraction operation"""
        from .minus import minus
        return minus(other, self)
    
    def __mul__(self, other):
        """Element-wise multiplication operation"""
        # For now, delegate to mtimes
        return self.__matmul__(other)
    
    def __rmul__(self, other):
        """Reverse element-wise multiplication operation"""
        return self.__rmatmul__(other)
    
    def __matmul__(self, other):
        """Matrix multiplication operation"""
        from .mtimes import mtimes
        return mtimes(self, other)
    
    def __rmatmul__(self, other):
        """Reverse matrix multiplication operation"""
        from .mtimes import mtimes
        return mtimes(other, self)
    
    # Numpy integration
    def __array__(self, dtype=None):
        """Prevent automatic numpy array conversion"""
        return NotImplemented
    
    # Set high priority for operations
    __array_priority__ = 1000
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle numpy universal functions"""
        if method == '__call__':
            if ufunc == np.add:
                return self.__add__(inputs[1] if inputs[0] is self else inputs[0])
            elif ufunc == np.subtract:
                if inputs[0] is self:
                    return self.__sub__(inputs[1])
                else:
                    return self.__rsub__(inputs[0])
            elif ufunc == np.multiply:
                return self.__mul__(inputs[1] if inputs[0] is self else inputs[0])
            elif ufunc == np.matmul:
                return self.__matmul__(inputs[1] if inputs[0] is self else inputs[0])
        
        # For other ufuncs, return NotImplemented to let numpy handle it
        return NotImplemented
    
    # Static methods (imported from separate files)
    @staticmethod
    def generateRandom(*args, **kwargs):
        """Generates random interval"""
        from .generateRandom import generateRandom
        return generateRandom(*args, **kwargs)
    
    @staticmethod
    def enclosePoints(points):
        """Enclosure of point cloud"""
        raise NotImplementedError("enclosePoints not implemented")
    
    @staticmethod
    def empty(n: int = 0):
        """Instantiates an empty interval"""
        from .empty import empty
        return empty(n)
    
    @staticmethod
    def Inf(n: int):
        """Instantiates a fullspace interval"""
        from .Inf import Inf
        return Inf(n)
    
    @staticmethod
    def origin(n: int):
        """Instantiates an interval representing the origin in R^n"""
        from .origin import origin
        return origin(n)
    
    # Protected methods (method signatures only)
    def _getPrintSetInfo(self):
        """Get abbreviation and print order for set"""
        raise NotImplementedError("_getPrintSetInfo not implemented")
    
    # String representation
    def __str__(self) -> str:
        """String representation of interval"""
        if self.inf.size == 0:
            return "interval: empty"
        return f"interval: inf={self.inf}, sup={self.sup}"
    
    def __repr__(self) -> str:
        """Detailed string representation of interval"""
        return self.__str__() 
