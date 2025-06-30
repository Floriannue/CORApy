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
import sys
import os
from typing import TYPE_CHECKING, Union, List, Tuple

# Add paths for imports
if __name__ == "__main__":
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from cora_python.g.functions.helper.sets.contSet.contSet.reorder_numeric import reorder_numeric
    from g.functions.matlab.validate.check.equal_dim_check import equal_dim_check
    from g.functions.matlab.validate.check.withinTol import withinTol

from ..contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


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
            raise CORAerror('CORA:noInputInSetConstructor', 
                           'No input arguments provided to interval constructor')
        
        if len(args) > 2:
            raise CORAerror('CORA:wrongInput', 
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
            # Check if it's a Zonotope object
            if hasattr(args[0], '__class__') and args[0].__class__.__name__ == 'Zonotope':
                # Convert zonotope to interval using MATLAB algorithm
                Z = args[0]
                c = Z.c.flatten()
                delta = np.sum(np.abs(Z.G), axis=1)
                lb = c - delta
                ub = c + delta
            else:
                lb = np.asarray(args[0], dtype=float)
                ub = lb.copy()
        elif len(args) == 2:
            lb = np.asarray(args[0], dtype=float) if args[0] is not None else np.array([])
            ub = np.asarray(args[1], dtype=float) if args[1] is not None else np.array([])
        else:
            raise CORAerror('CORA:wrongInput', 'Invalid number of arguments')
        
        return lb, ub
    
    def _check_input_args(self, lb, ub, n_in):
        """Check correctness of input arguments"""
        # Check for NaN values
        if np.any(np.isnan(lb)) or np.any(np.isnan(ub)):
            raise CORAerror('CORA:wrongInputInConstructor', 
                           'Input arguments contain NaN values')
        
        # Check dimension compatibility
        if lb.size > 0 and ub.size > 0:
            if lb.shape != ub.shape:
                raise CORAerror('CORA:wrongInputInConstructor',
                               'Lower and upper bounds have different dimensions')
            
            # Check bound ordering with tolerance
            if not np.all(lb <= ub):
                if not np.all(withinTol(lb, ub, 1e-6) | (lb <= ub)):
                    raise CORAerror('CORA:wrongInputInConstructor',
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
                    raise CORAerror('CORA:wrongInputInConstructor',
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
    
    def __pos__(self):
        """Unary plus operation (returns self)"""
        return self
    
    @property
    def T(self):
        """Transpose property"""
        return self.transpose()
    
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
    
    # String representation following Python best practices
    def __repr__(self) -> str:
        """
        Official string representation for programmers.
        Should be unambiguous and allow object reconstruction.
        """
        try:
            dim_val = self.dim()
            if self.is_empty():
                return f"Interval.empty({dim_val})"
            elif self.representsa_('point'):
                # For point intervals, show the point value
                point = (self.inf + self.sup) / 2
                if point.size == 1:
                    return f"Interval({point.item()})"
                else:
                    return f"Interval({point.tolist()})"
            else:
                # General case - show bounds
                if self.inf.size == 1:
                    return f"Interval({self.inf.item()}, {self.sup.item()})"
                else:
                    return f"Interval({self.inf.tolist()}, {self.sup.tolist()})"
        except:
            return f"Interval(dim=unknown)"
    
    def __getitem__(self, key):
        """Indexing operation (e.g., I[0:2, 1:3])"""
        # Apply the same indexing to both inf and sup
        new_inf = self.inf[key]
        new_sup = self.sup[key]
        return Interval(new_inf, new_sup) 
