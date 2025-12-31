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

Authors:       Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
Written:       19-June-2015
Last update:   18-November-2015
               26-January-2016
               15-July-2017 (NK)
               01-May-2020 (MW, delete redundant if-else)
               20-March-2021 (MW, error messages)
               14-December-2022 (TL, property check in inputArgsCheck)
               29-March-2023 (TL, optimized constructor)
               08-December-2023 (MW, handle [-Inf,-Inf] / [Inf,Inf] case)
Last revision: 16-June-2023 (MW, restructure using auxiliary functions)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from scipy.sparse import spmatrix


from ..contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


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
                lb = np.atleast_1d(np.asarray(args[0], dtype=float))
                ub = lb.copy()
        elif len(args) == 2:
            
            if isinstance(args[0], spmatrix):
                lb = args[0]
            else:
                lb = np.atleast_1d(np.asarray(args[0], dtype=float)) if args[0] is not None else np.array([])
            
            if isinstance(args[1], spmatrix):
                ub = args[1]
            else:
                ub = np.atleast_1d(np.asarray(args[1], dtype=float)) if args[1] is not None else np.array([])

        # If args has more than 2 elements, we have a problem
        else:
            raise CORAerror('CORA:wrongInputInConstructor', 'Invalid number of input arguments')
        
        return lb, ub
    
    def _check_input_args(self, lb, ub, n_in):
        """Check correctness of input arguments"""
        # Check for NaN values
        if isinstance(lb, spmatrix):
            if np.any(np.isnan(lb.data)):
                raise CORAerror('CORA:noNaNsAllowed')
        elif np.any(np.isnan(lb)):
            raise CORAerror('CORA:noNaNsAllowed')
        
        if isinstance(ub, spmatrix):
            if np.any(np.isnan(ub.data)):
                raise CORAerror('CORA:noNaNsAllowed')
        elif np.any(np.isnan(ub)):
            raise CORAerror('CORA:noNaNsAllowed')

        # Check for dimension mismatch
        if n_in == 2 and lb.shape != ub.shape:
            raise CORAerror('CORA:dimensionMismatch')

        # Check for wrong arguments
        if n_in == 2 and lb.size > 0 and ub.size > 0:
            lb_dense = lb.toarray() if isinstance(lb, spmatrix) else lb
            ub_dense = ub.toarray() if isinstance(ub, spmatrix) else ub
            if np.any(lb_dense > ub_dense):
                raise CORAerror('CORA:wrongArguments')

    def _compute_properties(self, lb, ub):
        """Compute properties and handle corner cases"""
        # Handle empty intervals - preserve shape for proper empty intervals
        if lb.size == 0 and ub.size == 0:
            # Keep the original shape if it was intentionally set (e.g., (n, 0))
            # In MATLAB, empty intervals preserve dimension information
            pass

        # Check for infinite bounds that indicate empty intervals
        if lb.size > 0 and ub.size > 0:
            lb_dense = lb.toarray() if isinstance(lb, spmatrix) else lb
            ub_dense = ub.toarray() if isinstance(ub, spmatrix) else ub
            # If any dimension has [-inf, -inf] or [inf, inf], it's empty
            inf_mask = np.isinf(lb_dense) & np.isinf(ub_dense) & (np.sign(lb_dense) == np.sign(ub_dense))
            if np.any(inf_mask):
                # Set corresponding intervals to be empty
                lb[inf_mask] = np.nan
                ub[inf_mask] = np.nan

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
    
    @property
    def shape(self):
        """Shape property returning the shape of the interval bounds"""
        return self.inf.shape
    
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
                if len(inputs) == 2:
                    if inputs[0] is self:
                        return self.__add__(inputs[1])
                    else:
                        return self.__radd__(inputs[0])
            elif ufunc == np.subtract:
                if len(inputs) == 2:
                    if inputs[0] is self:
                        return self.__sub__(inputs[1])
                    else:
                        return self.__rsub__(inputs[0])
            elif ufunc == np.multiply:
                if len(inputs) == 2:
                    if inputs[0] is self:
                        return self.__mul__(inputs[1])
                    else:
                        return self.__rmul__(inputs[0])
            elif ufunc == np.matmul:
                if len(inputs) == 2:
                    # Handle matrix multiplication: numeric @ interval or interval @ numeric
                    if inputs[0] is self:
                        # interval @ numeric
                        return self.__matmul__(inputs[1])
                    else:
                        # numeric @ interval - call __rmatmul__
                        return self.__rmatmul__(inputs[0])
        
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

    def take(self, indices, axis=None):
        """Take elements from an interval along an axis."""
        new_inf = np.take(self.inf, indices, axis=axis)
        new_sup = np.take(self.sup, indices, axis=axis)
        return Interval(new_inf, new_sup) 
