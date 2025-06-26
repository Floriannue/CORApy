"""
Zoo class - Range bounding using intervals and Taylor models in parallel

A zoo object combines multiple numerical methods in parallel for
higher precision range bounding.

Properties:
    method: cell array containing the names of the applied methods as strings
    objects: cell array containing the class objects for the applied methods

Authors: Dmitry Grebenyuk, Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 05-November-2017 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, Optional, Any, Tuple, List, Dict
from ..contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.macros import CHECKS_ENABLED

class Zoo:
    """
    Zoo class - Range bounding using multiple methods in parallel
    
    A zoo object combines multiple numerical methods to achieve
    higher precision in range bounding computations.
    """
    
    def __init__(self, *varargin):
        """
        Constructor for zoo objects
        
        Args:
            *varargin: Variable arguments
                     - zoo(int, methods, [names, max_order, eps, tolerance])
                     - zoo(other_zoo): copy constructor
        """
        # 0. avoid empty instantiation
        if len(varargin) == 0:
            raise CORAerror('CORA:noInputInSetConstructor')
        
        # Check number of input arguments
        if len(varargin) < 2 or len(varargin) > 6:
            raise CORAerror('CORA:wrongInputInConstructor', f'Expected 2-6 arguments, got {len(varargin)}')

        # 1. copy constructor
        if len(varargin) == 1 and isinstance(varargin[0], Zoo):
            other = varargin[0]
            self.method = other.method.copy() if hasattr(other, 'method') else []
            self.objects = other.objects.copy() if hasattr(other, 'objects') else []
            return

        # 2. parse input arguments: varargin -> vars
        int_, methods, names, max_order, eps, tolerance = _aux_parseInputArgs(*varargin)

        # 3. check correctness of input arguments
        _aux_checkInputArgs(int_, methods, names, max_order, eps, tolerance, len(varargin))

        # 4. compute object
        method, objects = _aux_computeObject(int_, methods, names, max_order, eps, tolerance)

        # 5. assign properties
        self.method = method
        self.objects = objects

    # Mathematical operations
    def acos(self):
        """Arccosine function"""
        return self._zoo_computation(lambda x: np.arccos(x))
    
    def asin(self):
        """Arcsine function"""
        return self._zoo_computation(lambda x: np.arcsin(x))
    
    def atan(self):
        """Arctangent function"""
        return self._zoo_computation(lambda x: np.arctan(x))
    
    def cos(self):
        """Cosine function"""
        return self._zoo_computation(lambda x: np.cos(x))
    
    def cosh(self):
        """Hyperbolic cosine function"""
        return self._zoo_computation(lambda x: np.cosh(x))
    
    def exp(self):
        """Exponential function"""
        return self._zoo_computation(lambda x: np.exp(x))
    
    def log(self):
        """Natural logarithm function"""
        return self._zoo_computation(lambda x: np.log(x))
    
    def sin(self):
        """Sine function"""
        return self._zoo_computation(lambda x: np.sin(x))
    
    def sinh(self):
        """Hyperbolic sine function"""
        return self._zoo_computation(lambda x: np.sinh(x))
    
    def sqrt(self):
        """Square root function"""
        return self._zoo_computation(lambda x: np.sqrt(x))
    
    def tan(self):
        """Tangent function"""
        return self._zoo_computation(lambda x: np.tan(x))
    
    def tanh(self):
        """Hyperbolic tangent function"""
        return self._zoo_computation(lambda x: np.tanh(x))
    
    def __pow__(self, other):
        """Power operation"""
        return self._zoo_computation(lambda x, y: np.power(x, y), other)
    
    def power(self, other):
        """Element-wise power operation"""
        return self._zoo_computation(lambda x, y: np.power(x, y), other)
    
    def __add__(self, other):
        """Addition operation"""
        return self._zoo_computation(lambda x, y: x + y, other)
    
    def __sub__(self, other):
        """Subtraction operation"""
        return self._zoo_computation(lambda x, y: x - y, other)
    
    def __mul__(self, other):
        """Element-wise multiplication"""
        return self._zoo_computation(lambda x, y: x * y, other)
    
    def __matmul__(self, other):
        """Matrix multiplication"""
        return self._zoo_computation(lambda x, y: np.matmul(x, y), other)
    
    def __truediv__(self, other):
        """Division operation"""
        return self._zoo_computation(lambda x, y: x / y, other)
    
    def __neg__(self):
        """Unary minus"""
        return self._zoo_computation(lambda x: -x)
    
    def __pos__(self):
        """Unary plus"""
        return self._zoo_computation(lambda x: +x)

    def isemptyobject(self):
        """Check if object is empty"""
        return False
    
    def _zoo_computation(self, func, *args):
        """Apply computation to all methods in parallel"""
        result_zoo = Zoo.__new__(Zoo)
        result_zoo.method = self.method.copy()
        result_zoo.objects = []
        
        for i, obj in enumerate(self.objects):
            try:
                if args:
                    # Binary operation
                    if hasattr(obj, func.__name__):
                        result = getattr(obj, func.__name__)(*args)
                    else:
                        # Fallback to function application
                        result = func(obj, *args)
                else:
                    # Unary operation
                    if hasattr(obj, func.__name__):
                        result = getattr(obj, func.__name__)()
                    else:
                        # Fallback to function application
                        result = func(obj)
                result_zoo.objects.append(result)
            except Exception as e:
                # If operation fails for one method, skip it
                result_zoo.objects.append(None)
        
        return result_zoo

    def __getitem__(self, index):
        """Indexing operation"""
        if isinstance(index, int):
            # Single index - return zoo with indexed objects
            result_zoo = Zoo.__new__(Zoo)
            result_zoo.method = self.method.copy()
            result_zoo.objects = []
            
            for obj in self.objects:
                if hasattr(obj, '__getitem__'):
                    result_zoo.objects.append(obj[index])
                else:
                    result_zoo.objects.append(obj)
            
            return result_zoo
        else:
            # Multiple indices
            return self._zoo_computation(lambda x: x[index])

    def interval(self):
        """Convert to interval representation"""
        # Try to convert all methods to interval and take intersection
        intervals = []
        
        for obj in self.objects:
            if hasattr(obj, 'interval'):
                intervals.append(obj.interval())
            elif obj.__class__.__name__ == 'Interval':
                intervals.append(obj)
            else:
                # Try to convert using common methods
                try:
                    if hasattr(obj, 'inf') and hasattr(obj, 'sup'):
                        # Already an interval-like object
                        intervals.append(obj)
                    else:
                        # Try other conversion methods
                        continue
                except:
                    continue
        
        if intervals:
            # Return the tightest bounds (intersection of all methods)
            from ..interval import Interval
            inf_vals = [getattr(i, 'inf', i) for i in intervals]
            sup_vals = [getattr(i, 'sup', i) for i in intervals]
            
            # Take maximum of lower bounds and minimum of upper bounds
            inf_result = np.maximum.reduce(inf_vals) if len(inf_vals) > 1 else inf_vals[0]
            sup_result = np.minimum.reduce(sup_vals) if len(sup_vals) > 1 else sup_vals[0]
            
            return Interval(inf_result, sup_result)
        
        return None

    def __repr__(self):
        """String representation"""
        return f"Zoo(methods={len(self.method)}, objects={[type(obj).__name__ for obj in self.objects]})"


# Auxiliary functions -----------------------------------------------------

def _aux_parseInputArgs(*varargin) -> Tuple[Any, List[str], Optional[List[str]], int, float, float]:
    """Parse input arguments from user and assign to variables"""
    
    # set default values (first two always given)
    int_, methods, names, max_order, eps, tolerance = setDefaultValues(
        [None, [], [], 6, 0.001, 1e-8], list(varargin)
    )
    
    return int_, methods, names, max_order, eps, tolerance


def _aux_checkInputArgs(int_, methods: List[str], names: Optional[List[str]], 
                       max_order: int, eps: float, tolerance: float, n_in: int):
    """Check correctness of input arguments"""
    
    # only check if macro set to true
    if CHECKS_ENABLED and n_in > 0:
        
        # int has to be an interval
        if not hasattr(int_, 'inf') and not hasattr(int_, 'sup'):
            raise CORAerror('CORA:wrongValue', 'first', 'has to be an interval object.')
        
        # check methods
        if not isinstance(methods, list) or not all(isinstance(m, str) for m in methods):
            raise CORAerror('CORA:wrongValue', 'second', 'has to be a list of method strings.')
        
        valid_methods = {
            'taylm(int)', 'taylm(bnb)', 'taylm(bnbAdv)', 'taylm(linQuad)',
            'affine(int)', 'affine(bnb)', 'affine(bnbAdv)', 'interval'
        }
        
        for method in methods:
            if method not in valid_methods:
                raise CORAerror('CORA:wrongValue', 'second',
                    "Valid methods: 'taylm(int)', 'taylm(bnb)', 'taylm(bnbAdv)', 'taylm(linQuad)', " +
                    "'affine(int)', 'affine(bnb)', 'affine(bnbAdv)', 'interval'")
        
        # correct value for max_order
        if not isinstance(max_order, int) or max_order <= 0:
            raise CORAerror('CORA:wrongInputInConstructor',
                'Maximum order must be an integer greater than zero.')
        
        # correct value for eps
        if not isinstance(eps, (int, float)) or eps <= 0:
            raise CORAerror('CORA:wrongInputInConstructor',
                'Precision for branch and bound optimization must be a scalar greater than zero.')
        
        # correct value for tolerance
        if not isinstance(tolerance, (int, float)) or tolerance <= 0:
            raise CORAerror('CORA:wrongInputInConstructor',
                'Tolerance must be a scalar greater than zero.')


def _aux_computeObject(int_, methods: List[str], names: Optional[List[str]], 
                      max_order: int, eps: float, tolerance: float) -> Tuple[List[str], List]:
    """Compute object properties"""
    
    # generate variable names if they are not provided
    if names is None:
        if hasattr(int_, 'dim'):
            dim = int_.dim()
        elif hasattr(int_, 'inf'):
            dim = len(int_.inf) if hasattr(int_.inf, '__len__') else 1
        else:
            dim = 1
        names = [f'x{i+1}' for i in range(dim)]
    
    # sort methods alphabetically
    method = sorted(methods)
    
    # generate the objects
    objects = []
    
    from ..taylm import Taylm
    from ..affine import Affine

    for m in method:
        if m.startswith('taylm'):
            taylm_method = m.split('(')[1][:-1]
            objects.append(Taylm(int_, max_order, names, taylm_method, eps, tolerance))
        elif m.startswith('affine'):
            affine_method = m.split('(')[1][:-1]
            objects.append(Affine(int_, names, affine_method, eps, tolerance))
        elif m == 'interval':
            objects.append(int_)
    
    return method, objects 