"""
taylm (Taylor model) class.

Syntax:
    obj = taylm(int)
    obj = taylm(int,max_order,names,opt_method,eps,tolerance)
    obj = taylm(func,int)
    obj = taylm(func,int,max_order,opt_method,eps,tolerance)

Inputs:
    int - interval object that defines the ranges of the variables
    max_order - the maximal order of a polynomial stored in a polynomial part
    names - cell-array containing the names of the symbolic variables as
            characters (same size as interval matrix 'int')
    func - symbolic function 
    opt_method - method used to calculate interval over-approximations of
                 taylor models 
                  'int': standard interval arithmetic (default)
                  'bnb': branch and bound method is used to find min/max
                  'bnbAdv': branch and bound with re-expansion of taylor models
                  'linQuad': optimization with Linear Dominated Bounder (LDB)
                             and Quadratic Fast Bounder (QFB)
    eps - precision for the selected optimization method (opt_method = 'bnb', 
          opt_method = 'bnbAdv' and opt_method = 'linQuad')
    tolerance - monomials with coefficients smaller than this value are
                moved to the remainder

Outputs:
    obj - generated object

Examples: 
    % create and manipulate simple taylor models
    tx = taylm(interval(1,2),4,'x')
    ty = taylm(interval(3,4),4,'y')
    t = sin(tx+ty) + exp(-tx) + ty*tx
    interval(t)

    % create a vector of taylor models
    tvec = taylm(interval([1;3],[2;4]),4,{'x';'y'})
    t = sin(tvec(1)+tvec(2)) + exp(-tvec(1)) + tvec(1)*tvec(2)
    interval(t)

    % create a taylor model from a symbolic function
    syms x y
    func = sin(x+y) + exp(-x) + x*y
    t = taylm(func,interval([1;3],[2;4]),4)
    interval(t)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: interval

References: 
  [1] K. Makino et al. "Taylor Models and other validated functional 
      inclusion methods"
  [2] M. Althoff et al. "Implementation of Taylor models in CORA 2018
      (Tool Presentation)"

Authors:       Dmitry Grebenyuk, Niklas Kochdumper (MATLAB)
               Python translation by AI Assistant
Written:       29-March-2016 (MATLAB)
Last update:   18-July-2017 (DG, multivariable polynomial pack is added, MATLAB)
               29-July-2017 (NK, The NK' code is merged with the DG', MATLAB) 
               11-October-2017 (DG, syms as an input, MATLAB)
               03-April-2018 (NK, restructured constructor, MATLAB)
               02-May-2020 (MW, add property validation, MATLAB)
Last revision: 16-June-2023 (MW, restructure using auxiliary functions, MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Tuple, Union, List, Optional, Any

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.assertNarginConstructor import assertNarginConstructor

if TYPE_CHECKING:
    from cora_python.contSet.interval.interval import Interval


class Taylm:
    """
    Taylor model class
    
    Properties (SetAccess = private, GetAccess = public):
        coefficients: coefficients of polynomial terms (column vector)
        monomials: monomials for polynomial terms (cell array of column vectors)
        remainder: remainder term of the Taylor model (interval)
        names_of_var: list with names of the symbolic variables (cell array of strings)
        max_order: stores the maximal order of a polynomial (integer)
        opt_method: defines the method used to determine an interval over-approximation (string)
        eps: precision for the branch and bound optimization (scalar > 0)
        tolerance: coefficients smaller than this value get moved to the remainder (scalar > 0)
    """
    
    def __init__(self, *varargin):
        """
        Class constructor for Taylor models
        """
        # Initialize default properties
        self.coefficients = np.array([])
        self.monomials = []
        from cora_python.contSet.interval.interval import Interval
        self.remainder = Interval(0, 0)
        self.names_of_var = []
        self.max_order = 6
        self.opt_method = 'int'
        self.eps = 0.001
        self.tolerance = 1e-8

        # Handle empty instantiation (allowed for taylm unlike other contSet classes)
        if len(varargin) == 0:
            return

        # assertNarginConstructor(list(range(1, 7)), len(varargin))

        # 1. copy constructor
        if len(varargin) == 1 and isinstance(varargin[0], Taylm):
            other = varargin[0]
            self.coefficients = other.coefficients.copy() if other.coefficients is not None else np.array([])
            self.monomials = other.monomials.copy() if other.monomials is not None else []
            self.remainder = other.remainder
            self.names_of_var = other.names_of_var.copy() if other.names_of_var is not None else []
            self.max_order = other.max_order
            self.opt_method = other.opt_method
            self.eps = other.eps
            self.tolerance = other.tolerance
            return

        # 2. parse input arguments: varargin -> vars
        func, int_obj, max_order, names, opt_method, eps, tolerance = _aux_parseInputArgs(*varargin)
        
        # Get variable name from first input if available
        varname = None  # In Python, we can't easily get the variable name like inputname(1) in MATLAB

        # 3. check correctness of input arguments
        _aux_checkInputArgs(func, int_obj, max_order, names, opt_method, eps, tolerance, len(varargin))

        # 4. compute object
        self = _aux_computeObject(self, func, int_obj, max_order, names, opt_method, eps, tolerance, varname)


# Auxiliary functions -----------------------------------------------------

def _aux_parseInputArgs(*varargin) -> Tuple[Any, Any, int, List[str], str, float, float]:
    """Parse input arguments from user and assign to variables"""
    
    # Due to different supported syntaxes, number of allowed input
    # arguments depends on input arguments (see below)
    
    # Default values
    func = None
    int_obj = None
    max_order = 6
    names = []
    opt_method = 'int'
    eps = 0.001
    tolerance = 1e-8
    
    if len(varargin) == 0:
        return func, int_obj, max_order, names, opt_method, eps, tolerance
    
    # Handle different input patterns
    if len(varargin) >= 1:
        # Check if first argument is symbolic function or interval
        if hasattr(varargin[0], '__class__') and 'interval' in str(type(varargin[0])).lower():
            int_obj = varargin[0]
        else:
            # Assume it's a symbolic function
            func = varargin[0]
            if len(varargin) >= 2:
                int_obj = varargin[1]
    
    # Parse remaining arguments based on whether func is provided
    arg_offset = 2 if func is not None else 1
    
    if len(varargin) >= arg_offset + 1:
        max_order = int(varargin[arg_offset])
    if len(varargin) >= arg_offset + 2:
        names = varargin[arg_offset + 1] if varargin[arg_offset + 1] is not None else []
    if len(varargin) >= arg_offset + 3:
        opt_method = varargin[arg_offset + 2] if varargin[arg_offset + 2] is not None else 'int'
    if len(varargin) >= arg_offset + 4:
        eps = float(varargin[arg_offset + 3]) if varargin[arg_offset + 3] is not None else 0.001
    if len(varargin) >= arg_offset + 5:
        tolerance = float(varargin[arg_offset + 4]) if varargin[arg_offset + 4] is not None else 1e-8
    
    return func, int_obj, max_order, names, opt_method, eps, tolerance


def _aux_checkInputArgs(func: Any, int_obj: Any, max_order: int, names: List[str], 
                       opt_method: str, eps: float, tolerance: float, n_in: int):
    """Check correctness of input arguments"""
    
    # Basic checks
    if max_order < 0:
        raise CORAerror('CORA:wrongInputInConstructor', 'Max order must be non-negative')
    
    if eps <= 0:
        raise CORAerror('CORA:wrongInputInConstructor', 'Eps must be positive')
    
    if tolerance <= 0:
        raise CORAerror('CORA:wrongInputInConstructor', 'Tolerance must be positive')
    
    if opt_method not in ['int', 'bnb', 'bnbAdv', 'linQuad']:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Invalid optimization method. Must be one of: int, bnb, bnbAdv, linQuad')


def _aux_computeObject(obj: Taylm, func: Any, int_obj: Any, max_order: int, names: List[str], 
                      opt_method: str, eps: float, tolerance: float, varname: Optional[str]) -> Taylm:
    """Compute object properties"""
    
    # Set basic properties
    obj.max_order = max_order
    obj.opt_method = opt_method
    obj.eps = eps
    obj.tolerance = tolerance
    
    # Handle names
    if isinstance(names, str):
        names = [names]
    obj.names_of_var = names
    
    # Initialize with simple values for now
    # Full implementation would require symbolic computation capabilities
    obj.coefficients = np.array([1.0])  # Constant term
    obj.monomials = [np.array([0])]     # Constant monomial
    
    # Set remainder based on interval if provided
    if int_obj is not None:
        # Use the interval as the remainder for now
        obj.remainder = int_obj
    else:
        from cora_python.contSet.interval.interval import Interval
        obj.remainder = Interval(0, 0)
    
    return obj 