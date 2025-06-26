"""
Affine arithmetic class.

This class implements affine arithmetic, which is a method for range analysis.
It is a specific case of a Taylor model of order 1.

Authors: Dmitry Grebenyuk, Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 22-September-2017
Python translation: 2025
"""

from cora_python.contSet.taylm import Taylm
from cora_python.contSet.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.macros import CHECKS_ENABLED

class Affine(Taylm):
    """
    Affine arithmetic class, inheriting from Taylm.
    """
    def __init__(self, *varargin):
        """
        Constructor for affine objects.

        Syntax:
           obj = affine(interval_obj)
           obj = affine(lb, ub)
           obj = affine(interval_obj, name, opt_method, eps, tolerance)
        """
        if not varargin:
            raise CORAerror('CORA:noInputInSetConstructor')

        # 1. Parse input arguments
        int_obj, name, opt_method, eps, tolerance = _aux_parseInputArgs(*varargin)

        # 2. Check correctness of input arguments
        _aux_checkInputArgs(int_obj, name, opt_method, eps, tolerance, len(varargin))

        # 3. Compute properties (generate default names if needed)
        name = _gen_default_var_names(int_obj, name, [])

        # 4. Call superclass constructor (taylm) with order 1
        super().__init__(int_obj, 1, name, opt_method, eps, tolerance)

    def taylm(self):
        """Convert an affine object to a taylm object."""
        # Since Affine is already a Taylm, we can create a new Taylm
        # instance from its properties, or simply return a copy.
        return Taylm(self)

    def isemptyobject(self):
        """Check if the affine object is empty."""
        # By definition in the MATLAB code, affine objects are not allowed to be empty.
        return False

# Auxiliary functions
def _aux_parseInputArgs(*varargin):
    # Default values
    int_obj, name, opt_method, eps, tolerance = None, None, 'int', 0.001, 1e-8
    
    # Check if first arg is interval or lower bound
    if isinstance(varargin[0], Interval):
        int_obj = varargin[0]
        rem_args = varargin[1:]
    else:
        lb, ub = varargin[0], varargin[1]
        int_obj = Interval(lb, ub)
        rem_args = varargin[2:]
        
    name, opt_method, eps, tolerance = setDefaultValues([None, 'int', 0.001, 1e-8], rem_args)
    
    return int_obj, name, opt_method, eps, tolerance

def _aux_checkInputArgs(int_obj, name, opt_method, eps, tolerance, n_in):
    if CHECKS_ENABLED and n_in > 0:
        if opt_method not in ['int', 'bnb', 'bnbAdv']:
            raise CORAerror('CORA:wrongValue', 'third/fourth', "must be 'int', 'bnb', or 'bnbAdv'.")

def _gen_default_var_names(int_obj, name, varname):
    if name:
        return name
    
    dim = int_obj.dim()
    if dim > 1:
        return [f'x{i}' for i in range(1, dim + 1)]
    else:
        return ['x'] 