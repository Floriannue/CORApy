"""
plus - Overloaded '+' operator for a Taylor model

Syntax:
    res = plus(factor1, factor2)

Inputs:
    factor1 and factor2 - a taylm objects
    order  - the cut-off order of the Taylor series. The constant term is
    the zero order.

Outputs:
    res - a taylm object

Example: 

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: taylm, minus, mtimes

References: 
  [1] K. Makino et al. "Taylor Models and other validated functional 
      inclusion methods"

Authors:       Dmitry Grebenyuk (MATLAB)
               Python translation by AI Assistant
Written:       20-April-2016 (MATLAB)
Last update:   21-July-2016 (DG, the polynomial part is changed to syms)
               18-July-2017 (DG, multivariable polynomial pack is added)
               21-August-2017 (DG, implementation fot matrices) 
               02-December-2017 (DG, new rank evaluation)
Python translation: 2025
"""

import numpy as np
from typing import Union, Any
from .taylm import Taylm


def plus(factor1: Union[Taylm, Union[int, float, np.ndarray]], 
         factor2: Union[Taylm, Union[int, float, np.ndarray]]) -> Taylm:
    """
    Overloaded '+' operator for a Taylor model
    
    Args:
        factor1: taylm object or numeric
        factor2: taylm object or numeric
        
    Returns:
        res: a taylm object
    """
    # Handle different cases
    if isinstance(factor1, Taylm) and isinstance(factor2, Taylm):
        # taylm + taylm
        return _taylm_plus_taylm(factor1, factor2)
    elif isinstance(factor1, Taylm) and isinstance(factor2, (int, float, np.ndarray)):
        # taylm + numeric
        return _taylm_plus_numeric(factor1, factor2)
    elif isinstance(factor1, (int, float, np.ndarray)) and isinstance(factor2, Taylm):
        # numeric + taylm
        return _taylm_plus_numeric(factor2, factor1)
    else:
        raise TypeError(f"Addition not supported between {type(factor1)} and {type(factor2)}")


def _taylm_plus_taylm(taylm1: Taylm, taylm2: Taylm) -> Taylm:
    """Add two Taylor models"""
    new_taylm = Taylm()
    
    # For now, simple addition of coefficients
    # This should be replaced with proper Taylor model arithmetic
    if taylm1.coefficients.size > 0 and taylm2.coefficients.size > 0:
        # Add coefficients (simplified)
        new_taylm.coefficients = taylm1.coefficients + taylm2.coefficients
        # For now, just copy monomials (this is simplified)
        new_taylm.monomials = taylm1.monomials.copy()
    else:
        # Handle empty cases
        if taylm1.coefficients.size > 0:
            new_taylm.coefficients = taylm1.coefficients.copy()
            new_taylm.monomials = taylm1.monomials.copy()
        else:
            new_taylm.coefficients = taylm2.coefficients.copy()
            new_taylm.monomials = taylm2.monomials.copy()
    
    # Copy other properties
    new_taylm.remainder = taylm1.remainder
    new_taylm.names_of_var = taylm1.names_of_var.copy()
    new_taylm.max_order = max(taylm1.max_order, taylm2.max_order)
    new_taylm.opt_method = taylm1.opt_method
    new_taylm.eps = taylm1.eps
    new_taylm.tolerance = taylm1.tolerance
    
    return new_taylm


def _taylm_plus_numeric(taylm_obj: Taylm, numeric_val: Union[int, float, np.ndarray]) -> Taylm:
    """Add Taylor model and numeric value"""
    new_taylm = Taylm()
    
    # Copy Taylor model properties
    new_taylm.coefficients = taylm_obj.coefficients.copy()
    new_taylm.monomials = taylm_obj.monomials.copy()
    new_taylm.remainder = taylm_obj.remainder
    new_taylm.names_of_var = taylm_obj.names_of_var.copy()
    new_taylm.max_order = taylm_obj.max_order
    new_taylm.opt_method = taylm_obj.opt_method
    new_taylm.eps = taylm_obj.eps
    new_taylm.tolerance = taylm_obj.tolerance
    
    # Add to constant term
    if new_taylm.coefficients.size > 0:
        new_taylm.coefficients[0] += float(numeric_val)
    else:
        new_taylm.coefficients = np.array([float(numeric_val)])
        new_taylm.monomials = [np.array([0])]
    
    return new_taylm
