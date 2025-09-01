"""
times - Overload '.*' operator for Taylor models

Syntax:
    res = times(factor1, factor2)

Inputs:
    factor1 - first taylm object
    factor2 - second taylm object

Outputs:
    res - resulting taylm object

Other m-files required: none
Subfunctions: multiply
MAT-files required: none

See also: taylm, plus, minus

References: 
  [1] K. Makino et al. "Taylor Models and other validated functional 
      inclusion methods"

Authors:       Niklas Kochdumper, Dmitry Grebenyuk (MATLAB)
               Python translation by AI Assistant
Written:       14-June-2017 (MATLAB)
Last update:   11-November-2017 (DG, extra cases are added)
               02-December-2017 (DG, new rank evaluation)
Python translation: 2025
"""

import numpy as np
from typing import Union, Any
from .taylm import Taylm


def times(factor1: Union[Taylm, Union[int, float, np.ndarray]], 
          factor2: Union[Taylm, Union[int, float, np.ndarray]]) -> Taylm:
    """
    Overload '.*' operator for Taylor models
    
    Args:
        factor1: first taylm object or numeric
        factor2: second taylm object or numeric
        
    Returns:
        res: resulting taylm object
    """
    # Handle different cases
    if isinstance(factor1, Taylm) and isinstance(factor2, Taylm):
        # taylm .* taylm
        return _taylm_times_taylm(factor1, factor2)
    elif isinstance(factor1, Taylm) and isinstance(factor2, (int, float, np.ndarray)):
        # taylm .* numeric
        return _taylm_times_numeric(factor1, factor2)
    elif isinstance(factor1, (int, float, np.ndarray)) and isinstance(factor2, Taylm):
        # numeric .* taylm
        return _taylm_times_numeric(factor2, factor1)
    else:
        raise TypeError(f"Element-wise multiplication not supported between {type(factor1)} and {type(factor2)}")


def _taylm_times_taylm(taylm1: Taylm, taylm2: Taylm) -> Taylm:
    """Element-wise multiplication of two Taylor models"""
    new_taylm = Taylm()
    
    # For now, simple multiplication of coefficients
    # This should be replaced with proper Taylor model arithmetic
    if taylm1.coefficients.size > 0 and taylm2.coefficients.size > 0:
        # Multiply coefficients (simplified)
        new_taylm.coefficients = taylm1.coefficients * taylm2.coefficients
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
    new_taylm.max_order = min(taylm1.max_order, taylm2.max_order)
    new_taylm.opt_method = taylm1.opt_method
    new_taylm.eps = taylm1.eps
    new_taylm.tolerance = taylm1.tolerance
    
    return new_taylm


def _taylm_times_numeric(taylm_obj: Taylm, numeric_val: Union[int, float, np.ndarray]) -> Taylm:
    """Element-wise multiplication of Taylor model and numeric value"""
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
    
    # Multiply by numeric value
    if new_taylm.coefficients.size > 0:
        new_taylm.coefficients = new_taylm.coefficients * float(numeric_val)
    
    return new_taylm
