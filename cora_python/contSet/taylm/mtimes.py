"""
mtimes - Overloaded '*' operator for a Taylor model

Syntax:
    res = mtimes(factor1, factor2)

Inputs:
    factor1 - taylm object
    factor2 - taylm object

Outputs:
    res - a taylm object

Example: 

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: taylm

Authors:       Dmitry Grebenyuk (MATLAB)
               Python translation by AI Assistant
Written:       20-August-2017 (MATLAB)
Last update:   ---
Python translation: 2025
"""

import numpy as np
from typing import Union, Any
from .taylm import Taylm
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def mtimes(factor1: Union[Taylm, np.ndarray], factor2: Union[Taylm, np.ndarray]) -> Taylm:
    """
    Overloaded '*' operator for a Taylor model
    
    Args:
        factor1: taylm object or numeric array
        factor2: taylm object or numeric array
        
    Returns:
        res: a taylm object
    """
    # Handle matrix multiplication: W @ taylm (for neural network layers)
    if isinstance(factor1, np.ndarray) and isinstance(factor2, Taylm):
        # Matrix * taylm: W @ taylm
        return _matrix_times_taylm(factor1, factor2)
    elif isinstance(factor1, Taylm) and isinstance(factor2, np.ndarray):
        # taylm * Matrix: taylm @ W
        return _taylm_times_matrix(factor1, factor2)
    elif isinstance(factor1, Taylm) and isinstance(factor2, Taylm):
        # taylm * taylm: matrix multiplication
        return _taylm_times_taylm(factor1, factor2)
    elif isinstance(factor1, (int, float)) and isinstance(factor2, Taylm):
        # scalar * taylm
        return _scalar_times_taylm(factor1, factor2)
    elif isinstance(factor1, Taylm) and isinstance(factor2, (int, float)):
        # taylm * scalar
        return _scalar_times_taylm(factor2, factor1)
    else:
        raise CORAerror('CORA:dimensionMismatch', factor1, factor2)


def _matrix_times_taylm(matrix: np.ndarray, taylm_obj: Taylm) -> Taylm:
    """Matrix multiplication: W @ taylm"""
    if matrix.ndim == 2:
        # Matrix multiplication
        result = np.zeros((matrix.shape[0], 1))
        for i in range(matrix.shape[0]):
            # For now, use center of Taylor model as approximation
            # This should be replaced with proper Taylor model arithmetic
            center_val = taylm_obj.coefficients[0] if taylm_obj.coefficients.size > 0 else 0.0
            result[i, 0] = np.sum(matrix[i, :]) * center_val
        
        # Create a new Taylor model with the result
        new_taylm = Taylm()
        new_taylm.coefficients = result.flatten()
        new_taylm.monomials = [[0]] * len(result)
        new_taylm.remainder = taylm_obj.remainder
        new_taylm.names_of_var = taylm_obj.names_of_var.copy()
        new_taylm.max_order = taylm_obj.max_order
        new_taylm.opt_method = taylm_obj.opt_method
        new_taylm.eps = taylm_obj.eps
        new_taylm.tolerance = taylm_obj.tolerance
        
        return new_taylm
    else:
        # Vector multiplication
        center_val = taylm_obj.coefficients[0] if taylm_obj.coefficients.size > 0 else 0.0
        result_val = np.sum(matrix) * center_val
        
        new_taylm = Taylm()
        new_taylm.coefficients = np.array([result_val])
        new_taylm.monomials = [[0]]
        new_taylm.remainder = taylm_obj.remainder
        new_taylm.names_of_var = taylm_obj.names_of_var.copy()
        new_taylm.max_order = taylm_obj.max_order
        new_taylm.opt_method = taylm_obj.opt_method
        new_taylm.eps = taylm_obj.eps
        new_taylm.tolerance = taylm_obj.tolerance
        
        return new_taylm


def _taylm_times_matrix(taylm_obj: Taylm, matrix: np.ndarray) -> Taylm:
    """Taylor model multiplication with matrix: taylm @ W"""
    # This case is less common in neural networks
    # For now, transpose and use the other case
    return _matrix_times_taylm(matrix.T, taylm_obj)


def _taylm_times_taylm(taylm1: Taylm, taylm2: Taylm) -> Taylm:
    """Taylor model multiplication: taylm * taylm"""
    # This would require proper Taylor model arithmetic
    # For now, create a simple approximation
    new_taylm = Taylm()
    
    # Multiply coefficients
    if taylm1.coefficients.size > 0 and taylm2.coefficients.size > 0:
        new_taylm.coefficients = taylm1.coefficients[0] * taylm2.coefficients[0]
    else:
        new_taylm.coefficients = np.array([0.0])
    
    # Combine monomials (this is simplified)
    new_taylm.monomials = [[0]]
    new_taylm.remainder = taylm1.remainder
    new_taylm.names_of_var = taylm1.names_of_var.copy()
    new_taylm.max_order = min(taylm1.max_order, taylm2.max_order)
    new_taylm.opt_method = taylm1.opt_method
    new_taylm.eps = taylm1.eps
    new_taylm.tolerance = taylm1.tolerance
    
    return new_taylm


def _scalar_times_taylm(scalar: Union[int, float], taylm_obj: Taylm) -> Taylm:
    """Scalar multiplication: scalar * taylm"""
    new_taylm = Taylm()
    new_taylm.coefficients = taylm_obj.coefficients * float(scalar)
    new_taylm.monomials = taylm_obj.monomials.copy()
    new_taylm.remainder = taylm_obj.remainder
    new_taylm.names_of_var = taylm_obj.names_of_var.copy()
    new_taylm.max_order = taylm_obj.max_order
    new_taylm.opt_method = taylm_obj.opt_method
    new_taylm.eps = taylm_obj.eps
    new_taylm.tolerance = taylm_obj.tolerance
    
    return new_taylm
