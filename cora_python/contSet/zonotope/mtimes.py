"""
mtimes - overloaded '*' operator for the multiplication of a matrix or an
    interval matrix with a zonotope

Syntax:
    Z = factor1 * factor2
    Z = mtimes(factor1, factor2)

Inputs:
    factor1 - zonotope object, numeric matrix or scalar
    factor2 - zonotope object, numeric scalar

Outputs:
    Z - zonotope object

Example:
    Z = zonotope([1, 1, 0], [[0, 0, 1]])
    M = [[1, 2], [1, 0]]
    Zmat = M * Z

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: plus

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       30-September-2006 (MATLAB)
Last update:   04-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from .zonotope import Zonotope

def mtimes(factor1, factor2):
    """
    Overloaded '*' operator for multiplication of matrix/scalar with zonotope
    
    Args:
        factor1: zonotope object, numeric matrix or scalar
        factor2: zonotope object, numeric scalar
        
    Returns:
        zonotope: Result of matrix multiplication
        
    Raises:
        CORAerror: If operation is not supported or dimensions don't match
    """
    
    try:
        # matrix/scalar * zonotope
        if isinstance(factor1, (int, float, np.number, list, tuple, np.ndarray)) and isinstance(factor2, Zonotope):
            factor1_mat = np.asarray(factor1)
            
            # Handle empty zonotope case
            if factor2.is_empty():
                if factor1_mat.ndim == 0 or (factor1_mat.ndim == 1 and factor1_mat.size == 1):
                    # Scalar * empty zonotope = empty zonotope of same dimension
                    return Zonotope.empty(factor2.dim())
                else:
                    # Matrix * empty zonotope = empty zonotope of matrix output dimension
                    if factor1_mat.ndim == 1:
                        factor1_mat = factor1_mat.reshape(1, -1)
                    return Zonotope.empty(factor1_mat.shape[0])
            
            # Handle scalar case
            if factor1_mat.ndim == 0 or (factor1_mat.ndim == 1 and factor1_mat.size == 1):
                scalar = factor1_mat.item() if factor1_mat.ndim > 0 else factor1_mat
                c = scalar * factor2.c
                G = scalar * factor2.G if factor2.G.size > 0 else factor2.G
                return Zonotope(c, G)
            
            # Handle matrix case
            if factor1_mat.ndim == 1:
                factor1_mat = factor1_mat.reshape(1, -1)
            
            # Check dimension compatibility
            if factor2.c.size > 0 and factor1_mat.shape[1] != factor2.dim():
                raise CORAerror('CORA:dimensionMismatch',
                              f'Matrix dimensions {factor1_mat.shape} incompatible with zonotope dimension {factor2.dim()}')
            
            # Apply linear transformation
            c = factor1_mat @ factor2.c if factor2.c.size > 0 else np.array([])
            G = factor1_mat @ factor2.G if factor2.G.size > 0 else np.zeros((factor1_mat.shape[0], 0))
            
            return Zonotope(c, G)
        
        # zonotope * matrix/scalar
        if isinstance(factor1, Zonotope) and isinstance(factor2, (int, float, np.number, list, tuple, np.ndarray)):
            factor2_mat = np.asarray(factor2)
            
            # Handle empty zonotope case
            if factor1.is_empty():
                if factor2_mat.ndim == 0 or (factor2_mat.ndim == 1 and factor2_mat.size == 1):
                    # Empty zonotope * scalar = empty zonotope of same dimension
                    return Zonotope.empty(factor1.dim())
                else:
                    # Empty zonotope * matrix = empty zonotope of matrix output dimension
                    if factor2_mat.ndim == 1:
                        factor2_mat = factor2_mat.reshape(-1, 1)
                    return Zonotope.empty(factor2_mat.shape[1])
            
            # Handle scalar case
            if factor2_mat.ndim == 0 or (factor2_mat.ndim == 1 and factor2_mat.size == 1):
                scalar = factor2_mat.item() if factor2_mat.ndim > 0 else factor2_mat
                c = scalar * factor1.c
                G = scalar * factor1.G if factor1.G.size > 0 else factor1.G
                return Zonotope(c, G)
            
            # Handle matrix case (right multiplication)
            if factor2_mat.ndim == 1:
                factor2_mat = factor2_mat.reshape(-1, 1)
            
            # Check dimension compatibility
            if factor1.c.size > 0 and factor1.dim() != factor2_mat.shape[0]:
                raise CORAerror('CORA:dimensionMismatch',
                              f'Zonotope dimension {factor1.dim()} incompatible with matrix dimensions {factor2_mat.shape}')
            
            # Apply linear transformation (right multiplication)
            # Handle the case where c is a column vector and needs to be transposed for right multiplication
            if factor1.c.size > 0:
                if factor1.c.ndim == 2 and factor1.c.shape[1] == 1:
                    # c is a column vector, transpose it for right multiplication
                    c = (factor1.c.T @ factor2_mat).T
                else:
                    c = factor1.c @ factor2_mat
            else:
                c = np.array([])
            
            if factor1.G.size > 0:
                G = factor1.G @ factor2_mat
            else:
                G = np.zeros((0, factor2_mat.shape[1]))
            
            return Zonotope(c, G)

    except Exception as e:
        # Check whether different dimension of ambient space
        if hasattr(factor1, 'dim') and hasattr(factor2, 'dim'):
            try:
                if factor1.dim() != factor2.dim():
                    raise CORAerror('CORA:dimensionMismatch',
                                  f'Dimension mismatch: {factor1.dim()} vs {factor2.dim()}')
            except:
                pass  # One of them might not have dim() method
        
        # Re-raise original error if it's already a CORAerror
        if isinstance(e, CORAerror):
            raise e
        
        # Convert other errors to CORAerror
        raise CORAerror('CORA:noops', f'Error in matrix multiplication: {str(e)}')
    
    # If we get here, operation is not supported
    raise CORAerror('CORA:noops', f'Operation not supported between {type(factor1)} and {type(factor2)}') 