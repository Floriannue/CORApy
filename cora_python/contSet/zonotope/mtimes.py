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

Authors:       Matthias Althoff
Written:       30-September-2006 
Last update:   04-October-2024
Last revision: ---
"""

import numpy as np
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
    """
    
    try:
        # matrix/scalar * zonotope
        if isinstance(factor1, (int, float, np.number, list, tuple, np.ndarray)):
            factor1_mat = np.asarray(factor1)
            # Handle scalar case
            if factor1_mat.ndim == 0:
                c = factor1_mat * factor2.c
                G = factor1_mat * factor2.G
            else:
                c = factor1_mat @ factor2.c
                G = factor1_mat @ factor2.G
            return Zonotope(c, G)
        
        # zonotope * matrix/scalar
        if isinstance(factor2, (int, float, np.number, list, tuple, np.ndarray)):
            factor2_mat = np.asarray(factor2)
            # Handle scalar case
            if factor2_mat.ndim == 0:
                c = factor1.c * factor2_mat
                G = factor1.G * factor2_mat
            else:
                c = factor1.c @ factor2_mat
                G = factor1.G @ factor2_mat
            return Zonotope(c, G)
    
    except Exception as e:
        # check whether different dimension of ambient space
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