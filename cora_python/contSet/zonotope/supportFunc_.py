"""
supportFunc_ - calculates the upper or lower bound of a zonotope along a certain direction

Syntax:
    val, x, fac = supportFunc_(Z, dir)
    val, x, fac = supportFunc_(Z, dir, type)

Inputs:
    Z - zonotope object
    dir - direction for which the bounds are calculated (vector)
    type - upper bound, lower bound, or both ('upper','lower','range')

Outputs:
    val - bound of the zonotope in the specified direction
    x - support vector
    fac - factor values that correspond to the upper bound

Example:
    from cora_python.contSet.zonotope import Zonotope, supportFunc_
    import numpy as np
    Z = Zonotope(np.array([[0], [0]]), np.eye(2))
    direction = np.array([[1], [0]])
    val, x, fac = supportFunc_(Z, direction, 'upper')
    print('Upper bound:', val)
    print('Support vector:', x)
    print('Factors:', fac)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/supportFunc, conZonotope/supportFunc_

Authors:       Niklas Kochdumper (MATLAB)
               Python translation by AI Assistant
Written:       19-November-2019 (MATLAB)
Last update:   10-December-2022 (MW, add type = 'range') (MATLAB)
Last revision: 27-March-2023 (MW, rename supportFunc_) (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import Tuple, Union, Optional
from .zonotope import Zonotope
from cora_python.contSet.interval.interval import Interval

def supportFunc_(Z: Zonotope, 
                 direction: np.ndarray, 
                 type_: str = 'upper',
                 *args, **kwargs) -> Tuple[Union[float, Interval], np.ndarray, np.ndarray]:
    """
    Calculates the upper or lower bound of a zonotope along a certain direction.
    """
    # Ensure direction is a column vector
    direction = np.asarray(direction)
    if direction.ndim == 1:
        direction = direction.reshape(-1, 1)
    
    # zonotope is empty if and only if the center is empty
    if Z.c.size == 0:
        x = np.array([])
        if type_ == 'upper':
            val = float('-inf')
        elif type_ == 'lower':
            val = float('inf')
        elif type_ == 'range':
            val = Interval(float('-inf'), float('inf'))
        else:
            raise ValueError(f"Invalid type '{type_}'. Use 'lower', 'upper', or 'range'.")
        return val, x, np.array([])
    
    # get object properties
    c = Z.c
    G = Z.G
    
    # project zonotope onto the direction
    c_ = float(direction.T @ c)  # Convert to scalar
    G_ = direction.T @ G
    
    # upper or lower bound
    if type_ == 'lower':
        val = c_ - float(np.sum(np.abs(G_)))  # Convert to scalar
        fac = -np.sign(G_).T
        
    elif type_ == 'upper':
        val = c_ + float(np.sum(np.abs(G_)))  # Convert to scalar
        fac = np.sign(G_).T
        
    elif type_ == 'range':
        val = Interval(c_ - float(np.sum(np.abs(G_))), c_ + float(np.sum(np.abs(G_))))  # Convert to scalar
        fac = np.column_stack([-np.sign(G_).T, np.sign(G_).T])
        
    else:
        raise ValueError(f"Invalid type '{type_}'. Use 'lower', 'upper', or 'range'.")
    
    # compute support vector
    x = c + G @ fac
    
    return val, x, fac 