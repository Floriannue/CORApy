"""
plus - Overloaded '+' operator for the Minkowski addition of a
   constrained zonotope with another set or vector

Syntax:
    S_out = cZ + S
    S_out = plus(cZ,S)

Inputs:
    cZ - conZonotope object, numeric
    S - contSet object, numeric

Outputs:
    S_out - conZonotope object

References: 
  [1] J. Scott et al. "Constrained zonotope: A new tool for set-based
      estimation and fault detection"

Authors:       Dmitry Grebenyuk, Niklas Kochdumper
Written:       05-December-2017 
Last update:   15-May-2018
                05-May-2020 (MW, standardized error message)
"""

import numpy as np
from typing import Union, TYPE_CHECKING
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from .conZonotope import ConZonotope
    from cora_python.contSet.zonotope import Zonotope
    from cora_python.contSet.interval import Interval


def plus(self: 'ConZonotope', S: Union['ConZonotope', 'Zonotope', 'Interval', np.ndarray]) -> 'ConZonotope':
    """
    Overloaded '+' operator for the Minkowski addition of a
    constrained zonotope with another set or vector
    
    Args:
        S: contSet object or numeric vector
    
    Returns:
        conZonotope object
    """
    from .conZonotope import ConZonotope
    
    # Ensure that numeric is second input argument
    if isinstance(S, (int, float, np.ndarray)):
        # Numeric: add to center
        if isinstance(S, np.ndarray) and S.ndim == 2 and S.shape[1] == 1:
            # Column vector
            result = ConZonotope(self.c + S, self.G, self.A, self.b)
            return result
        else:
            # Scalar or other numeric
            result = ConZonotope(self.c + S, self.G, self.A, self.b)
            return result
    
    # Sum with constrained zonotope
    if isinstance(S, ConZonotope):
        return aux_plus_conZonotope(self, S)
    
    # Other set representations: convert to conZonotope
    if isinstance(S, (Zonotope, Interval)):
        from cora_python.contSet.zonotope import conZonotope
        S_conZ = conZonotope(S)
        return aux_plus_conZonotope(self, S_conZ)
    
    # Check for empty sets
    if self.representsa('emptySet', 1e-12) or S.representsa('emptySet', 1e-12):
        return ConZonotope.empty(self.dim())
    
    # Other error
    raise CORAerror('CORA:noops', self, S)


def aux_plus_conZonotope(S_out: 'ConZonotope', S: 'ConZonotope') -> 'ConZonotope':
    """
    Equation (12) in reference paper [1]
    
    Args:
        S_out: first conZonotope
        S: second conZonotope
    
    Returns:
        conZonotope: sum of the two conZonotopes
    """
    from .conZonotope import ConZonotope
    
    # Add centers
    c_new = S_out.c + S.c
    
    # Concatenate generators
    G_new = np.hstack([S_out.G, S.G])
    
    # Block diagonal constraint matrix
    A_new = np.block([[S_out.A, np.zeros((S_out.A.shape[0], S.A.shape[1]))],
                      [np.zeros((S.A.shape[0], S_out.A.shape[1])), S.A]])
    
    # Concatenate constraint vectors
    b_new = np.vstack([S_out.b, S.b])
    
    return ConZonotope(c_new, G_new, A_new, b_new) 