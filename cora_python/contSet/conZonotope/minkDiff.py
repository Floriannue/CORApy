"""
minkDiff - compute the Minkowski difference of two constrained zonotopes:
         cZ1 - cZ2 = cZ <-> cZ + cZ2 \subseteq cZ1

Syntax:
    cZ = cZ1.minkDiff(S)
    cZ = cZ1.minkDiff(S, method)

Inputs:
    cZ1 - conZonotope object
    S - conZonotope object, contSet object, or numerical vector
    method - type of computation
           'exact' [1, Thm. 1]
           'inner:vertices'
           'inner:Vinod' [2, Alg. 1]

Outputs:
    cZ - conZonotope object after Minkowski difference

References:
    [1] M. Althoff, "On Computing the Minkowski Difference of Zonotopes"
    [2] A. Vinod, A. Weiss, S. Di Cairano, "Projection-free computation of
        robust controllable sets with constrained zonotopes",
        https://arxiv.org/pdf/2403.13730.pdf

Authors:       Niklas Kochdumper, Mark Wetzlinger
Written:       04-February-2021
Last update:   09-November-2022 (MW, rename 'minkDiff')
                02-April-2024 (MW, new method, refactor)
"""

import numpy as np
from typing import Union, TYPE_CHECKING
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues

if TYPE_CHECKING:
    from .conZonotope import ConZonotope
    from cora_python.contSet.zonotope import Zonotope
    from cora_python.contSet.interval import Interval


def minkDiff(self: 'ConZonotope', S: Union['ConZonotope', 'Zonotope', 'Interval', np.ndarray], 
             method: str = 'exact') -> 'ConZonotope':
    """
    Compute the Minkowski difference of two constrained zonotopes
    
    Args:
        S: conZonotope object, contSet object, or numerical vector
        method: type of computation ('exact', 'inner:vertices', 'inner:Vinod')
    
    Returns:
        conZonotope object after Minkowski difference
    """
    # Different algorithms for different set representations
    if isinstance(S, (int, float, np.ndarray)):
        return self + (-S)
    
    # Default method
    method = setDefaultValues(['exact'], [method])
    admissibleMethods = ['exact', 'inner:vertices', 'inner:Vinod']
    
    if method not in admissibleMethods:
        raise CORAerror('CORA:wrongValue', 'third',
                       f"has to be {', '.join(admissibleMethods)}")
    
    if method == 'exact':
        # Compute exact result according to Lemma 1 in [1]
        V = S.vertices_()
        cZ = self - V[:, 0:1]
        for i in range(1, V.shape[1]):
            cZ = cZ.and_(self - V[:, i:i+1], 'exact')
        return cZ
    
    elif method == 'inner:vertices':
        # Enclose subtrahend by zonotope
        Z = S.zonotope()
        
        # Compute Minkowski difference according to Theorem 1 in [1]
        c = Z.center()
        G = Z.generators()
        
        cZ = self + (-c)
        
        for i in range(G.shape[1]):
            cZ = cZ.and_(cZ + G[:, i:i+1], cZ + (-G[:, i:i+1]), 'exact')
        
        return cZ
    
    elif method == 'inner:Vinod':
        # Implements [2, Alg. 1]
        from cora_python.contSet.zonotope import Zonotope
        from cora_python.contSet.interval import Interval
        
        if not isinstance(S, (Zonotope, Interval)):
            raise CORAerror('CORA:notSupported')
        
        n = self.dim()
        M_C, N_C = self.A.shape
        c_S = S.center()
        S0 = S - c_S
        
        # Compute pseudo-inverse
        Gamma = np.linalg.pinv(np.vstack([self.G, self.A])) @ np.vstack([np.eye(n), np.zeros((M_C, n))])
        
        # Compute support function for each basis vector
        S0_sF = np.zeros(N_C)
        for i in range(N_C):
            basisvector = np.zeros((N_C, 1))
            basisvector[i] = 1
            S0_sF[i] = S0.supportFunc_((basisvector.T @ Gamma).T, 'upper')[0]
            if S0_sF[i] > 1:
                from cora_python.contSet.emptySet import EmptySet
                return EmptySet(n)
        
        D = np.diag(np.ones(N_C) - S0_sF)
        return ConZonotope(self.c - c_S, self.G @ D, self.A @ D, self.b)
    
    else:
        raise CORAerror('CORA:wrongValue', 'third',
                       f"has to be {', '.join(admissibleMethods)}") 