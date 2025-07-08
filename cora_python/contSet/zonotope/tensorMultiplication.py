"""
tensorMultiplication method for zonotope class
"""

import numpy as np
from typing import List, Dict, Any
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def tensorMultiplication(Z: Zonotope, M: List[np.ndarray], options: Dict[str, Any]) -> Zonotope:
    """
    Computes {M_{ijk...l}*x_j*x_k*...*x_l|x \in Z} when the center of Z is the origin
    
    Args:
        Z: zonotope object
        M: tensor
        options: options dictionary containing list with perm, comb, and map
        
    Returns:
        Zonotope object
        
    Example:
        Z = Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))
        M = [np.random.rand(2, 2, 2)]  # 3D tensor
        options = {'list': {'perm': np.array([[0, 1], [1, 0]]), 
                           'comb': np.array([[0, 1]]), 
                           'map': np.array([0, 0])}}
        Zres = tensorMultiplication(Z, M, options)
    """
    # Check for None values
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Zonotope center or generators are None')
    
    # Retrieve dimension
    n = Z.c.shape[0]
    
    # Get center and generators
    c = Z.c
    G = Z.G
    
    # Get order of tensor for one dimension
    tensorOrder = len(M[0].shape)
    
    # Check if center is the origin
    if np.linalg.norm(c) == 0:
        # Generate list of permutations with replacement
        permList = options['list']['perm']
        # Generate list of combinations with replacement
        combList = options['list']['comb']
        
        # Initialize generator list
        H = np.zeros((n, len(combList)))
        
        # Go through all permutations
        for iPerm in range(len(permList)):
            # Choose generators
            genComb = []
            for i in range(tensorOrder):
                ind = permList[iPerm, i]
                genComb.append(G[:, ind:ind+1])
            genComb = np.hstack(genComb)
            
            # Do tensor multiplication
            if tensorOrder == 2:
                tildeH = _aux_generator_multiplication_2d(M, genComb)
            elif tensorOrder == 3:
                tildeH = _aux_generator_multiplication_3d(M, genComb)
            else:
                raise CORAerror('CORA:wrongValue', 
                               f'Tensor order {tensorOrder} not supported')
            
            # Add result to H
            # Map index
            cInd = options['list']['map'][iPerm]
            
            # Add to mapped index
            H[:, cInd] = H[:, cInd] + tildeH.flatten()
    else:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Center must be at origin for tensor multiplication')
    
    # Generate zonotope
    Zres = Zonotope(np.zeros((n, 1)), H)
    
    return Zres


def _aux_generator_multiplication_2d(M: List[np.ndarray], genComb: np.ndarray) -> np.ndarray:
    """
    Auxiliary function for 2D tensor multiplication
    
    Args:
        M: list of 2D tensors
        genComb: generator combination
        
    Returns:
        Result of tensor multiplication
    """
    # Obtain data
    n = len(M)
    
    # Initialize result
    res = np.zeros((n, 1))
    
    for iDim in range(n):
        for ind1 in range(n):
            for ind2 in range(n):
                res[iDim, 0] += M[iDim][ind1, ind2] * genComb[ind1, 0] * genComb[ind2, 1]
    
    return res


def _aux_generator_multiplication_3d(M: List[np.ndarray], genComb: np.ndarray) -> np.ndarray:
    """
    Auxiliary function for 3D tensor multiplication
    
    Args:
        M: list of 3D tensors
        genComb: generator combination
        
    Returns:
        Result of tensor multiplication
    """
    # Obtain data
    n = len(M)
    
    # Initialize result
    res = np.zeros((n, 1))
    
    for iDim in range(n):
        for ind1 in range(n):
            for ind2 in range(n):
                for ind3 in range(n):
                    res[iDim, 0] += M[iDim][ind1, ind2, ind3] * genComb[ind1, 0] * genComb[ind2, 1] * genComb[ind3, 2]
    
    return res 