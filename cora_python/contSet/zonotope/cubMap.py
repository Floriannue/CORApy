"""
cubMap - computes an enclosure of the set corresponding to the cubic 
   multiplication of a zonotope with a third-order tensor

Description:
    Calculates the following set:
    { z = (x' T x) * x | x \in Z }

    If three polyZonotopes are provided, the function calculates the set:
    { z = (x1' T x2) * x3 | x1 \in Z1, x2 \in Z2, x3 \in Z3 }

Syntax:
    res = cubMap(Z,T)
    res = cubMap(Z,T,ind)
    res = cubMap(Z1,Z2,Z3,T)
    res = cubMap(Z1,Z2,Z3,T,ind)

Inputs:
    Z,Z1,Z2,Z3 - zonotope objects
    T - third-order tensor
    ind - list containing the non-zero indices of the tensor

Outputs:
    res - zonotope object representing the set of the cubic mapping

Example: 
    % cubic multiplication
    Z = zonotope([1;-1],[1 3 -2 -1; 0 2 -1 1])
    
    T{1,1} = rand(2); T{1,2} = rand(2)
    T{2,1} = rand(2); T{2,2} = rand(2)

    Zcub = cubMap(Z,T)

    figure;
    subplot(1,2,1);
    plot(Z,[1,2],'FaceColor','r');
    subplot(1,2,2);
    plot(Zcub,[1,2],'FaceColor','b');

    % mixed cubic multiplication
    Z2 = zonotope([1;-1],[-1 3 -2 0 3; -2 1 0 2 -1]);
    Z3 = zonotope([1;-1],[-3 2 0; 2 1 -1]);

    ZcubMixed = cubMap(Z,Z2,Z3,T);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: quadMap

Authors:       Niklas Kochdumper (MATLAB)
               Python translation by AI Assistant
Written:       17-August-2018 (MATLAB)
Last update:   ---
Last revision: ---
Python translation: 2025
"""

import numpy as np
from typing import Union, List, Optional, Any
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from .zonotope import Zonotope


def cubMap(Z: Zonotope, *args) -> Zonotope:
    """
    Computes an enclosure of the set corresponding to the cubic 
    multiplication of a zonotope with a third-order tensor.
    
    Args:
        Z: Zonotope object
        *args: Variable arguments:
               - T: third-order tensor
               - ind: list containing the non-zero indices of the tensor (optional)
               - Z2, Z3: additional zonotopes for mixed cubic multiplication
               
    Returns:
        Zonotope: Object representing the set of the cubic mapping
        
    Raises:
        CORAerror: If inputs are invalid or computation fails
    """
    # Check number of input arguments
    if len(args) < 1 or len(args) > 4:
        raise CORAerror('CORA:wrongInput', 'Invalid number of arguments')
    
    # Cubic multiplication or mixed cubic multiplication
    if len(args) == 3 or len(args) == 4:
        # Assign input arguments
        Z2 = args[0]
        Z3 = args[1]
        T = args[2]
        
        # Parse optional input arguments
        if len(args) == 4:
            ind = args[3]
        else:
            temp = list(range(1, len(T[0]) + 1))
            ind = [temp] * len(T)
        
        # Check input arguments
        if not isinstance(Z, Zonotope):
            raise CORAerror('CORA:wrongValue', 'first', 'zonotope')
        if not isinstance(Z2, Zonotope):
            raise CORAerror('CORA:wrongValue', 'second', 'zonotope')
        if not isinstance(Z3, Zonotope):
            raise CORAerror('CORA:wrongValue', 'third', 'zonotope')
        if not isinstance(T, list):
            raise CORAerror('CORA:wrongValue', 'fourth', 'list')
        if not isinstance(ind, list):
            raise CORAerror('CORA:wrongValue', 'fifth', 'list')
        
        # Mixed cubic multiplication
        return _aux_cubMapMixed(Z, Z2, Z3, T, ind)
        
    elif len(args) == 1 or len(args) == 2:
        # res = cubMap(Z,T)
        # res = cubMap(Z,T,ind)
        
        # Assign input arguments
        T = args[0]
        
        # Parse optional input arguments
        if len(args) > 1:
            ind = args[1]
        else:
            temp = list(range(1, len(T[0]) + 1))
            ind = [temp] * len(T)
        
        # Check input arguments
        if not isinstance(Z, Zonotope):
            raise CORAerror('CORA:wrongValue', 'first', 'zonotope')
        if not isinstance(T, list):
            raise CORAerror('CORA:wrongValue', 'second', 'list')
        if not isinstance(ind, list):
            raise CORAerror('CORA:wrongValue', 'third', 'list')
        
        # Cubic multiplication
        return _aux_cubMapSingle(Z, T, ind)
    else:
        raise CORAerror('CORA:wrongInput', 'Invalid number of arguments')


def _aux_cubMapSingle(Z: Zonotope, T: List, ind: List) -> Zonotope:
    """Calculates the following set: { z = (x' T x) * x | x \in Z }"""
    
    # Initialize variables
    n = len(ind)
    N = Z.generators.shape[1] + 1
    Zcub = np.zeros((n, N**3))
    
    # Loop over all system dimensions
    for i in range(len(ind)):
        listQuad = [np.zeros((N, N)) for _ in range(N)]
        
        # Loop over all quadratic matrices: \sum_k (x' T_k x) * x_k 
        for k in range(len(ind[i])):
            # Quadratic evaluation
            Z_full = np.hstack([Z.center, Z.generators])
            quadMat = Z_full.T @ T[i][ind[i][k] - 1] @ Z_full
            
            # Add up all entries that correspond to identical factors
            temp = np.tril(quadMat, -1)
            quadMat = quadMat - temp
            quadMat = quadMat + temp.T
            
            # Multiply with the zonotope generators of the corresponding dimension
            for j in range(N):
                Zj = np.hstack([Z.center, Z.generators])
                listQuad[j] = listQuad[j] + quadMat * Zj[ind[i][k] - 1, j]
        
        # Add up all entries that belong to identical factors
        for k in range(1, N):
            # Loop over all quadratic matrix rows whose factors already appear in
            # one of the previous quadratic matrices
            for j in range(k):
                # Loop over all row entries
                for h in range(j, N):
                    if h <= k:
                        listQuad[j][h, k] = listQuad[j][h, k] + listQuad[k][j, h]
                    else:
                        listQuad[j][k, h] = listQuad[j][k, h] + listQuad[k][j, h]
        
        # Half the entries for purely quadratic factors
        temp = np.diag(listQuad[0])
        listQuad[0][0, 0] = listQuad[0][0, 0] + 0.5 * np.sum(temp[1:])
        
        for k in range(1, N):
            listQuad[0][k, k] = 0.5 * listQuad[0][k, k]
        
        # Summarize all identical factors in one matrix
        counter = 0
        
        for k in range(N):
            # Loop over all matrix rows that contain unique factors
            for j in range(k, N):
                m = N - j  # number of elements in the row
                Zcub[i, counter:counter + m] = listQuad[k][j, j:]
                counter += m
    
    # Concatenate the generator matrices
    Zcub = Zcub[:, :counter]
    
    # Construct the resulting zonotope
    return Zonotope(Zcub)


def _aux_cubMapMixed(Z1: Zonotope, Z2: Zonotope, Z3: Zonotope, 
                     T: List, ind: List) -> Zonotope:
    """Calculates the following set:
    { z = (x1' T x2) * x3 | x1 \in pZ1, x2 \in pZ2, x3 \in pZ3 }"""
    
    # Initialize variables
    n = len(ind)
    N1 = Z1.generators.shape[1] + 1
    N2 = Z2.generators.shape[1] + 1
    N3 = Z3.generators.shape[1] + 1
    Nq = N1 * N2
    
    Zcub = np.zeros((n, N1 * N2 * N3))
    
    # Loop over all system dimensions
    for i in range(len(ind)):
        # Loop over all quadratic matrices: \sum_k (x1' T_k x2) * x3_k 
        for k in range(len(ind[i])):
            # Quadratic evaluation
            Z1_full = np.hstack([Z1.center, Z1.generators])
            Z2_full = np.hstack([Z2.center, Z2.generators])
            quadMat = Z1_full.T @ T[i][ind[i][k] - 1] @ Z2_full
            quadVec = quadMat.flatten()
            
            # Multiply with Z3
            for j in range(N3):
                Z3j = np.hstack([Z3.center, Z3.generators])
                Zcub[i, (j-1)*Nq:(j*Nq)] = (Zcub[i, (j-1)*Nq:(j*Nq)] + 
                                            quadVec * Z3j[ind[i][k] - 1, j])
    
    # Construct the resulting zonotope
    return Zonotope(Zcub) 