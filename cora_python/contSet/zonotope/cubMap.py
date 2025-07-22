"""
cubMap - computes an enclosure of the set corresponding to the cubic 
   multiplication of a zonotope with a third-order tensor

Description:
    Calculates the following set:
    { z = (x' T x) * x | x \in Z }

    If three zonotopes are provided, the function calculates the set:
    { z = (x1' T x2) * x3 | x1 \in Z1, x2 \in Z2, x3 \in Z3 }

Syntax:
    res = cubMap(Z,T)
    res = cubMap(Z,T,ind)
    res = cubMap(Z1,Z2,Z3,T)
    res = cubMap(Z1,Z2,Z3,T,ind)

Inputs:
    Z,Z1,Z2,Z3 - zonotope objects
    T - third-order tensor (list of lists of numpy arrays)
    ind - list of lists containing the non-zero indices of the tensor (optional)

Outputs:
    res - zonotope object representing the set of the cubic mapping

Example:
    # cubic multiplication
    Z = Zonotope(np.array([[1],[-1]]), np.array([[1, 3, -2, -1], [0, 2, -1, 1]]))
    temp = np.random.rand(2,2)
    T = [[temp, temp], [temp, temp]]
    Zcub = cubMap(Z,T)

    # mixed cubic multiplication
    Z2 = Zonotope(np.array([[1],[-1]]), np.array([[-1, 3, -2, 0, 3], [-2, 1, 0, 2, -1]]))
    Z3 = Zonotope(np.array([[1],[-1]]), np.array([[-3, 2, 0], [2, 1, -1]]))
    ZcubMixed = cubMap(Z,Z2,Z3,T)

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
from typing import List
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from .zonotope import Zonotope

def cubMap(*args):
    """
    Computes an enclosure of the set corresponding to the cubic 
    multiplication of a zonotope with a third-order tensor.
    """
    # Argument parsing to match MATLAB logic
    if len(args) == 2 or len(args) == 3:
        # Single zonotope cubic multiplication
        Z = args[0]
        T = args[1]
        if len(args) == 3:
            ind = args[2]
        else:
            temp = list(range(1, len(T[0]) + 1))
            ind = [temp for _ in range(len(T))]
        return _aux_cubMapSingle(Z, T, ind)
    elif len(args) == 4 or len(args) == 5:
        # Mixed cubic multiplication
        Z1 = args[0]
        Z2 = args[1]
        Z3 = args[2]
        T = args[3]
        if len(args) == 5:
            ind = args[4]
        else:
            temp = list(range(1, len(T[0]) + 1))
            ind = [temp for _ in range(len(T))]
        return _aux_cubMapMixed(Z1, Z2, Z3, T, ind)
    else:
        raise CORAerror('CORA:wrongInput', 'Invalid number of arguments for cubMap')

def _aux_cubMapSingle(Z: Zonotope, T: List, ind: List) -> Zonotope:
    """Calculates the set: { z = (x' T x) * x | x in Z }"""
    n = len(ind)
    N = Z.generators.shape[1] + 1
    Zcub = np.zeros((n, N**3))
    for i in range(n):
        listQuad = [np.zeros((N, N)) for _ in range(N)]
        for k in range(len(ind[i])):
            Z_full = np.hstack([Z.center, Z.generators])
            quadMat = Z_full.T @ T[i][ind[i][k] - 1] @ Z_full
            temp = np.tril(quadMat, -1)
            quadMat = quadMat - temp + temp.T
            for j in range(N):
                Zj = np.hstack([Z.center, Z.generators])
                listQuad[j] += quadMat * Zj[ind[i][k] - 1, j]
        for k in range(1, N):
            for j in range(k):
                for h in range(j, N):
                    if h <= k:
                        listQuad[j][h, k] += listQuad[k][j, h]
                    else:
                        listQuad[j][k, h] += listQuad[k][j, h]
        temp_diag = np.diag(listQuad[0])
        listQuad[0][0, 0] += 0.5 * np.sum(temp_diag[1:])
        for k in range(1, N):
            listQuad[0][k, k] = 0.5 * listQuad[0][k, k]
        counter = 0
        for k in range(N):
            for j in range(k, N):
                m = N - j
                Zcub[i, counter:counter + m] = listQuad[k][j, j:]
                counter += m
    Zcub = Zcub[:, :counter]
    return Zonotope(Zcub)

def _aux_cubMapMixed(Z1: Zonotope, Z2: Zonotope, Z3: Zonotope, T: List, ind: List) -> Zonotope:
    """Calculates the set: { z = (x1' T x2) * x3 | x1 in Z1, x2 in Z2, x3 in Z3 }"""
    n = len(ind)
    N1 = Z1.generators.shape[1] + 1
    N2 = Z2.generators.shape[1] + 1
    N3 = Z3.generators.shape[1] + 1
    Nq = N1 * N2
    Zcub = np.zeros((n, N1 * N2 * N3))
    for i in range(n):
        for k in range(len(ind[i])):
            Z1_full = np.hstack([Z1.center, Z1.generators])
            Z2_full = np.hstack([Z2.center, Z2.generators])
            quadMat = Z1_full.T @ T[i][ind[i][k] - 1] @ Z2_full
            quadVec = quadMat.flatten()
            for j in range(N3):
                Z3j = np.hstack([Z3.center, Z3.generators])
                Zcub[i, (j)*Nq:(j+1)*Nq] += quadVec * Z3j[ind[i][k] - 1, j]
    # The output should be the full matrix as in MATLAB: Zonotope(Zcub)
    return Zonotope(Zcub) 