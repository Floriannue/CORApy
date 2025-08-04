"""
test_zonotope_randPoint - unit test function of randPoint

Syntax:
    res = test_zonotope_randPoint

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       05-October-2024
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.check import compareMatrices


def test_zonotope_randPoint():
    """
    Unit test function of randPoint
    """
    # Tolerance
    tol = 1e-7
    
    # nD, empty zonotope
    n = 4
    Z = Zonotope.empty(n)
    p = Z.randPoint()
    assert p.shape == (n, 0)
    
    # 2D, zonotope that's a single point
    Z = Zonotope(np.array([2, 3]))
    p = Zonotope(np.array([2, 3])).randPoint(1)
    p_extr = Zonotope(np.array([2, 3])).randPoint(1, 'extreme')
    assert p.shape[1] == 1 and compareMatrices(p, Z.c.reshape(-1, 1), tol)
    assert p_extr.shape[1] == 1 and compareMatrices(p, Z.c.reshape(-1, 1), tol)
    
    # 3D, degenerate zonotope
    Z = Zonotope(np.array([1, 2, -1]), np.array([[1, 3, -2], [1, 0, 1], [2, 3, -1]]))
    numPoints = 10
    p_random = Z.randPoint(numPoints, 'standard')
    assert np.all(Z.contains_(p_random, 'exact', tol))
    p_random = Z.randPoint(numPoints, 'uniform')
    assert np.all(Z.contains_(p_random, 'exact', tol))
    p_random = Z.randPoint(numPoints, 'uniform:hitAndRun')
    assert np.all(Z.contains_(p_random, 'exact', tol))
    p_random = Z.randPoint(numPoints, 'uniform:billiardWalk')
    assert np.all(Z.contains_(p_random, 'exact', tol))
    
    # Less points than extreme points
    nrPtsExtreme = int(np.ceil(2**n * 0.5))
    p_random = Z.randPoint(nrPtsExtreme, 'extreme')
    assert np.all(Z.contains_(p_random, 'exact', tol))
    
    # As many points as extreme points
    nrPtsExtreme = int(np.ceil(2**n))
    p_random = Z.randPoint(nrPtsExtreme, 'extreme')
    assert np.all(Z.contains_(p_random, 'exact', tol))
    
    # More points than extreme points
    nrPtsExtreme = int(np.ceil(2**n * 5))
    p_random = Z.randPoint(nrPtsExtreme, 'extreme')
    assert np.all(Z.contains_(p_random, 'exact', tol))
    
    # 3D, parallelotope
    Z = Zonotope(np.array([1, 2, -1]), np.array([[1, 3, -2], [1, 0, 1], [4, 1, -1]]))
    numPoints = 10
    p_random = Z.randPoint(numPoints, 'standard')
    assert np.all(Z.contains_(p_random, 'exact', tol))
    p_random = Z.randPoint(numPoints, 'uniform')
    assert np.all(Z.contains_(p_random, 'exact', tol))
    p_random = Z.randPoint(numPoints, 'uniform:hitAndRun')
    assert np.all(Z.contains_(p_random, 'exact', tol))
    p_random = Z.randPoint(numPoints, 'uniform:billiardWalk')
    assert np.all(Z.contains_(p_random, 'exact', tol))
    
    # Less points than extreme points
    nrPtsExtreme = int(np.ceil(2**n * 0.5))
    p_random = Z.randPoint(nrPtsExtreme, 'extreme')
    assert np.all(Z.contains_(p_random, 'exact', tol))
    
    # As many points as extreme points
    nrPtsExtreme = int(np.ceil(2**n))
    p_random = Z.randPoint(nrPtsExtreme, 'extreme')
    assert np.all(Z.contains_(p_random, 'exact', tol))
    
    # More points than extreme points
    nrPtsExtreme = int(np.ceil(2**n * 5))
    p_random = Z.randPoint(nrPtsExtreme, 'extreme')
    assert np.all(Z.contains_(p_random, 'exact', tol))
    
    # Gaussian sampling (containment not guaranteed)
    p_random = Z.randPoint(10, 'gaussian')
    
    # 3D, general zonotope
    Z = Zonotope(np.array([0, 1, 1]), np.array([[0, -1, 1, 4, 2, 3, -2], [0, 1, 4, 2, 1, 3, -8], [0, 1, -3, 2, 1, 2, 6]]))
    numPoints = 10
    p_random = Z.randPoint(numPoints, 'standard')
    assert np.all(Z.contains_(p_random, 'exact', tol))
    p_random = Z.randPoint(numPoints, 'uniform')
    assert np.all(Z.contains_(p_random, 'exact', tol))
    p_random = Z.randPoint(numPoints, 'uniform:hitAndRun')
    assert np.all(Z.contains_(p_random, 'exact', tol))
    p_random = Z.randPoint(numPoints, 'uniform:billiardWalk')
    assert np.all(Z.contains_(p_random, 'exact', tol))
    
    # Less points than extreme points
    nrPtsExtreme = int(np.ceil(2**n * 0.5))
    p_random = Z.randPoint(nrPtsExtreme, 'extreme')
    assert np.all(Z.contains_(p_random, 'exact', tol))
    
    # As many points as extreme points
    nrPtsExtreme = int(np.ceil(2**n))
    p_random = Z.randPoint(nrPtsExtreme, 'extreme')
    assert np.all(Z.contains_(p_random, 'exact', tol))
    
    # More points than extreme points
    nrPtsExtreme = int(np.ceil(2**n * 5))
    p_random = Z.randPoint(nrPtsExtreme, 'extreme')
    assert np.all(Z.contains_(p_random, 'exact', tol))
    
    # Gaussian sampling (containment not guaranteed)
    p_random = Z.randPoint(10, 'gaussian')
    
    return True


if __name__ == "__main__":
    test_zonotope_randPoint()
    print("MATLAB test case passed!") 