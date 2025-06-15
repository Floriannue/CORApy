"""
Test file for zonotope randPoint_ method - translated from MATLAB

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 05-October-2024 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.zonotope.randPoint_ import randPoint_
from cora_python.contSet.contSet.contains import contains
from cora_python.g.functions.matlab.validate.check.compareMatrices import compareMatrices


def test_zonotope_randPoint():
    """Test zonotope randPoint_ method - translated from MATLAB"""
    
    # tolerance
    tol = 1e-7
    
    # nD, empty zonotope
    n = 4
    Z = Zonotope.empty(n)
    p = randPoint_(Z)
    assert p.shape == (n, 0)
    
    # 2D, zonotope that's a single point
    Z = Zonotope(np.array([2, 3]))
    p = randPoint_(Z, 1)
    p_extr = randPoint_(Z, 1, 'extreme')
    assert p.shape[1] == 1 and compareMatrices(p, Z.c.reshape(-1, 1), tol)
    assert p_extr.shape[1] == 1 and compareMatrices(p_extr, Z.c.reshape(-1, 1), tol)
    
    # 3D, degenerate zonotope
    c = np.array([1, 2, -1])
    G = np.array([[1, 3, -2], [1, 0, 1], [2, 3, -1]])
    Z = Zonotope(c, G)
    numPoints = 10
    
    # Test standard sampling
    p_random = randPoint_(Z, numPoints, 'standard')
    assert np.all(contains(Z, p_random, 'exact', tol))
    
    # Test uniform sampling
    p_random = randPoint_(Z, numPoints, 'uniform')
    assert np.all(contains(Z, p_random, 'exact', tol))
    
    # Test uniform:hitAndRun sampling
    p_random = randPoint_(Z, numPoints, 'uniform:hitAndRun')
    assert np.all(contains(Z, p_random, 'exact', tol))
    
    # Test uniform:billiardWalk sampling
    p_random = randPoint_(Z, numPoints, 'uniform:billiardWalk')
    assert np.all(contains(Z, p_random, 'exact', tol))
    
    # Test extreme sampling - less points than extreme points
    n = 3  # dimension
    nrPtsExtreme = int(np.ceil(2**n * 0.5))
    p_random = randPoint_(Z, nrPtsExtreme, 'extreme')
    assert np.all(contains(Z, p_random, 'exact', tol))
    
    # Test extreme sampling - as many points as extreme points
    nrPtsExtreme = int(np.ceil(2**n))
    p_random = randPoint_(Z, nrPtsExtreme, 'extreme')
    assert np.all(contains(Z, p_random, 'exact', tol))
    
    # Test extreme sampling - more points than extreme points
    nrPtsExtreme = int(np.ceil(2**n * 5))
    p_random = randPoint_(Z, nrPtsExtreme, 'extreme')
    assert np.all(contains(Z, p_random, 'exact', tol))
    
    # 3D, parallelotope
    c = np.array([1, 2, -1])
    G = np.array([[1, 3, -2], [1, 0, 1], [4, 1, -1]])
    Z = Zonotope(c, G)
    numPoints = 10
    
    # Test standard sampling
    p_random = randPoint_(Z, numPoints, 'standard')
    assert np.all(contains(Z, p_random, 'exact', tol))
    
    # Test uniform sampling
    p_random = randPoint_(Z, numPoints, 'uniform')
    assert np.all(contains(Z, p_random, 'exact', tol))
    
    # Test uniform:hitAndRun sampling
    p_random = randPoint_(Z, numPoints, 'uniform:hitAndRun')
    assert np.all(contains(Z, p_random, 'exact', tol))
    
    # Test uniform:billiardWalk sampling
    p_random = randPoint_(Z, numPoints, 'uniform:billiardWalk')
    assert np.all(contains(Z, p_random, 'exact', tol))
    
    # Test extreme sampling - less points than extreme points
    nrPtsExtreme = int(np.ceil(2**n * 0.5))
    p_random = randPoint_(Z, nrPtsExtreme, 'extreme')
    assert np.all(contains(Z, p_random, 'exact', tol))
    
    # Test extreme sampling - as many points as extreme points
    nrPtsExtreme = int(np.ceil(2**n))
    p_random = randPoint_(Z, nrPtsExtreme, 'extreme')
    assert np.all(contains(Z, p_random, 'exact', tol))
    
    # Test extreme sampling - more points than extreme points
    nrPtsExtreme = int(np.ceil(2**n * 5))
    p_random = randPoint_(Z, nrPtsExtreme, 'extreme')
    assert np.all(contains(Z, p_random, 'exact', tol))
    
    # Gaussian sampling (containment not guaranteed) - skip for now
    # p_random = randPoint_(Z, 10, 'gaussian', 0.8)
    
    # 3D, general zonotope
    c = np.array([0, 1, 1])
    G = np.array([[0, -1, 1, 4, 2, 3, -2], 
                  [0, 1, 4, 2, 1, 3, -8], 
                  [0, 1, -3, 2, 1, 2, 6]])
    Z = Zonotope(c, G)
    numPoints = 10
    
    # Test standard sampling
    p_random = randPoint_(Z, numPoints, 'standard')
    assert np.all(contains(Z, p_random, 'exact', tol))
    
    # Test uniform sampling
    p_random = randPoint_(Z, numPoints, 'uniform')
    assert np.all(contains(Z, p_random, 'exact', tol))
    
    # Test uniform:hitAndRun sampling
    p_random = randPoint_(Z, numPoints, 'uniform:hitAndRun')
    assert np.all(contains(Z, p_random, 'exact', tol))
    
    # Test uniform:billiardWalk sampling
    p_random = randPoint_(Z, numPoints, 'uniform:billiardWalk')
    assert np.all(contains(Z, p_random, 'exact', tol))
    
    # Test extreme sampling - less points than extreme points
    nrPtsExtreme = int(np.ceil(2**n * 0.5))
    p_random = randPoint_(Z, nrPtsExtreme, 'extreme')
    assert np.all(contains(Z, p_random, 'exact', tol))
    
    # Test extreme sampling - as many points as extreme points
    nrPtsExtreme = int(np.ceil(2**n))
    p_random = randPoint_(Z, nrPtsExtreme, 'extreme')
    assert np.all(contains(Z, p_random, 'exact', tol))
    
    # Test extreme sampling - more points than extreme points
    nrPtsExtreme = int(np.ceil(2**n * 5))
    p_random = randPoint_(Z, nrPtsExtreme, 'extreme')
    assert np.all(contains(Z, p_random, 'exact', tol))
    
    # Gaussian sampling (containment not guaranteed) - skip for now
    # p_random = randPoint_(Z, 10, 'gaussian', 0.8)


if __name__ == '__main__':
    test_zonotope_randPoint()
    print("All tests passed!") 