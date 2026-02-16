"""
testLong_zonotope_minkDiff - unit test function of minkDiff

This module contains long unit tests for the zonotope minkDiff method, which computes
the Minkowski difference of two zonotopes.

Authors: Tobias Ladner
Written: 25-May-2023
Last update: 24-January-2024
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.interval import Interval
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def testLong_zonotope_minkDiff():
    """Unit test function of minkDiff - mirrors MATLAB testLong_zonotope_minkDiff.m"""
    tol = 1e-10
    
    # Quick test for all methods for syntax errors
    # Very simple tests; TODO: needs thorough testing
    # Can possibly be improved with new contains functionality:
    #     test containment, ignore if result not certified.
    
    methods = ['inner', 'outer', 'outer:coarse', 
               'approx', 'inner:conZonotope', 'inner:RaghuramanKoeln']
    
    for method in methods:
        try:
            # init
            n = 3
            difference = Zonotope.generateRandom(dimension=n, nr_generators=2*n)
            subtrahend = Zonotope.generateRandom(dimension=n, nr_generators=2*n)
            minuend = difference + subtrahend
            
            # minkDiff
            diff_comp = minuend.minkDiff(subtrahend, method)
            
        except Exception as e:
            pytest.fail(f"Method {method} failed: {e}")
    
    # Method 'exact' ----------------------------------------------------------
    method = 'exact'
    
    # Test 2-dimensional zonotopes
    n = 2
    for i in range(10):
        # init
        difference = Zonotope.generateRandom(dimension=n, nr_generators=2*n)
        subtrahend = Zonotope.generateRandom(dimension=n, nr_generators=2*n)
        minuend = difference + subtrahend
        
        # minkDiff
        diff_comp = minuend.minkDiff(subtrahend, method)
        
        # diff + sub = min => min - sub = diff
        assert np.allclose(difference.c, diff_comp.c, atol=tol)
        assert np.allclose(difference.G, diff_comp.G, atol=tol)
    
    # Test aligned
    for i in range(10):
        # init
        n = np.random.randint(1, 11)
        minuend = Zonotope.generateRandom(dimension=n, nr_generators=2*n)
        subtrahend = minuend * np.random.random()  # scale down
        
        # minkDiff
        difference = minuend.minkDiff(subtrahend, method)
        
        # min - sub = diff => diff + sub = min
        minuend_restored = Zonotope(difference.c + subtrahend.c, 
                                   np.hstack([difference.G, subtrahend.G]))
        assert np.allclose(minuend_restored.c, minuend.c, atol=tol)
        assert np.allclose(minuend_restored.G, minuend.G, atol=tol)
    
    # Test error for exact
    ns = [3, 5, 10]
    for n in ns:
        minuend = Zonotope.generateRandom(dimension=n, nr_generators=2*n)
        subtrahend = Zonotope.generateRandom(dimension=n, nr_generators=2*n)
        with pytest.raises(CORAerror):
            minuend.minkDiff(subtrahend, method)
    
    # Method 'outer:scaling' --------------------------------------------------
    method = 'outer:scaling'
    
    for i in range(10):
        # init
        n = np.random.randint(1, 11)
        difference = Zonotope.generateRandom(dimension=n, nr_generators=2*n)
        subtrahend = Interval.generateRandom('Dimension', n)
        minuend = difference + subtrahend
        
        # minkDiff
        with pytest.raises(Exception):  # Should throw an error
            minuend.minkDiff(subtrahend, method)
    
    # Test non full-dimensional zonotopes -------------------------------------
    n = 2
    for i in range(10):
        # init
        minuend = Zonotope(Interval.generateRandom('Dimension', n))
        subtrahend = minuend * np.random.random()  # scale down
        
        # project to higher dimensions
        P = np.array([[1, 0], [0, 1], [1, 1]])
        minuend_proj = P @ minuend
        subtrahend_proj = P @ subtrahend
        
        # minkDiff
        diff_comp_proj = minuend_proj.minkDiff(subtrahend_proj, 'exact')
        
        # compute svd projection matrix 
        U, S, _ = np.linalg.svd(minuend_proj.generators())
        newDim = np.sum(S > 1e-12)  # nr. of new dimensions
        P_minuend = U[:newDim, :]  # projection matrix
        
        # project down using svd projection matrix
        minuend_projected = P_minuend @ minuend_proj
        subtrahend_projected = P_minuend @ subtrahend_proj
        diff_comp = P_minuend @ diff_comp_proj
        
        # compute exact difference in projected space
        difference = minuend_projected.minkDiff(subtrahend_projected, 'exact')
        
        # test equality
        assert np.allclose(difference.c, diff_comp.c, atol=tol)
        assert np.allclose(difference.G, diff_comp.G, atol=tol)


if __name__ == '__main__':
    testLong_zonotope_minkDiff()
    print("testLong_zonotope_minkDiff passed!") 