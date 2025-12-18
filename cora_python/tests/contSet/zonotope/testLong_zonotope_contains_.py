"""
testLong_zonotope_contains_ - unit test function of contains_

Long-running tests for zonotope containment checking, including tests for all
methods and comprehensive containment checks across all ContSet types.

Authors: MATLAB: Matthias Althoff, Adrian Kulmburg
         Python: AI Assistant
Written: 26-July-2016 (MATLAB)
Last update: 21-January-2025 (MATLAB, added general containment checks)
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.interval.interval import Interval


class TestLongZonotopeContains:
    """Long-running tests for zonotope contains_ method"""
    
    def test_zonotope_parallelotope_all_methods(self):
        """Test zonotope x parallelotope containment with all methods"""
        # Create test zonotopes
        Z1 = Zonotope(np.array([[-4, -3, -2, -1], [1, 2, 3, 4]]))
        P1 = Zonotope(np.array([[-3.8, -4, 3], [1.2, 3, -4]]))
        P2 = Zonotope(np.array([[-3.8, -8, 2], [1.2, 10, -10]]))
        
        # Containment test with all methods
        res1, cert1, scaling1 = P1.contains_(Z1)
        res2, cert2, scaling2 = P2.contains_(Z1)
        # res1 and res2 are already booleans for set-set containment, not arrays
        assert not res1, "P1 should not contain Z1"
        assert res2, "P2 should contain Z1"
        
        # exact
        res1, cert1, scaling1 = P1.contains_(Z1, 'exact')
        res2, cert2, scaling2 = P2.contains_(Z1, 'exact')
        assert not res1, "P1 should not contain Z1 (exact)"
        assert res2, "P2 should contain Z1 (exact)"
        
        # exact:venum
        res1, cert1, scaling1 = P1.contains_(Z1, 'exact:venum')
        res2, cert2, scaling2 = P2.contains_(Z1, 'exact:venum')
        assert not res1, "P1 should not contain Z1 (exact:venum)"
        assert res2, "P2 should contain Z1 (exact:venum)"
        
        # exact:polymax
        res1, cert1, scaling1 = P1.contains_(Z1, 'exact:polymax')
        res2, cert2, scaling2 = P2.contains_(Z1, 'exact:polymax')
        assert not res1, "P1 should not contain Z1 (exact:polymax)"
        assert res2, "P2 should contain Z1 (exact:polymax)"
        
        # opt
        res1, cert1, scaling1 = P1.contains_(Z1, 'opt', 0, 200)
        res2, cert2, scaling2 = P2.contains_(Z1, 'opt', 0, 200)
        assert not res1, "P1 should not contain Z1 (opt)"
        assert res2, "P2 should contain Z1 (opt)"
        
        # approx
        res1, cert1, scaling1 = P1.contains_(Z1, 'approx')
        res2, cert2, scaling2 = P2.contains_(Z1, 'approx')
        assert not res1, "P1 should not contain Z1 (approx)"
        assert res2, "P2 should contain Z1 (approx)"
        
        # approx:st
        res1, cert1, scaling1 = P1.contains_(Z1, 'approx:st')
        res2, cert2, scaling2 = P2.contains_(Z1, 'approx:st')
        assert not res1, "P1 should not contain Z1 (approx:st)"
        assert res2, "P2 should contain Z1 (approx:st)"
        
        # approx:stDual
        res1, cert1, scaling1 = P1.contains_(Z1, 'approx:stDual')
        res2, cert2, scaling2 = P2.contains_(Z1, 'approx:stDual')
        assert not res1, "P1 should not contain Z1 (approx:stDual)"
        assert res2, "P2 should contain Z1 (approx:stDual)"
        
        # sampling
        res1, cert1, scaling1 = P1.contains_(Z1, 'sampling', 0, 200)
        res2, cert2, scaling2 = P2.contains_(Z1, 'sampling', 0, 200)
        assert not res1, "P1 should not contain Z1 (sampling)"
        assert res2, "P2 should contain Z1 (sampling)"
        
        # sampling:primal
        res1, cert1, scaling1 = P1.contains_(Z1, 'sampling:primal', 0, 200)
        res2, cert2, scaling2 = P2.contains_(Z1, 'sampling:primal', 0, 200)
        assert not res1, "P1 should not contain Z1 (sampling:primal)"
        assert res2, "P2 should contain Z1 (sampling:primal)"
        
        # sampling:dual
        res1, cert1, scaling1 = P1.contains_(Z1, 'sampling:dual', 0, 200)
        res2, cert2, scaling2 = P2.contains_(Z1, 'sampling:dual', 0, 200)
        assert not res1, "P1 should not contain Z1 (sampling:dual)"
        assert res2, "P2 should contain Z1 (sampling:dual)"
    
    def test_zonotope_point_containment(self):
        """Test zonotope x point containment"""
        n = 2
        nr_gen = 5
        np.random.seed(42)  # For reproducibility
        Z = Zonotope(np.zeros((n, 1)), np.random.rand(n, nr_gen))
        
        # Point inside: center of zonotope
        p_inside = Z.center()
        # Point outside: add all generators (with max of rand -> 1)
        p_outside = nr_gen * np.ones((n, 1))
        # Array of points
        num = 10
        p_array = nr_gen * (np.ones((n, num)) + np.random.rand(n, num))
        
        # Check if correct results for containment
        res, cert, scaling = Z.contains_(p_inside)
        assert np.all(res), "Center point should be contained"
        
        res, cert, scaling = Z.contains_(p_outside)
        assert not np.all(res), "Far point should not be contained"
        
        res, cert, scaling = Z.contains_(p_array)
        assert not np.all(res), "Not all points in array should be contained"
    
    def test_all_containment_combinations(self):
        """Test all containment combinations using checkAllContainments"""
        from cora_python.tests.g.functions.helper.sets.contSet.checkAllContainments import checkAllContainments
        
        # Create test sets
        I = Interval(np.array([-1, -1]), np.array([1, 1]))
        Ideg = Interval(np.array([-1, 0]), np.array([1, 0]))
        
        S = Zonotope(I)
        Sdeg = Zonotope(Ideg)
        Sempty = Zonotope.empty(2)
        
        implemented_sets = ['capsule', 'conPolyZono', 'conZonotope', 'interval', 'polytope',
                           'zonoBundle', 'zonotope', 'ellipsoid', 'taylm',
                           'polyZonotope', 'conPolyZono', 'spectraShadow']
        
        sets_non_exact = ['taylm', 'conPolyZono', 'polyZonotope']
        # 'spectraShadow' is generally not exact, but for the specific zonotope (an interval), it is.
        # However, checkAllContainments expects it that way...
        
        additional_algorithms = ['exact:venum', 'exact:polymax', 'approx:st', 'approx:stDual', 
                                'sampling', 'sampling:primal', 'sampling:dual']
        
        additional_algorithms_specific_sets = [
            ['conZonotope', 'interval', 'polytope', 'zonoBundle', 'zonotope'],  # exact:venum
            ['ellipsoid', 'conZonotope', 'interval', 'polytope', 'zonoBundle', 'zonotope'],  # exact:polymax
            ['zonotope'],  # approx:st
            ['zonotope'],  # approx:stDual
            ['zonotope'],  # sampling
            ['zonotope'],  # sampling:primal
            ['zonotope'],  # sampling:dual
        ]
        
        # This will test all containment combinations
        checkAllContainments(S, Sdeg, Sempty, implemented_sets, sets_non_exact, 
                           additional_algorithms, additional_algorithms_specific_sets)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

