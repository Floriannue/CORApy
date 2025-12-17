"""
testLong_polytope_contains_ - unit test function for containment check

Long-running tests for polytope containment checking, including random tests
and comprehensive containment checks across all ContSet types.

Authors: MATLAB: Viktor Kotsev, Adrian Kulmburg
         Python: AI Assistant
Written: --- (MATLAB)
Last update: 21-January-2025 (MATLAB, added general containment checks)
"""

import numpy as np
import pytest
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.contains_ import contains_
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.contSet.interval.interval import Interval


class TestLongPolytopeContains:
    """Long-running tests for polytope contains_ method"""
    
    def test_random_polytope_point_containment(self):
        """Test random polytope point containment"""
        tol = 1e-6
        nr_tests = 25
        
        for i in range(nr_tests):
            # Random dimension
            n = np.random.randint(1, 6)
            
            # Create random polytope
            P = Polytope.generate_random('Dimension', n)
            
            # Check if randomly generated points are inside
            Y = P.randPoint_(n)
            res, cert, scaling = contains_(P, Y, 'exact', tol)
            assert np.all(res), f"Test {i+1}: Random point should be contained"
            assert cert, f"Test {i+1}: Should be certified"
    
    def test_random_polytope_containment(self):
        """Test random polytope containment"""
        tol = 1e-6
        nr_tests = 25
        
        for i in range(nr_tests):
            # Random dimension
            n = np.random.randint(1, 6)
            
            # Instantiate polytope
            P2 = Polytope.generate_random('Dimension', n)
            # Create larger polytope by increasing b
            P2.constraints()  # Ensure H-representation
            if hasattr(P2, 'b') and P2.b is not None:
                b_ = P2.b + 1
                P1 = Polytope(P2.A, b_)
            else:
                # If P2 doesn't have b, create a new random one
                P1 = Polytope.generate_random('Dimension', n)
                P1.constraints()  # Ensure H-representation
                # Make P1 larger by scaling
                if hasattr(P1, 'A') and P1.A is not None:
                    P1 = Polytope(P1.A, P1.b + 1) if hasattr(P1, 'b') and P1.b is not None else P1
            
            # Check containment
            res, cert, scaling = contains_(P1, P2)
            assert np.all(res), f"Test {i+1}: P2 should be contained in P1"
    
    def test_polytope_ellipsoid_containment(self):
        """Test polytope contains ellipsoid"""
        tol = 1e-6
        nr_tests = 25
        
        for i in range(nr_tests):
            # Random dimension
            n = np.random.randint(1, 6)
            
            # Generate zonotope
            Z = Zonotope.generateRandom('Dimension', n, 'NrGenerators', 5)
            # Create inner ellipsoid
            E = Ellipsoid(Z, method='inner:norm')
            # Convert zonotope to polytope
            P = Polytope(Z)
            
            # Check if E is in P
            res, cert, scaling = contains_(P, E, 'exact', tol)
            assert np.all(res), f"Test {i+1}: Ellipsoid should be contained in polytope"
    
    def test_all_containment_combinations(self):
        """Test all containment combinations using checkAllContainments"""
        from cora_python.tests.g.functions.helper.sets.contSet.checkAllContainments import checkAllContainments
        
        # Create test sets
        I = Interval(np.array([-1, -1]), np.array([1, 1]))
        Ideg = Interval(np.array([-1, 0]), np.array([1, 0]))
        
        S = Polytope(I)
        Sdeg = Polytope(Ideg)
        Sempty = Polytope.empty(2)
        
        implemented_sets = ['capsule', 'conPolyZono', 'conZonotope', 'interval', 'polytope',
                           'zonoBundle', 'zonotope', 'ellipsoid', 'taylm',
                           'polyZonotope', 'conPolyZono', 'spectraShadow']
        
        sets_non_exact = ['taylm', 'conPolyZono', 'polyZonotope']
        
        additional_algorithms = []
        additional_algorithms_specific_sets = []
        
        # This will test all containment combinations
        checkAllContainments(S, Sdeg, Sempty, implemented_sets, sets_non_exact, 
                           additional_algorithms, additional_algorithms_specific_sets)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

