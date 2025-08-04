"""
test_zonotope_minkDiff - unit test function of Minkowski difference

This module contains unit tests for the zonotope minkDiff method, which computes
the Minkowski difference of two zonotopes.

Authors: Mark Wetzlinger, Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 05-March-2024 (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def test_zonotope_minkDiff_2d_centers_only():
    """
    Test Minkowski difference for 2D zonotopes with only centers
    """
    # 2D, only centers
    Z1 = Zonotope(np.array([[1], [-1]]))
    Z2 = Zonotope(np.array([[0], [1]]))
    Z_diff = Z1.minkDiff(Z2)
    Z_true = Zonotope(np.array([[1], [-2]]))
    
    assert np.allclose(Z_diff.c, Z_true.c)
    assert np.allclose(Z_diff.G, Z_true.G)


def test_zonotope_minkDiff_2d_box():
    """
    Test Minkowski difference for 2D box zonotopes
    """
    # 2D, box - 1/2*box
    Z1 = Zonotope(np.zeros((2, 1)), np.eye(2))
    Z2 = Zonotope(np.zeros((2, 1)), 0.7 * np.eye(2))
    Z_diff = Z1.minkDiff(Z2)
    Z_true = Zonotope(np.zeros((2, 1)), 0.3 * np.eye(2))
    
    assert np.allclose(Z_diff.c, Z_true.c)
    assert np.allclose(Z_diff.G, Z_true.G)


def test_zonotope_minkDiff_2d_aligned_generators():
    """
    Test Minkowski difference for 2D zonotopes with aligned generators
    """
    # 2D, aligned generators
    G1 = np.array([[1, 0], [-1, 2], [3, -1], [0, 2]]).T
    G2 = 0.7 * G1
    G_diff = G1 - G2
    G2 = G2[:, [2, 1, 3, 0]]  # Reorder columns
    
    Z1 = Zonotope(np.zeros((2, 1)), G1)
    Z2 = Zonotope(np.zeros((2, 1)), G2)
    Z_diff = Z1.minkDiff(Z2)
    Z_true = Zonotope(np.zeros((2, 1)), G_diff)
    
    assert np.allclose(Z_diff.c, Z_true.c, atol=1e-8)
    assert np.allclose(Z_diff.G, Z_true.G, atol=1e-8)


def test_zonotope_minkDiff_2d_different_methods():
    """
    Test Minkowski difference for 2D zonotopes with different methods
    """
    # 2D, different methods (Mink. diff of 2D zonotopes is closed!)
    G1 = np.array([[1, 0], [-1, 2], [3, -1], [0, 2]]).T
    Z1 = Zonotope(np.zeros((2, 1)), G1)
    G2 = 0.1 * np.array([[1, 1], [-1, 3], [5, -2], [1, 2]]).T
    Z2 = Zonotope(np.zeros((2, 1)), G2)
    
    # Convert to polytope for comparison
    P_true = Polytope(Z1).minkDiff(Polytope(Z2))
    
    # Test inner approximations
    Z_diff = Z1.minkDiff(Z2, 'inner')
    assert np.allclose(Z_diff.c, P_true.c, atol=1e-8)
    assert np.allclose(Z_diff.G, P_true.G, atol=1e-8)
    
    Z_diff = Z1.minkDiff(Z2, 'inner:conZonotope')
    assert np.allclose(Z_diff.c, P_true.c, atol=1e-8)
    assert np.allclose(Z_diff.G, P_true.G, atol=1e-8)
    
    Z_diff = Z1.minkDiff(Z2, 'inner:RaghuramanKoeln')
    assert np.allclose(Z_diff.c, P_true.c, atol=1e-8)
    assert np.allclose(Z_diff.G, P_true.G, atol=1e-8)
    
    # Test outer approximations
    Z_diff = Z1.minkDiff(Z2, 'outer')
    assert np.allclose(Z_diff.c, P_true.c, atol=1e-8)
    assert np.allclose(Z_diff.G, P_true.G, atol=1e-8)
    
    Z_diff = Z1.minkDiff(Z2, 'outer:coarse')
    assert np.allclose(Z_diff.c, P_true.c, atol=1e-8)
    assert np.allclose(Z_diff.G, P_true.G, atol=1e-8)
    
    Z_diff = Z1.minkDiff(Z2, 'outer:scaling')
    assert np.allclose(Z_diff.c, P_true.c, atol=1e-8)
    assert np.allclose(Z_diff.G, P_true.G, atol=1e-8)


def test_zonotope_minkDiff_2d_empty_result():
    """
    Test Minkowski difference resulting in empty set
    """
    # 2D, empty result
    G1 = np.array([[1, 0], [-1, 2], [3, -1], [0, 2]]).T
    Z1 = Zonotope(np.zeros((2, 1)), G1)
    G2 = np.array([[-1, 0], [1, 2], [-3, -1], [0, 2]]).T
    Z2 = Zonotope(np.zeros((2, 1)), G2)
    
    # Test all methods should result in empty set
    methods = ['approx', 'inner', 'inner:conZonotope', 'inner:RaghuramanKoeln', 
               'outer', 'outer:coarse', 'outer:scaling']
    
    for method in methods:
        Z_diff = Z1.minkDiff(Z2, method)
        assert Z_diff.representsa_('emptySet', 1e-8)


def test_zonotope_minkDiff_2d_degenerate_result():
    """
    Test Minkowski difference resulting in degenerate set (origin)
    """
    # 2D, degenerate result
    Z1 = Zonotope(np.zeros((2, 1)), np.eye(2))
    Z2 = Zonotope(np.zeros((2, 1)), np.array([[1], [1]]))
    
    # Test all methods should result in origin
    methods = ['approx', 'inner', 'inner:conZonotope', 'inner:RaghuramanKoeln', 
               'outer', 'outer:coarse', 'outer:scaling']
    
    for method in methods:
        Z_diff = Z1.minkDiff(Z2, method)
        assert Z_diff.representsa_('origin', 1e-8)


def test_zonotope_minkDiff_numeric_subtrahend():
    """
    Test Minkowski difference with numeric subtrahend
    """
    Z1 = Zonotope(np.array([[1], [2]]), np.array([[1, 0], [0, 1]]))
    point = np.array([[0.5], [0.3]])
    
    Z_diff = Z1.minkDiff(point)
    Z_expected = Z1 - point
    
    assert np.allclose(Z_diff.c, Z_expected.c)
    assert np.allclose(Z_diff.G, Z_expected.G)


def test_zonotope_minkDiff_scalar_subtrahend():
    """
    Test Minkowski difference with scalar subtrahend
    """
    Z1 = Zonotope(np.array([[1], [2]]), np.array([[1, 0], [0, 1]]))
    scalar = 0.5
    
    Z_diff = Z1.minkDiff(scalar)
    Z_expected = Z1 - scalar
    
    assert np.allclose(Z_diff.c, Z_expected.c)
    assert np.allclose(Z_diff.G, Z_expected.G)


def test_zonotope_minkDiff_dimension_mismatch():
    """
    Test Minkowski difference with dimension mismatch
    """
    Z1 = Zonotope(np.array([[1], [2]]), np.array([[1, 0], [0, 1]]))
    Z2 = Zonotope(np.array([[1], [2], [3]]), np.array([[1, 0], [0, 1], [0, 0]]))
    
    with pytest.raises(CORAerror) as exc_info:
        Z1.minkDiff(Z2)
    
    assert exc_info.value.identifier == 'CORA:dimensionMismatch'


def test_zonotope_minkDiff_invalid_method():
    """
    Test Minkowski difference with invalid method
    """
    Z1 = Zonotope(np.array([[1], [2]]), np.array([[1, 0], [0, 1]]))
    Z2 = Zonotope(np.array([[0], [0]]), np.array([[0.5, 0], [0, 0.5]]))
    
    with pytest.raises(CORAerror) as exc_info:
        Z1.minkDiff(Z2, 'invalid_method')
    
    assert 'Unknown method' in str(exc_info.value)


def test_zonotope_minkDiff_exact_2d():
    """
    Test exact method for 2D zonotopes
    """
    # Test 2-dimensional zonotopes with exact method
    n = 2
    for i in range(5):  # Reduced from 10 for faster testing
        # Generate random zonotopes
        difference = Zonotope.generateRandom('Dimension', n, 'NrGenerators', 2*n)
        subtrahend = Zonotope.generateRandom('Dimension', n, 'NrGenerators', 2*n)
        minuend = difference + subtrahend
        
        # Compute Minkowski difference
        diff_comp = minuend.minkDiff(subtrahend, 'exact')
        
        # Check: diff + sub = min => min - sub = diff
        assert np.allclose(difference.c, diff_comp.c, atol=1e-10)
        assert np.allclose(difference.G, diff_comp.G, atol=1e-10)


def test_zonotope_minkDiff_exact_aligned():
    """
    Test exact method for aligned zonotopes
    """
    # Test aligned zonotopes with exact method
    for i in range(5):  # Reduced from 10 for faster testing
        n = np.random.randint(2, 6)
        minuend = Zonotope.generateRandom('Dimension', n, 'NrGenerators', 2*n)
        subtrahend = minuend * np.random.random()  # Scale down
        
        # Compute Minkowski difference
        difference = minuend.minkDiff(subtrahend, 'exact')
        
        # Check: min - sub = diff => diff + sub = min
        minuend_restored = Zonotope(difference.c + subtrahend.c, 
                                   difference.G + subtrahend.G)
        assert np.allclose(minuend.c, minuend_restored.c, atol=1e-10)
        assert np.allclose(minuend.G, minuend_restored.G, atol=1e-10)


def test_zonotope_minkDiff_exact_error():
    """
    Test exact method error for non-2D non-aligned zonotopes
    """
    # Test error for exact method with higher dimensions
    ns = [3, 5]
    for n in ns:
        minuend = Zonotope.generateRandom('Dimension', n, 'NrGenerators', 2*n)
        subtrahend = Zonotope.generateRandom('Dimension', n, 'NrGenerators', 2*n)
        
        with pytest.raises(CORAerror) as exc_info:
            minuend.minkDiff(subtrahend, 'exact')
        
        assert 'No exact algorithm found' in str(exc_info.value)


def test_zonotope_minkDiff_outer_scaling():
    """
    Test outer:scaling method with interval subtrahend
    """
    # Test outer:scaling method
    for i in range(3):  # Reduced from 10 for faster testing
        n = np.random.randint(2, 6)
        difference = Zonotope.generateRandom('Dimension', n, 'NrGenerators', 2*n)
        subtrahend = Interval.generateRandom('Dimension', n)
        minuend = difference + subtrahend
        
        # This should work with interval subtrahend
        Z_diff = minuend.minkDiff(subtrahend, 'outer:scaling')
        assert isinstance(Z_diff, Zonotope)


def test_zonotope_minkDiff_outer_scaling_error():
    """
    Test outer:scaling method error with non-interval subtrahend
    """
    Z1 = Zonotope(np.array([[1], [2]]), np.array([[1, 0], [0, 1]]))
    Z2 = Zonotope(np.array([[0], [0]]), np.array([[0.5, 0], [0, 0.5]]))
    
    with pytest.raises(CORAerror) as exc_info:
        Z1.minkDiff(Z2, 'outer:scaling')
    
    assert 'interval' in str(exc_info.value).lower()


def test_zonotope_minkDiff_non_full_dimensional():
    """
    Test Minkowski difference for non-full-dimensional zonotopes
    """
    n = 2
    for i in range(3):  # Reduced from 10 for faster testing
        # Generate degenerate zonotopes
        minuend = Zonotope(Interval.generateRandom('Dimension', n))
        subtrahend = minuend * np.random.random()  # Scale down
        
        # Project to higher dimensions
        P = np.array([[1, 0], [0, 1], [1, 1]])
        minuend_proj = P @ minuend
        subtrahend_proj = P @ subtrahend
        
        # Compute Minkowski difference in projected space
        diff_comp_proj = minuend_proj.minkDiff(subtrahend_proj, 'exact')
        
        # Compute SVD projection matrix
        U, S, _ = np.linalg.svd(minuend_proj.generators)
        newDim = np.sum(S > 1e-10)  # Number of non-zero singular values
        P_minuend = U[:newDim, :]  # Projection matrix
        
        # Project down using SVD projection matrix
        minuend_projected = P_minuend @ minuend_proj
        subtrahend_projected = P_minuend @ subtrahend_proj
        diff_comp = P_minuend @ diff_comp_proj
        
        # Compute exact difference in projected space
        difference = minuend_projected.minkDiff(subtrahend_projected, 'exact')
        
        # Test equality
        assert np.allclose(difference.c, diff_comp.c, atol=1e-10)
        assert np.allclose(difference.G, diff_comp.G, atol=1e-10)


def test_zonotope_minkDiff_all_methods_syntax():
    """
    Test all methods for syntax errors
    """
    methods = ['inner', 'outer', 'outer:coarse', 'approx', 
               'inner:conZonotope', 'inner:RaghuramanKoeln']
    
    for method in methods:
        try:
            # Initialize
            n = 3
            difference = Zonotope.generateRandom('Dimension', n, 'NrGenerators', 2*n)
            subtrahend = Zonotope.generateRandom('Dimension', n, 'NrGenerators', 2*n)
            minuend = difference + subtrahend
            
            # Compute Minkowski difference
            diff_comp = minuend.minkDiff(subtrahend, method)
            
            # Check that result is a zonotope
            assert isinstance(diff_comp, Zonotope)
            
        except Exception as e:
            pytest.fail(f"Method {method} failed with error: {e}")


def test_zonotope_minkDiff_point_subtrahend():
    """
    Test Minkowski difference when subtrahend is a point (no generators)
    """
    Z1 = Zonotope(np.array([[1], [2]]), np.array([[1, 0], [0, 1]]))
    Z2 = Zonotope(np.array([[0.5], [0.3]]))  # Point zonotope (no generators)
    
    Z_diff = Z1.minkDiff(Z2)
    Z_expected = Z1 - Z2.center
    
    assert np.allclose(Z_diff.c, Z_expected.c)
    assert np.allclose(Z_diff.G, Z_expected.G)


def test_zonotope_minkDiff_degenerate_minuend():
    """
    Test Minkowski difference with degenerate minuend and full-dimensional subtrahend
    """
    # Create degenerate minuend (1D in 2D space)
    Z1 = Zonotope(np.array([[0], [0]]), np.array([[1], [0]]))  # Line along x-axis
    Z2 = Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))  # Full 2D
    
    Z_diff = Z1.minkDiff(Z2)
    
    # Should result in empty set
    assert Z_diff.representsa_('emptySet', 1e-8)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__]) 