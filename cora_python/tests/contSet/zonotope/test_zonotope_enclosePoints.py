"""
test_zonotope_enclosePoints - unit test function for Zonotope.enclosePoints

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope.zonotope import Zonotope


def test_zonotope_enclosePoints():
    """Test enclosePoints method for Zonotope."""
    
    # Test points from MATLAB test
    p = np.array([
        [1, 3, -2, 4, 3, -1, 1, 0],
        [2, -1, 1, -3, 2, 1, 0, 1]
    ])
    
    # Compute enclosing zonotope with default method (maiga)
    Z = Zonotope.enclosePoints(p)
    
    # Compute with different method
    Z_stursberg = Zonotope.enclosePoints(p, 'stursberg')
    
    # Check if all points are contained
    # Use contains method which returns just the boolean result
    assert Z.contains(p), "Default method should contain all points"
    assert Z_stursberg.contains(p), "Stursberg method should contain all points"
    
    # Test that both zonotopes are valid (have correct dimensions)
    assert Z.dim() == 2, "Zonotope should have 2 dimensions"
    assert Z_stursberg.dim() == 2, "Zonotope should have 2 dimensions"
    
    # Test that zonotopes have reasonable properties
    assert Z.c.shape[0] == 2, "Center should be 2D"
    assert Z.G.shape[0] == 2, "Generators should have 2 rows"
    assert Z_stursberg.c.shape[0] == 2, "Center should be 2D"
    assert Z_stursberg.G.shape[0] == 2, "Generators should have 2 rows"


def test_zonotope_enclosePoints_edge_cases():
    """Test edge cases for enclosePoints method."""
    
    # Single point
    single_point = np.array([[2], [3]])
    Z_single = Zonotope.enclosePoints(single_point)
    
    # Check that the single point is contained
    assert Z_single.contains(single_point), "Single point should be contained"
    assert Z_single.dim() == 2, "Dimension should be preserved"
    
    # Two identical points
    identical_points = np.array([[1, 1], [2, 2]])
    Z_identical = Zonotope.enclosePoints(identical_points)
    assert Z_identical.contains(identical_points), "Identical points should be contained"
    
    # 1D case
    points_1d = np.array([1, 4, 2, 5, 3])
    Z_1d = Zonotope.enclosePoints(points_1d)
    assert Z_1d.dim() == 5, "1D input should be treated as 5 dimensions with 1 point each"
    # Convert points to proper format for containment check
    points_1d_matrix = points_1d.reshape(-1, 1)
    assert Z_1d.contains(points_1d_matrix), "1D points should be contained"


def test_zonotope_enclosePoints_methods():
    """Test different methods for enclosePoints."""
    
    # Generate random points
    np.random.seed(42)  # For reproducibility
    points = np.random.randn(3, 20)
    
    # Test both methods
    Z_maiga = Zonotope.enclosePoints(points, 'maiga')
    Z_stursberg = Zonotope.enclosePoints(points, 'stursberg')
    
    # Both should contain all points
    assert Z_maiga.contains(points), "Maiga method should contain all points"
    assert Z_stursberg.contains(points), "Stursberg method should contain all points"
    
    # Both should have same dimension
    assert Z_maiga.dim() == Z_stursberg.dim() == 3, "Both should have 3 dimensions"


def test_zonotope_enclosePoints_errors():
    """Test error cases for enclosePoints method."""
    
    # Empty point cloud should raise error
    with pytest.raises((ValueError, Exception)):
        Zonotope.enclosePoints(np.array([]))
        
    with pytest.raises((ValueError, Exception)):
        Zonotope.enclosePoints(np.array([]).reshape(0, 0))
    
    # Invalid method should raise error
    points = np.array([[1, 2], [3, 4]])
    with pytest.raises((ValueError, Exception)):
        Zonotope.enclosePoints(points, 'invalid_method')


def test_zonotope_enclosePoints_large_dataset():
    """Test enclosePoints with larger dataset."""
    
    # Generate larger point cloud
    np.random.seed(123)
    n_points = 100
    n_dims = 4
    points = np.random.randn(n_dims, n_points)
    
    # Test with both methods
    Z_maiga = Zonotope.enclosePoints(points, 'maiga')
    Z_stursberg = Zonotope.enclosePoints(points, 'stursberg')
    
    # Check containment
    assert Z_maiga.contains(points), "Maiga should contain all points"
    assert Z_stursberg.contains(points), "Stursberg should contain all points"
    
    # Check dimensions
    assert Z_maiga.dim() == n_dims, f"Should have {n_dims} dimensions"
    assert Z_stursberg.dim() == n_dims, f"Should have {n_dims} dimensions"


if __name__ == "__main__":
    test_zonotope_enclosePoints()
    test_zonotope_enclosePoints_edge_cases()
    test_zonotope_enclosePoints_methods()
    test_zonotope_enclosePoints_errors()
    test_zonotope_enclosePoints_large_dataset()
    print("All tests passed!") 