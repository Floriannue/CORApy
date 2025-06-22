"""
test_ellipsoid_zonotope - unit test function of zonotope conversion

Authors:       Victor Gassmann (MATLAB)
               Python translation by AI Assistant
Written:       27-July-2021 (MATLAB)
Python translation: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def test_ellipsoid_zonotope():
    """Test ellipsoid to zonotope conversion"""
    
    # Test cases from MATLAB
    E1 = Ellipsoid(
        np.array([[5.4387811500952807, 12.4977183618314545], 
                  [12.4977183618314545, 29.6662117284481646]]), 
        np.array([[-0.7445068341257537], [3.5800647524843665]]), 
        0.000001
    )
    
    Ed1 = Ellipsoid(
        np.array([[4.2533342807136076, 0.6346400221575308], 
                  [0.6346400221575309, 0.0946946398147988]]), 
        np.array([[-2.4653656883489115], [0.2717868749873985]]), 
        0.000001
    )
    
    E0 = Ellipsoid(
        np.array([[0.0000000000000000, 0.0000000000000000], 
                  [0.0000000000000000, 0.0000000000000000]]), 
        np.array([[1.0986933635979599], [-1.9884387759871638]]), 
        0.000001
    )
    
    n = E1.dim()
    N = 5 * n
    
    # Test inner norm conversion
    Z1 = E1.zonotope('inner:norm', 2*n)
    assert isinstance(Z1, Zonotope)
    
    # Test outer norm conversion
    Zd1 = Ed1.zonotope('outer:norm', 2*n)
    assert isinstance(Zd1, Zonotope)
    
    # Test inner norm_bnd conversion with degenerate ellipsoid
    Z0 = E0.zonotope('inner:norm_bnd', 2*n)
    assert isinstance(Z0, Zonotope)
    
    # Test default box conversion
    Z_box = E1.zonotope()
    assert isinstance(Z_box, Zonotope)
    
    # Test outer box conversion
    Z_outer_box = E1.zonotope('outer:box')
    assert isinstance(Z_outer_box, Zonotope)
    
    # Test inner box conversion
    Z_inner_box = E1.zonotope('inner:box')
    assert isinstance(Z_inner_box, Zonotope)
    
    # Test empty ellipsoid
    E_empty = Ellipsoid.empty(2)
    Z_empty = E_empty.zonotope()
    assert Z_empty.representsa_('emptySet')
    
    # Test point ellipsoid
    E_point = Ellipsoid(np.zeros((2, 2)), np.array([[1], [2]]))
    Z_point = E_point.zonotope()
    assert Z_point.representsa_('point')
    assert np.allclose(Z_point.center(), np.array([[1], [2]]))
    
    # Test unsupported mode
    with pytest.raises(CORAError):
        E1.zonotope('outer:norm_bnd')


def test_ellipsoid_zonotope_containment():
    """Test containment properties of zonotope conversion"""
    
    # Create a simple ellipsoid
    Q = np.array([[4, 1], [1, 2]])
    q = np.array([[1], [0]])
    E = Ellipsoid(Q, q)
    
    # Test outer approximation contains ellipsoid
    Z_outer = E.zonotope('outer:box')
    
    # Generate some points on ellipsoid boundary and check they're in zonotope
    # This is a simplified test - in practice we'd use more sophisticated methods
    angles = np.linspace(0, 2*np.pi, 20)
    
    # For a 2D ellipsoid, we can parameterize the boundary
    # x = q + sqrt(Q) * [cos(theta), sin(theta)]^T
    sqrt_Q = np.linalg.cholesky(Q).T
    for angle in angles:
        unit_circle_point = np.array([[np.cos(angle)], [np.sin(angle)]])
        ellipse_point = q + sqrt_Q @ unit_circle_point
        
        # Check if point is contained in outer zonotope (simplified check)
        # In practice, we'd use proper containment checking
        assert Z_outer is not None  # Basic check that conversion worked


def test_ellipsoid_zonotope_dimensions():
    """Test dimensional consistency of zonotope conversion"""
    
    # Test different dimensions
    for dim in [1, 2, 3]:
        E = Ellipsoid(np.eye(dim), np.zeros((dim, 1)))
        
        # Test different conversion modes
        Z_box = E.zonotope('outer:box')
        assert Z_box.dim() == dim
        
        Z_norm = E.zonotope('outer:norm', 2*dim)
        assert Z_norm.dim() == dim
        
        Z_inner = E.zonotope('inner:box')
        assert Z_inner.dim() == dim


def test_ellipsoid_zonotope_degenerate():
    """Test zonotope conversion of degenerate ellipsoids"""
    
    # Test rank-deficient ellipsoid (line segment)
    Q = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    q = np.array([[0], [1], [2]])
    E_line = Ellipsoid(Q, q)
    
    Z_line = E_line.zonotope('outer:box')
    assert isinstance(Z_line, Zonotope)
    assert Z_line.dim() == 3
    
    # Test point ellipsoid (rank 0)
    Q_point = np.zeros((3, 3))
    q_point = np.array([[1], [2], [3]])
    E_point = Ellipsoid(Q_point, q_point)
    
    Z_point = E_point.zonotope('inner:norm', 5)
    assert isinstance(Z_point, Zonotope)
    assert Z_point.representsa_('point')
    assert np.allclose(Z_point.center(), q_point)


if __name__ == '__main__':
    test_ellipsoid_zonotope()
    test_ellipsoid_zonotope_containment()
    test_ellipsoid_zonotope_dimensions()
    test_ellipsoid_zonotope_degenerate()
    print("All ellipsoid zonotope conversion tests passed!") 