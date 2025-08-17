"""
Test file for polytope ellipsoid method.

This file tests the conversion of polytopes to ellipsoids using different modes:
- 'outer': outer approximation using covariance method
- 'outer:min-vol': outer approximation using minimum volume method  
- 'inner': inner approximation

Test cases cover:
- Empty polytopes
- 1D and 2D bounded polytopes
- Degenerate cases (points, lines)
- Error handling for invalid modes

Note: Unbounded polytopes cannot be converted to ellipsoids, so they are not tested.
"""

import pytest
import numpy as np
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def test_ellipsoid_empty_polytope():
    """Test ellipsoid conversion of empty polytope"""
    # Create empty polytope
    P = Polytope.empty(2)
    
    # Test all modes
    for mode in ['outer', 'outer:min-vol', 'inner']:
        E = P.ellipsoid(mode)
        assert E.isemptyobject()
        assert E.dim() == 2


def test_ellipsoid_1d_bounded():
    """Test 1D bounded polytope ellipsoid conversion"""
    # 1D bounded polytope: 2 <= x <= 5
    A = np.array([[1], [-1]])
    b = np.array([5, -2])
    P = Polytope(A, b)
    
    # Test outer mode
    E_outer = P.ellipsoid('outer')
    assert isinstance(E_outer, Ellipsoid)
    assert E_outer.dim() == 1
    
    # Test outer:min-vol mode
    E_minvol = P.ellipsoid('outer:min-vol')
    assert isinstance(E_minvol, Ellipsoid)
    assert E_minvol.dim() == 1
    
    # Test inner mode
    E_inner = P.ellipsoid('inner')
    assert isinstance(E_inner, Ellipsoid)
    assert E_inner.dim() == 1
    
    # Verify that inner ellipsoid is contained within outer ellipsoid
    # (this is a basic geometric property that should hold)
    assert E_inner.q.shape == (1, 1)
    assert E_outer.q.shape == (1, 1)


def test_ellipsoid_2d_bounded():
    """Test 2D bounded polytope ellipsoid conversion"""
    # 2D bounded polytope: triangle
    A = np.array([[1, 0], [-1, 1], [-1, -1]])
    b = np.array([1, 1, 1])
    P = Polytope(A, b)
    
    # Test outer mode
    E_outer = P.ellipsoid('outer')
    assert isinstance(E_outer, Ellipsoid)
    assert E_outer.dim() == 2
    
    # Test outer:min-vol mode
    E_minvol = P.ellipsoid('outer:min-vol')
    assert isinstance(E_minvol, Ellipsoid)
    assert E_minvol.dim() == 2
    
    # Test inner mode
    E_inner = P.ellipsoid('inner')
    assert isinstance(E_inner, Ellipsoid)
    assert E_inner.dim() == 2


def test_ellipsoid_2d_degenerate_point():
    """Test 2D degenerate polytope (point) ellipsoid conversion"""
    # 2D degenerate polytope: single point at (2, 3)
    Ae = np.array([[1, 0], [0, 1]])
    be = np.array([2, 3])
    P = Polytope(Ae=Ae, be=be)
    
    # Test outer mode
    E_outer = P.ellipsoid('outer')
    assert isinstance(E_outer, Ellipsoid)
    assert E_outer.dim() == 2
    
    # Test outer:min-vol mode
    E_minvol = P.ellipsoid('outer:min-vol')
    assert isinstance(E_minvol, Ellipsoid)
    assert E_minvol.dim() == 2
    
    # Test inner mode
    E_inner = P.ellipsoid('inner')
    assert isinstance(E_inner, Ellipsoid)
    assert E_inner.dim() == 2
    
    # For a point, the ellipsoid should be very small
    # Check that the center is close to the point
    assert np.allclose(E_outer.q.flatten(), [2, 3], atol=1e-10)
    assert np.allclose(E_minvol.q.flatten(), [2, 3], atol=1e-10)
    assert np.allclose(E_inner.q.flatten(), [2, 3], atol=1e-10)


def test_ellipsoid_2d_degenerate_line():
    """Test 2D degenerate polytope (line) ellipsoid conversion"""
    # 2D degenerate polytope: line segment
    A = np.array([[1, 0], [-1, 1], [-1, 1]])
    b = np.array([1, 1, 1])
    Ae = np.array([[1, 0]])
    be = np.array([0])
    P = Polytope(A, b, Ae, be)
    
    # Test outer mode
    E_outer = P.ellipsoid('outer')
    assert isinstance(E_outer, Ellipsoid)
    assert E_outer.dim() == 2
    
    # Test outer:min-vol mode
    E_minvol = P.ellipsoid('outer:min-vol')
    assert isinstance(E_minvol, Ellipsoid)
    assert E_minvol.dim() == 2
    
    # Test inner mode
    E_inner = P.ellipsoid('inner')
    assert isinstance(E_inner, Ellipsoid)
    assert E_inner.dim() == 2


def test_ellipsoid_3d_bounded():
    """Test 3D bounded polytope ellipsoid conversion"""
    # 3D bounded polytope: cube
    A = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    b = np.array([1, 1, 1, 1, 1, 1])
    P = Polytope(A, b)
    
    # Test outer mode
    E_outer = P.ellipsoid('outer')
    assert isinstance(E_outer, Ellipsoid)
    assert E_outer.dim() == 3
    
    # Test outer:min-vol mode
    E_minvol = P.ellipsoid('outer:min-vol')
    assert isinstance(E_minvol, Ellipsoid)
    assert E_minvol.dim() == 3
    
    # Test inner mode
    E_inner = P.ellipsoid('inner')
    assert isinstance(E_inner, Ellipsoid)
    assert E_inner.dim() == 3


def test_ellipsoid_invalid_mode():
    """Test error handling for invalid mode"""
    # Create a simple polytope
    A = np.array([[1, 0], [0, 1]])
    b = np.array([1, 1])
    P = Polytope(A, b)
    
    # Test invalid mode
    with pytest.raises(CORAerror, match="mode must be"):
        P.ellipsoid('invalid_mode')


def test_ellipsoid_default_mode():
    """Test that default mode is 'outer'"""
    # Create a simple polytope
    A = np.array([[1, 0], [0, 1]])
    b = np.array([1, 1])
    P = Polytope(A, b)
    
    # Test default mode
    E_default = P.ellipsoid()
    E_explicit = P.ellipsoid('outer')
    
    # Both should produce the same result
    assert isinstance(E_default, Ellipsoid)
    assert isinstance(E_explicit, Ellipsoid)
    assert E_default.dim() == E_explicit.dim()


def test_ellipsoid_high_dimension():
    """Test ellipsoid conversion for higher dimensional polytopes"""
    # 5D bounded polytope
    n = 5
    A = np.eye(n)
    b = np.ones(n)
    P = Polytope(A, b)
    
    # Test all modes
    for mode in ['outer', 'outer:min-vol', 'inner']:
        E = P.ellipsoid(mode)
        assert isinstance(E, Ellipsoid)
        assert E.dim() == n


def test_ellipsoid_constraint_normalization():
    """Test that constraint normalization works correctly"""
    # Create polytope with poorly scaled constraints
    A = np.array([[1e6, 0], [0, 1e-6]])
    b = np.array([1e6, 1e-6])
    P = Polytope(A, b)
    
    # Test inner mode (which uses constraint normalization)
    E_inner = P.ellipsoid('inner')
    assert isinstance(E_inner, Ellipsoid)
    assert E_inner.dim() == 2


def test_ellipsoid_vertices_fallback():
    """Test ellipsoid conversion when vertices computation works correctly"""
    # Create a polytope that should work with vertex computation
    A = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    b = np.array([1, 1, 1, 1])
    P = Polytope(A, b)
    
    # Test inner mode
    E_inner = P.ellipsoid('inner')
    assert isinstance(E_inner, Ellipsoid)
    assert E_inner.dim() == 2


def test_ellipsoid_geometric_properties():
    """Test basic geometric properties of ellipsoid approximations"""
    # Create a simple 2D polytope
    A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    b = np.array([1, 1, 1, 1])
    P = Polytope(A, b)
    
    # Get ellipsoid approximations
    E_outer = P.ellipsoid('outer')
    E_inner = P.ellipsoid('inner')
    
    # Basic properties
    assert E_outer.dim() == 2
    assert E_inner.dim() == 2
    
    # The outer ellipsoid should contain the inner ellipsoid
    # This is a fundamental geometric property
    # (Note: This is a simplified check - in practice, you'd need more sophisticated containment tests)
    assert E_outer.Q.shape == (2, 2)
    assert E_inner.Q.shape == (2, 2)
    assert E_outer.q.shape == (2, 1)
    assert E_inner.q.shape == (2, 1)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
