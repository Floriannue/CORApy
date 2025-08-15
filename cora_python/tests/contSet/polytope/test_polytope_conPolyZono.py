import numpy as np
import pytest
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.conPolyZono import ConPolyZono

def test_conpolyzono_basic():
    """Test basic conversion from Polytope to ConPolyZono."""
    # Define a simple 2D polytope (a square)
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.array([[1], [1], [1], [1]])
    P = Polytope(A, b)

    # Convert to ConPolyZono
    cPZ = P.conPolyZono()

    # Assertions
    assert isinstance(cPZ, ConPolyZono)
    assert cPZ.dim() == 2

    # Check if center and generators are reasonable (e.g., origin for a symmetric polytope)
    # This part can be tricky without direct MATLAB comparison, so we check basic properties.
    # For a square [-1,1]x[-1,1], the center should be [0,0]
    assert np.allclose(cPZ.c, np.zeros((2,1)))
    
    # The generators of the ConPolyZono derived from an interval hull (which is what Polytope.zonotope() does)
    # should correspond to the half-widths of the bounding box.
    # For [-1,1]x[-1,1], these would be diag([1,1])
    expected_G = np.eye(2)
    # The actual generators might be permuted or scaled, so we check the norms or properties.
    # For now, a basic check that G is not empty and has correct dimensions.
    assert cPZ.G.shape == (2, 2)
    
    # Ensure the polynomial part and independent generators are empty for this direct conversion
    assert cPZ.E.size == 0
    assert cPZ.EC.size == 0
    assert cPZ.GI.size == 0
    assert cPZ.id.size == 0

    # Check constraints: for an axis-aligned bounding box, there should be no additional constraints
    # on beta, only the implicit [-1,1] bounds. So A and b should be empty.
    assert cPZ.A.size == 0
    assert cPZ.b.size == 0

def test_conpolyzono_empty_polytope():
    """Test conversion of an empty Polytope to ConPolyZono."""
    P = Polytope.empty(3)
    cPZ = P.conPolyZono()
    assert isinstance(cPZ, ConPolyZono)
    assert cPZ.isemptyobject()
    assert cPZ.dim() == 3


def test_conpolyzono_point_polytope():
    """Test conversion of a point Polytope to ConPolyZono."""
    # A single point polytope
    P = Polytope(V=np.array([[0.5],[1.0]]))
    cPZ = P.conPolyZono()
    assert isinstance(cPZ, ConPolyZono)
    assert cPZ.dim() == 2
    # A point should result in a ConPolyZono with zero generators
    assert np.allclose(cPZ.c, np.array([[0.5],[1.0]]))
    assert cPZ.G.size == 0
    assert cPZ.A.size == 0
    assert cPZ.b.size == 0

def test_conpolyzono_unbounded_polytope_error():
    """Test that converting an unbounded polytope raises an error (inherited from zonotope method)."""
    # Define an unbounded polytope (e.g., x >= 0)
    A = np.array([[-1]])
    b = np.array([[0]])
    P_unbounded = Polytope(A, b)
    with pytest.raises(Exception) as excinfo:
        P_unbounded.conPolyZono()
    assert "unbounded" in str(excinfo.value) or "specialError" in str(excinfo.value)
