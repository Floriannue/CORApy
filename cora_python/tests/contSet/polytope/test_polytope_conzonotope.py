"""
test_polytope_conzonotope - unit test function of conversion to constrained zonotopes

This test file tests the conZonotope method of the Polytope class, which converts
polytopes to constrained zonotopes using different methods.

Authors:       Mark Wetzlinger (MATLAB), Florian Nüssel (Python translation)
Written:       29-November-2022 (MATLAB), 2025 (Python translation)
Last update:   14-July-2023 (MW, add empty cases), 27-July-2023 (MW, add more cases)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import pytest
import numpy as np
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.conZonotope.conZonotope import ConZonotope
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def test_conzonotope_empty_polytope():
    """Test conversion of empty polytope"""
    # fully empty polytope
    P = Polytope.empty(2)
    cZ = P.conZonotope()
    assert cZ.isemptyobject() and cZ.dim() == 2


def test_conzonotope_1d_inequalities():
    """Test 1D polytope with only inequalities, bounded"""
    A = np.array([[1], [-1]])
    b = np.array([2, 5])
    P = Polytope(A, b)
    V = P.vertices_()
    cZ = P.conZonotope()
    V_ = cZ.vertices_()
    # Compare vertices with tolerance
    assert np.allclose(V, V_, atol=1e-14)


def test_conzonotope_1d_single_point():
    """Test 1D polytope with single point (equality constraint)"""
    Ae = np.array([[1]])
    be = np.array([4])
    P = Polytope(Ae=Ae, be=be)
    V = P.vertices_()
    cZ = P.conZonotope()
    V_ = cZ.vertices_()
    assert np.allclose(V, V_, atol=1e-14)


def test_conzonotope_2d_inequalities_empty():
    """Test 2D polytope with only inequalities, empty result"""
    A = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    b = np.array([1, 1, 1, -2])
    P = Polytope(A, b)
    # convert to constrained zonotope
    cZ = P.conZonotope()
    assert cZ.representsa_('emptySet', 0)
    cZ = P.conZonotope('exact:vertices')
    assert cZ.representsa_('emptySet', 0)


def test_conzonotope_2d_bounded_vertex_instantiation():
    """Test 2D bounded polytope with vertex instantiation"""
    V = np.array([[3, -2], [3, 2], [1, 3], [1, -1]]).T
    P = Polytope(V)
    # convert to constrained zonotope with default method
    cZ = P.conZonotope()
    V_ = cZ.vertices_()
    assert np.allclose(V, V_, atol=1e-7)
    
    # convert to constrained zonotope with special method
    cZ_vert = P.conZonotope('exact:vertices')
    V_vert = cZ_vert.vertices_()
    assert np.allclose(V, V_vert, atol=1e-14)


def test_conzonotope_2d_inequalities():
    """Test 2D polytope with only inequalities"""
    A = np.array([[2, 1], [-1, 1], [-1, -3], [4, -1]])
    # Normalize rows
    A = (A.T / np.linalg.norm(A, axis=1)).T
    b = np.ones(4)
    P = Polytope(A, b)
    # compute vertices
    V = P.vertices_()
    # convert to constrained zonotope with default method
    cZ = P.conZonotope()
    V_ = cZ.vertices_()
    assert np.allclose(V, V_, atol=1e-7)
    # convert to constrained zonotope with special method
    cZ_vert = P.conZonotope('exact:vertices')
    V_vert = cZ_vert.vertices_()
    assert np.allclose(V, V_vert, atol=1e-7)


def test_conzonotope_2d_degenerate_mixed():
    """Test 2D degenerate polytope with inequalities and equalities"""
    A = np.array([[0, 1], [0, -1]])
    b = np.array([3, -1])
    Ae = np.array([[1, 1]])
    be = np.array([1])
    P = Polytope(A, b, Ae, be)
    V = P.vertices_()
    # convert to constrained zonotope with default method
    cZ = P.conZonotope()
    V_ = cZ.vertices_()
    assert np.allclose(V, V_, atol=1e-14)
    # convert to constrained zonotope with special method
    cZ_vert = P.conZonotope('exact:vertices')
    V_vert = cZ_vert.vertices_()
    assert np.allclose(V, V_vert, atol=1e-14)


def test_conzonotope_2d_fullspace_error():
    """Test error for 2D fullspace (unbounded) polytope"""
    A = np.zeros((0, 2))
    b = np.zeros((0, 0))
    P = Polytope(A, b)
    with pytest.raises(CORAerror, match="Polytope is unbounded"):
        P.conZonotope()


def test_conzonotope_2d_unbounded_error():
    """Test error for 2D unbounded polytope"""
    A = np.array([[1, 0]])
    b = np.array([1])
    P = Polytope(A, b)
    with pytest.raises(CORAerror, match="Polytope is unbounded"):
        P.conZonotope()
    with pytest.raises(CORAerror, match="Polytope is unbounded"):
        P.conZonotope('exact:vertices')


def test_conzonotope_method_validation():
    """Test method parameter validation"""
    A = np.array([[1, 0], [0, 1]])
    b = np.array([1, 1])
    P = Polytope(A, b)
    
    # Valid methods should work
    cZ1 = P.conZonotope('exact:supportFunc')
    cZ2 = P.conZonotope('exact:vertices')
    assert isinstance(cZ1, ConZonotope)
    assert isinstance(cZ2, ConZonotope)
    
    # Invalid method should raise error
    with pytest.raises(CORAerror, match="method must be exact:vertices or exact:supportFunc"):
        P.conZonotope('invalid_method')


def test_conzonotope_with_box_parameter():
    """Test conZonotope with box parameter"""
    A = np.array([[1, 0], [0, 1]])
    b = np.array([1, 1])
    P = Polytope(A, b)
    
    # Create custom box
    box = Interval([-2, -2], [2, 2])
    cZ = P.conZonotope('exact:supportFunc', box)
    assert isinstance(cZ, ConZonotope)


def test_conzonotope_vertex_representation_empty():
    """Test conZonotope with empty vertex representation"""
    # Create polytope with empty vertex representation
    V = np.zeros((2, 0))  # 2D, 0 vertices
    P = Polytope(V)
    P.isVRep = True
    P.V = V
    
    cZ = P.conZonotope()
    assert cZ.isemptyobject() and cZ.dim() == 2


def test_conzonotope_known_empty():
    """Test conZonotope with known empty polytope"""
    # Create polytope known to be empty
    A = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    b = np.array([1, 1, 1, -2])
    P = Polytope(A, b)
    
    # Set the emptySet property
    P._emptySet_val = True
    
    cZ = P.conZonotope()
    assert cZ.isemptyobject() and cZ.dim() == 2


def test_conzonotope_known_unbounded():
    """Test conZonotope with known unbounded polytope"""
    # Create polytope known to be unbounded
    A = np.array([[1, 0]])
    b = np.array([1])
    P = Polytope(A, b)
    
    # Set the bounded property
    P._bounded_val = False
    
    with pytest.raises(CORAerror, match="Polytope is unbounded"):
        P.conZonotope()


def test_conzonotope_vertices_method_unbounded():
    """Test conZonotope vertices method with unbounded polytope"""
    # Create polytope that will fail vertex computation
    A = np.array([[1, 0]])
    b = np.array([1])
    P = Polytope(A, b)
    P._bounded_val = False
    
    with pytest.raises(CORAerror, match="Polytope is unbounded"):
        P.conZonotope('exact:vertices')


def test_conzonotope_input_validation():
    """Test input validation for conZonotope method"""
    # Test with non-polytope input
    with pytest.raises(CORAerror, match="First argument must be a polytope"):
        from cora_python.contSet.polytope import conZonotope
        conZonotope("not_a_polytope")


def test_conzonotope_random_polytopes():
    """Test conZonotope with random polytopes (similar to long test)"""
    np.random.seed(42)  # For reproducible tests
    
    for i in range(3):  # Reduced number for faster testing
        # random dimension
        n = np.random.randint(1, 4)  # Reduced max dimension
        
        # instantiation: generateRandom
        P = Polytope.generate_random('Dimension', n, 'NrConstraints', 2*n)
        # compute vertices
        V_P = P.vertices_()
        
        # convert to constrained zonotope
        cZ_sF = P.conZonotope('exact:supportFunc')
        cZ_vert = P.conZonotope('exact:vertices')
        
        # compute vertices
        V_sF = cZ_sF.vertices_()
        V_vert = cZ_vert.vertices_()
        
        # compare vertices with tolerance
        assert np.allclose(V_P, V_sF, atol=1e-12), f"Test {i}: supportFunc method failed"
        assert np.allclose(V_P, V_vert, atol=1e-12), f"Test {i}: vertices method failed"
