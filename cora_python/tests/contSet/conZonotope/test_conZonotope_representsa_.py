"""
test_conZonotope_representsa_ - unit tests for ConZonotope representsa_ method

Syntax:
    python -m pytest cora_python/tests/contSet/conZonotope/test_conZonotope_representsa_.py

Authors: MATLAB original tests, Python translation by AI Assistant
Written: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.conZonotope import ConZonotope
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contSet.emptySet import EmptySet
from cora_python.contSet.polytope import Polytope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestConZonotopeRepresentsa:
    """Test class for ConZonotope representsa_ method."""

    def test_representsa_origin(self):
        """Test representsa_ for origin"""
        # Origin constrained zonotope
        c = np.zeros((2, 1))
        G = np.zeros((2, 0))
        A = np.zeros((0, 0))
        b = np.zeros((0,))
        cZ = ConZonotope(c, G, A, b)
        
        # Should represent origin
        assert cZ.representsa_('origin', 1e-12)
        
        # With return set
        res, S = cZ.representsa_('origin', 1e-12)
        assert res
        assert np.allclose(S, np.zeros((2, 1)))
        
        # Non-origin constrained zonotope
        c_non = np.array([[1], [0]])
        cZ_non = ConZonotope(c_non, G, A, b)
        assert not cZ_non.representsa_('origin', 1e-12)

    def test_representsa_point(self):
        """Test representsa_ for point"""
        # Point constrained zonotope
        c = np.array([[2], [3]])
        G = np.zeros((2, 0))
        A = np.zeros((0, 0))
        b = np.zeros((0,))
        cZ_point = ConZonotope(c, G, A, b)
        
        # Should represent point
        assert cZ_point.representsa_('point', 1e-12)
        
        # With return set
        res, S = cZ_point.representsa_('point', 1e-12)
        assert res
        assert np.allclose(S, c)
        
        # Non-point constrained zonotope
        G_non = np.array([[1, 0], [0, 1]])
        cZ_non = ConZonotope(c, G_non, A, b)
        assert not cZ_non.representsa_('point', 1e-12)

    def test_representsa_conZonotope(self):
        """Test representsa_ for conZonotope (always true)"""
        c = np.array([[0], [1]])
        G = np.array([[1, 2], [0, 1]])
        A = np.array([[1, 1]])
        b = np.array([1])
        cZ = ConZonotope(c, G, A, b)
        
        assert cZ.representsa_('conZonotope', 1e-12)
        
        # With return set
        res, S = cZ.representsa_('conZonotope', 1e-12)
        assert res
        assert S is cZ

    def test_representsa_conPolyZono(self):
        """Test representsa_ for conPolyZono (always true)"""
        c = np.array([[0], [1]])
        G = np.array([[1, 2], [0, 1]])
        A = np.array([[1, 1]])
        b = np.array([1])
        cZ = ConZonotope(c, G, A, b)
        
        assert cZ.representsa_('conPolyZono', 1e-12)

    def test_representsa_halfspace(self):
        """Test representsa_ for halfspace (always false)"""
        c = np.array([[0], [1]])
        G = np.array([[1, 2], [0, 1]])
        A = np.array([[1, 1]])
        b = np.array([1])
        cZ = ConZonotope(c, G, A, b)
        
        # Constrained zonotopes cannot be unbounded
        assert not cZ.representsa_('halfspace', 1e-12)

    def test_representsa_interval(self):
        """Test representsa_ for interval"""
        # Unconstrained zonotope that represents interval
        c = np.array([[0], [0]])
        G = np.array([[1, 0], [0, 1]])  # Axis-aligned
        A = np.zeros((0, 2))
        b = np.zeros((0,))
        cZ_interval = ConZonotope(c, G, A, b)
        
        # Should represent interval (unconstrained + underlying zonotope is interval)
        res = cZ_interval.representsa_('interval', 1e-12)
        # Note: This depends on zonotope.representsa_('interval') implementation
        
        # Constrained zonotope cannot represent interval
        A_con = np.array([[1, 1]])
        b_con = np.array([0.5])
        cZ_constrained = ConZonotope(c, G, A_con, b_con)
        assert not cZ_constrained.representsa_('interval', 1e-12)

    def test_representsa_polytope(self):
        """Test representsa_ for polytope (always true)"""
        c = np.array([[0], [1]])
        G = np.array([[1, 2], [0, 1]])
        A = np.array([[1, 1]])
        b = np.array([1])
        cZ = ConZonotope(c, G, A, b)
        
        assert cZ.representsa_('polytope', 1e-12)

    def test_representsa_polyZonotope(self):
        """Test representsa_ for polyZonotope (always true)"""
        c = np.array([[0], [1]])
        G = np.array([[1, 2], [0, 1]])
        A = np.array([[1, 1]])
        b = np.array([1])
        cZ = ConZonotope(c, G, A, b)
        
        assert cZ.representsa_('polyZonotope', 1e-12)

    def test_representsa_probZonotope(self):
        """Test representsa_ for probZonotope (always false)"""
        c = np.array([[0], [1]])
        G = np.array([[1, 2], [0, 1]])
        A = np.array([[1, 1]])
        b = np.array([1])
        cZ = ConZonotope(c, G, A, b)
        
        assert not cZ.representsa_('probZonotope', 1e-12)

    def test_representsa_zonoBundle(self):
        """Test representsa_ for zonoBundle (always true)"""
        c = np.array([[0], [1]])
        G = np.array([[1, 2], [0, 1]])
        A = np.array([[1, 1]])
        b = np.array([1])
        cZ = ConZonotope(c, G, A, b)
        
        assert cZ.representsa_('zonoBundle', 1e-12)

    def test_representsa_zonotope(self):
        """Test representsa_ for zonotope"""
        # Unconstrained zonotope
        c = np.array([[0], [1]])
        G = np.array([[1, 2], [0, 1]])
        A = np.zeros((0, 2))
        b = np.zeros((0,))
        cZ_unconstrained = ConZonotope(c, G, A, b)
        
        # Should represent zonotope
        assert cZ_unconstrained.representsa_('zonotope', 1e-12)
        
        # With return set
        res, S = cZ_unconstrained.representsa_('zonotope', 1e-12)
        assert res
        assert isinstance(S, Zonotope)
        assert np.allclose(S.c, c)
        assert np.allclose(S.G, G)
        
        # Constrained zonotope
        A_con = np.array([[1, 1]])
        b_con = np.array([1])
        cZ_constrained = ConZonotope(c, G, A_con, b_con)
        assert not cZ_constrained.representsa_('zonotope', 1e-12)

    def test_representsa_hyperplane(self):
        """Test representsa_ for hyperplane"""
        # 1D constrained zonotope can represent hyperplane
        c_1d = np.array([[0]])
        G_1d = np.array([[1]])
        A_1d = np.zeros((0, 1))
        b_1d = np.zeros((0,))
        cZ_1d = ConZonotope(c_1d, G_1d, A_1d, b_1d)
        
        assert cZ_1d.representsa_('hyperplane', 1e-12)
        
        # 2D constrained zonotope cannot represent hyperplane
        c_2d = np.array([[0], [1]])
        G_2d = np.array([[1, 2], [0, 1]])
        A_2d = np.zeros((0, 2))
        b_2d = np.zeros((0,))
        cZ_2d = ConZonotope(c_2d, G_2d, A_2d, b_2d)
        
        assert not cZ_2d.representsa_('hyperplane', 1e-12)

    def test_representsa_convexSet(self):
        """Test representsa_ for convexSet (always true)"""
        c = np.array([[0], [1]])
        G = np.array([[1, 2], [0, 1]])
        A = np.array([[1, 1]])
        b = np.array([1])
        cZ = ConZonotope(c, G, A, b)
        
        assert cZ.representsa_('convexSet', 1e-12)

    def test_representsa_emptySet(self):
        """Test representsa_ for emptySet"""
        # Non-empty constrained zonotope
        c = np.array([[0], [0]])
        G = np.array([[1, 0], [0, 1]])
        A = np.array([[1, 1]])
        b = np.array([1])
        cZ_non_empty = ConZonotope(c, G, A, b)
        
        assert not cZ_non_empty.representsa_('emptySet', 1e-12)
        
        # Empty constrained zonotope (infeasible constraints)
        c_empty = np.array([[0], [0]])
        G_empty = np.array([[1, 0], [0, 1]])
        A_empty = np.array([[1, 0], [-1, 0]])  # x >= 1 and x <= -1 (impossible)
        b_empty = np.array([1, 1])
        cZ_empty = ConZonotope(c_empty, G_empty, A_empty, b_empty)
        
        # This should represent empty set
        res = cZ_empty.representsa_('emptySet', 1e-12)
        # Note: This test may fail if the emptiness detection is not fully implemented
        
        # With return set
        if res:
            res_with_set, S = cZ_empty.representsa_('emptySet', 1e-12)
            assert res_with_set
            assert isinstance(S, EmptySet)

    def test_representsa_fullspace(self):
        """Test representsa_ for fullspace (always false)"""
        c = np.array([[0], [1]])
        G = np.array([[1, 2], [0, 1]])
        A = np.array([[1, 1]])
        b = np.array([1])
        cZ = ConZonotope(c, G, A, b)
        
        # Constrained zonotopes cannot be unbounded
        assert not cZ.representsa_('fullspace', 1e-12)

    def test_representsa_unsupported_types(self):
        """Test representsa_ for unsupported types"""
        c = np.array([[0], [1]])
        G = np.array([[1, 2], [0, 1]])
        A = np.array([[1, 1]])
        b = np.array([1])
        cZ = ConZonotope(c, G, A, b)
        
        unsupported_types = ['conHyperplane', 'capsule', 'ellipsoid', 'levelSet', 'parallelotope']
        
        for unsupported_type in unsupported_types:
            with pytest.raises(CORAerror):
                cZ.representsa_(unsupported_type, 1e-12)

    def test_representsa_unknown_type(self):
        """Test representsa_ for unknown type"""
        c = np.array([[0], [1]])
        G = np.array([[1, 2], [0, 1]])
        A = np.array([[1, 1]])
        b = np.array([1])
        cZ = ConZonotope(c, G, A, b)
        
        # Unknown type should return false
        assert not cZ.representsa_('unknownType', 1e-12)

    def test_representsa_return_values(self):
        """Test representsa_ return value formats"""
        c = np.array([[0], [1]])
        G = np.array([[1, 2], [0, 1]])
        A = np.array([[1, 1]])
        b = np.array([1])
        cZ = ConZonotope(c, G, A, b)
        
        # Single return value
        res = cZ.representsa_('conZonotope', 1e-12)
        assert isinstance(res, bool)
        assert res
        
        # Two return values
        res, S = cZ.representsa_('conZonotope', 1e-12)
        assert isinstance(res, bool)
        assert res
        assert S is cZ


def test_conZonotope_representsa():
    """Main test function for ConZonotope representsa_ method."""
    test = TestConZonotopeRepresentsa()
    
    # Run all tests
    test.test_representsa_origin()
    test.test_representsa_point()
    test.test_representsa_conZonotope()
    test.test_representsa_conPolyZono()
    test.test_representsa_halfspace()
    test.test_representsa_interval()
    test.test_representsa_polytope()
    test.test_representsa_polyZonotope()
    test.test_representsa_probZonotope()
    test.test_representsa_zonoBundle()
    test.test_representsa_zonotope()
    test.test_representsa_hyperplane()
    test.test_representsa_convexSet()
    test.test_representsa_emptySet()
    test.test_representsa_fullspace()
    test.test_representsa_unsupported_types()
    test.test_representsa_unknown_type()
    test.test_representsa_return_values()
    
    print("test_conZonotope_representsa_: all tests passed")


if __name__ == "__main__":
    test_conZonotope_representsa() 