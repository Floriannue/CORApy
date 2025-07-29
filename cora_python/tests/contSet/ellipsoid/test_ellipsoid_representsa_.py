"""
test_ellipsoid_representsa_ - unit test function of representsa_

This module tests the ellipsoid representsa_ implementation.

Authors:       Python translation by AI Assistant
Python translation: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.representsa_ import representsa_
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.contSet.interval import Interval
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.emptySet import EmptySet
from cora_python.contSet.fullspace import Fullspace


class TestEllipsoidRepresentsa:

    def create_ellipsoid(self, Q, q=None, TOL=1e-6):
        if q is None:
            q = np.zeros((Q.shape[0], 1))
        return Ellipsoid(Q, q, TOL)

    # Test cases for an empty ellipsoid, handled by representsa_emptyObject
    def test_representsa_empty_ellipsoid_empty_set(self):
        E_empty = self.create_ellipsoid(np.array([[]]).reshape(0,0), np.array([[]]).reshape(0,1), 1e-6)
        res, S_conv = representsa_(E_empty, 'emptySet', 1e-6, return_set=True)
        assert res is True
        assert isinstance(S_conv, EmptySet)
        assert S_conv.dim() == 0 # Assuming 0-dim empty set for empty ellipsoid

    def test_representsa_empty_ellipsoid_fullspace(self):
        # An empty ellipsoid should NOT represent a fullspace typically
        E_empty = self.create_ellipsoid(np.array([[]]).reshape(0,0), np.array([[]]).reshape(0,1), 1e-6)
        res, S_conv = representsa_(E_empty, 'fullspace', 1e-6, return_set=True)
        assert res is False
        assert S_conv is None

    # Test cases for a point ellipsoid
    def test_representsa_point_ellipsoid_origin(self):
        E_point = self.create_ellipsoid(np.zeros((2,2)), np.zeros((2,1)), 1e-6)
        res, S_conv = representsa_(E_point, 'origin', 1e-6, return_set=True)
        assert res == True
        assert np.allclose(S_conv, np.zeros((2,1)))

    def test_representsa_point_ellipsoid_point(self):
        E_point = self.create_ellipsoid(np.zeros((2,2)), np.array([[1],[2]]), 1e-6)
        res, S_conv = representsa_(E_point, 'point', 1e-6, return_set=True)
        assert res == True
        assert np.allclose(S_conv, np.array([[1],[2]]))

    def test_representsa_point_ellipsoid_interval_1d(self):
        E_point_1d = self.create_ellipsoid(np.zeros((1,1)), np.array([[5]]), 1e-6)
        res, S_conv = representsa_(E_point_1d, 'interval', 1e-6, return_set=True)
        assert res is True
        assert isinstance(S_conv, Interval)
        assert np.allclose(S_conv.inf, np.array([5]))
        assert np.allclose(S_conv.sup, np.array([5]))

    def test_representsa_point_ellipsoid_zonotope_1d(self):
        E_point_1d = self.create_ellipsoid(np.zeros((1,1)), np.array([[5]]), 1e-6)
        res, S_conv = representsa_(E_point_1d, 'zonotope', 1e-6, return_set=True)
        assert res is True
        assert isinstance(S_conv, Zonotope)
        assert np.allclose(S_conv.c, np.array([5]))
        assert S_conv.G.size == 0 # No generators for a point zonotope

    # Test cases for 1D ellipsoid as interval/zonotope
    def test_representsa_1d_ellipsoid_interval(self):
        Q = np.array([[4.0]]) # Corresponds to radius 2
        q = np.array([[1.0]])
        E_1d = self.create_ellipsoid(Q, q, 1e-6)
        res, S_conv = representsa_(E_1d, 'interval', 1e-6, return_set=True)
        assert res is True
        assert isinstance(S_conv, Interval)
        assert np.allclose(S_conv.inf, np.array([-1.0]))
        assert np.allclose(S_conv.sup, np.array([3.0]))

    def test_representsa_1d_ellipsoid_zonotope(self):
        Q = np.array([[4.0]]) # Corresponds to radius 2
        q = np.array([[1.0]])
        E_1d = self.create_ellipsoid(Q, q, 1e-6)
        res, S_conv = representsa_(E_1d, 'zonotope', 1e-6, return_set=True)
        assert res is True
        assert isinstance(S_conv, Zonotope)
        assert np.allclose(S_conv.c, np.array([1.0]))
        assert np.allclose(S_conv.G, np.array([[2.0]]))

    # Test cases for unsupported conversions that *should* raise CORAerror when S is requested
    # These match MATLAB's behavior of throwing CORA:notSupported if nargout == 2 and res is true
    def test_representsa_unsupported_conversion_capsule_throws_error(self):
        E = self.create_ellipsoid(np.eye(2) * 2) # Makes res=True possible
        with pytest.raises(CORAerror) as excinfo:
            representsa_(E, 'capsule', 1e-6, return_set=True)
        assert excinfo.value.identifier == 'CORA:notSupported'

    def test_representsa_unsupported_conversion_conHyperplane_throws_error(self):
        E = self.create_ellipsoid(np.eye(1) * 2) # Makes res=True possible (1D)
        with pytest.raises(CORAerror) as excinfo:
            representsa_(E, 'conHyperplane', 1e-6, return_set=True)
        assert excinfo.value.identifier == 'CORA:notSupported'

    def test_representsa_unsupported_conversion_polytope_throws_error(self):
        E = self.create_ellipsoid(np.eye(1) * 2) # Makes res=True possible (1D)
        with pytest.raises(CORAerror) as excinfo:
            representsa_(E, 'polytope', 1e-6, return_set=True)
        assert excinfo.value.identifier == 'CORA:notSupported'

    def test_representsa_unsupported_conversion_zonoBundle_throws_error(self):
        E = self.create_ellipsoid(np.eye(1) * 2) # Makes res=True possible (1D)
        with pytest.raises(CORAerror) as excinfo:
            representsa_(E, 'zonoBundle', 1e-6, return_set=True)
        assert excinfo.value.identifier == 'CORA:notSupported'

    # Test cases for comparisons that always throw CORAerror if S is requested
    def test_representsa_unsupported_comparison_conPolyZono_throws_error(self):
        E = self.create_ellipsoid(np.eye(2) * 2)
        with pytest.raises(CORAerror) as excinfo:
            representsa_(E, 'conPolyZono', 1e-6, return_set=True)
        assert excinfo.value.identifier == 'CORA:notSupported'

    def test_representsa_unsupported_comparison_levelSet_throws_error(self):
        E = self.create_ellipsoid(np.eye(2) * 2)
        with pytest.raises(CORAerror) as excinfo:
            representsa_(E, 'levelSet', 1e-6, return_set=True)
        assert excinfo.value.identifier == 'CORA:notSupported'

    def test_representsa_unsupported_comparison_polyZonotope_throws_error(self):
        E = self.create_ellipsoid(np.eye(2) * 2)
        with pytest.raises(CORAerror) as excinfo:
            representsa_(E, 'polyZonotope', 1e-6, return_set=True)
        assert excinfo.value.identifier == 'CORA:notSupported'

    def test_representsa_unsupported_comparison_parallelotope_throws_error(self):
        E = self.create_ellipsoid(np.eye(2) * 2)
        with pytest.raises(CORAerror) as excinfo:
            representsa_(E, 'parallelotope', 1e-6, return_set=True)
        assert excinfo.value.identifier == 'CORA:notSupported'

    # Test cases for general ellipsoid properties (no conversion requested or returns False)
    def test_representsa_ellipsoid_itself(self):
        E = self.create_ellipsoid(np.eye(2))
        res, S_conv = representsa_(E, 'ellipsoid', 1e-6, return_set=True)
        assert res is True
        assert S_conv is E # Should return the same object

    def test_representsa_ellipsoid_convex_set(self):
        E = self.create_ellipsoid(np.eye(2))
        res, S_conv = representsa_(E, 'convexSet', 1e-6, return_set=True)
        assert res is True
        assert S_conv is None # convexSet doesn't have a direct conversion object here

    def test_representsa_ellipsoid_fullspace_false(self):
        E = self.create_ellipsoid(np.eye(2))
        res, S_conv = representsa_(E, 'fullspace', 1e-6, return_set=True)
        assert res is False
        assert S_conv is None

    def test_representsa_ellipsoid_halfspace_false(self):
        E = self.create_ellipsoid(np.eye(2))
        res, S_conv = representsa_(E, 'halfspace', 1e-6, return_set=True)
        assert res is False
        assert S_conv is None

    def test_representsa_ellipsoid_probZonotope_false(self):
        E = self.create_ellipsoid(np.eye(2))
        res, S_conv = representsa_(E, 'probZonotope', 1e-6, return_set=True)
        assert res is False
        assert S_conv is None

    # Test cases for invalid type_str (returns False in MATLAB)
    def test_representsa_invalid_type_string(self):
        E = self.create_ellipsoid(np.eye(2))
        res, S_conv = representsa_(E, 'invalid_type', 1e-6, return_set=True)
        assert res is False
        assert S_conv is None 