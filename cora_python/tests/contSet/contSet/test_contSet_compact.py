"""
test_compact - unit test function for compact

Syntax:
    pytest test_compact.py

Inputs:
    -

Outputs:
    test results

Other modules required: none
Subfunctions: none

See also: none

Authors: AI Assistant
Written: 2025
Last update: ---
Last revision: ---
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from cora_python.contSet.contSet.contSet import ContSet
from cora_python.contSet.contSet.compact import compact
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def create_mock_compact_class(name, compact_side_effect=None):
    class MockCompactSet(ContSet):
        def __init__(self, dim=2, is_empty=False):
            super().__init__()
            self._dim = dim
            self._is_empty = is_empty
            # Mock the compact_ method
            self.compact_ = MagicMock(name=f"{name}.compact_", side_effect=compact_side_effect)

        def __repr__(self):
            return f"{self.__class__.__name__}(dim={self._dim})"

        def dim(self):
            return self._dim

        def isemptyobject(self):
            return self._is_empty
        
        def representsa_(self, setType, tol):
            return self._is_empty and setType == 'emptySet'

    MockCompactSet.__name__ = name
    return MockCompactSet


class TestCompact:
    """Test class for compact function"""

    def test_compact_minimal_classes(self):
        """Test compact with classes that are always minimal"""
        minimal_classes = ['Capsule', 'Ellipsoid', 'EmptySet', 'Fullspace', 
                           'Halfspace', 'Interval', 'ZonoBundle', 'SpectraShadow', 'Taylm']
        
        for class_name in minimal_classes:
            MockClass = create_mock_compact_class(class_name)
            S = MockClass()
            result = compact(S)
            assert result is S
            S.compact_.assert_not_called()

    def test_compact_zonotope_defaults(self):
        """Test compact with Zonotope using default parameters"""
        MockZonotope = create_mock_compact_class("Zonotope")
        S = MockZonotope()
        S.compact_.return_value = "Compacted"
        result = compact(S)
        S.compact_.assert_called_once_with('zeros', np.finfo(float).eps)
        assert result == "Compacted"

    def test_compact_zonotope_methods(self):
        """Test compact with Zonotope using different methods"""
        MockZonotope = create_mock_compact_class("Zonotope")
        S = MockZonotope()
        
        valid_methods = ['all', 'zeros', 'aligned']
        for method in valid_methods:
            S.compact_.reset_mock()
            compact(S, method=method)
            S.compact_.assert_called_once()
        
        with pytest.raises(ValueError, match="Invalid method"):
            compact(S, method='invalid')

    def test_compact_polytope_methods(self):
        """Test compact with Polytope using different methods"""
        MockPolytope = create_mock_compact_class("Polytope")
        S = MockPolytope()
        
        valid_methods = ['all', 'zeros', 'A', 'Ae', 'aligned', 'V', 'AtoAe']
        for method in valid_methods:
            S.compact_.reset_mock()
            compact(S, method=method)
            S.compact_.assert_called_once()
        
        with pytest.raises(ValueError, match="Invalid method"):
            compact(S, method='invalid')

    def test_compact_tolerance_validation(self):
        """Test compact with tolerance validation"""
        MockZonotope = create_mock_compact_class("Zonotope")
        S = MockZonotope()
        
        compact(S, tol=1e-6)
        S.compact_.assert_called_with('zeros', 1e-6)
        
        with pytest.raises(ValueError, match="tol must be a non-negative number"):
            compact(S, tol=-1)
        
        with pytest.raises(ValueError, match="tol must be a non-negative number"):
            compact(S, tol="invalid")

    def test_compact_not_implemented_error(self):
        """Test compact when method is not implemented"""
        side_effect = CORAerror('CORA:noops', 'compact not implemented')
        side_effect.identifier = ''
        MockCustomSet = create_mock_compact_class("CustomSet", compact_side_effect=side_effect)
        S = MockCustomSet()
        with pytest.raises(CORAerror, match="compact not implemented for CustomSet"):
            compact(S)

    def test_compact_other_exception(self):
        """Test compact when other exception occurs"""
        side_effect = RuntimeError("Some other error")
        MockErrorSet = create_mock_compact_class("ErrorSet", compact_side_effect=side_effect)
        S = MockErrorSet()
        with pytest.raises(RuntimeError, match="Some other error"):
            compact(S)


if __name__ == "__main__":
    pytest.main([__file__]) 