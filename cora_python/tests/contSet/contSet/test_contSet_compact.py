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
from unittest.mock import Mock
import builtins
from cora_python.contSet.contSet.compact import compact
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


# Store the original type function
original_type = builtins.type


class MockContSet:
    """Mock ContSet for testing compact method"""
    
    def __init__(self, class_name="Zonotope", dim_val=2, empty=False):
        self._class_name = class_name
        self._dim = dim_val
        self._empty = empty
        
    def __class__(self):
        class MockClass:
            __name__ = self._class_name
        return MockClass()
    
    def __class_getitem__(cls, item):
        return cls
    
    @property 
    def __name__(self):
        return self._class_name
    
    def dim(self):
        return self._dim
    
    def isemptyobject(self):
        return self._empty
    
    def compact_(self, method, tol):
        """Mock implementation of compact_"""
        if self._class_name in ['Capsule', 'Ellipsoid', 'EmptySet']:
            raise Exception("Should not call compact_ for minimal classes")
        return MockContSet(self._class_name, self._dim, self._empty)
    
    def representsa_(self, setType, tol):
        return self._empty and setType == 'emptySet'


def mock_type(obj):
    if isinstance(obj, MockContSet):
        class MockType:
            __name__ = obj._class_name
        return MockType()
    return original_type(obj)


class TestCompact:
    """Test class for compact function"""
    
    @pytest.fixture(autouse=True)
    def patch_type(self, monkeypatch):
        monkeypatch.setattr('builtins.type', mock_type)

    def test_compact_minimal_classes(self):
        """Test compact with classes that are always minimal"""
        minimal_classes = ['Capsule', 'Ellipsoid', 'EmptySet', 'Fullspace', 
                           'Halfspace', 'Interval', 'ZonoBundle', 'SpectraShadow', 'Taylm']
        
        for class_name in minimal_classes:
            S = MockContSet(class_name, 2)
            result = compact(S)
            assert result is S  # Should return the same object
    
    def test_compact_zonotope_defaults(self):
        """Test compact with Zonotope using default parameters"""
        S = MockContSet("Zonotope", 2)
        result = compact(S)
        
        assert result is not None
        assert mock_type(result).__name__ == "Zonotope"
    
    def test_compact_zonotope_methods(self):
        """Test compact with Zonotope using different methods"""
        S = MockContSet("Zonotope", 2)
        
        valid_methods = ['all', 'zeros', 'aligned']
        for method in valid_methods:
            result = compact(S, method=method)
            assert result is not None
        
        # Test invalid method
        with pytest.raises(ValueError, match="Invalid method"):
            compact(S, method='invalid')
    
    def test_compact_polytope_methods(self):
        """Test compact with Polytope using different methods"""
        S = MockContSet("Polytope", 2)
        
        valid_methods = ['all', 'zeros', 'A', 'Ae', 'aligned', 'V', 'AtoAe']
        for method in valid_methods:
            result = compact(S, method=method)
            assert result is not None
        
        # Test invalid method
        with pytest.raises(ValueError, match="Invalid method"):
            compact(S, method='invalid')
    
    def test_compact_conzonotope_methods(self):
        """Test compact with ConZonotope using different methods"""
        S = MockContSet("ConZonotope", 2)
        
        valid_methods = ['all', 'zeros']
        for method in valid_methods:
            result = compact(S, method=method)
            assert result is not None
        
        # Test invalid method
        with pytest.raises(ValueError, match="Invalid method"):
            compact(S, method='invalid')
    
    def test_compact_polyzonotope_methods(self):
        """Test compact with PolyZonotope using different methods"""
        S = MockContSet("PolyZonotope", 2)
        
        valid_methods = ['all', 'states', 'exponentMatrix']
        for method in valid_methods:
            result = compact(S, method=method)
            assert result is not None
        
        # Test invalid method
        with pytest.raises(ValueError, match="Invalid method"):
            compact(S, method='invalid')
    
    def test_compact_tolerance_validation(self):
        """Test compact with tolerance validation"""
        S = MockContSet("Zonotope", 2)
        
        # Valid tolerance
        result = compact(S, tol=1e-6)
        assert result is not None
        
        # Invalid tolerance - negative
        with pytest.raises(ValueError, match="tol must be a non-negative number"):
            compact(S, tol=-1)
        
        # Invalid tolerance - non-numeric
        with pytest.raises(ValueError, match="tol must be a non-negative number"):
            compact(S, tol="invalid")
    
    def test_compact_aligned_method_tolerance_reset(self):
        """Test that aligned method resets tolerance to 1e-3"""
        class ZonotopeWithTolCheck(MockContSet):
            def compact_(self, method, tol):
                if method == 'aligned':
                    assert tol == 1e-3, f"Expected tolerance 1e-3 for aligned method, got {tol}"
                return super().compact_(method, tol)
        
        S = ZonotopeWithTolCheck("Zonotope", 2)
        result = compact(S, method='aligned')  # Should reset tol to 1e-3
        assert result is not None
    
    def test_compact_unknown_class(self):
        """Test compact with unknown class type"""
        S = MockContSet("UnknownClass", 2)
        result = compact(S)  # Should use default values
        assert result is not None
    
    def test_compact_not_implemented_error(self):
        """Test compact when method is not implemented"""
        class UnimplementedSet(MockContSet):
            def compact_(self, method, tol):
                from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
                error = CORAerror('CORA:noops', 'compact not implemented')
                error.identifier = ''
                raise error
        
        S = UnimplementedSet("CustomSet", 2)
        with pytest.raises(CORAerror, match="compact not implemented"):
            compact(S)
    
    def test_compact_other_exception(self):
        """Test compact when other exception occurs"""
        class ErrorSet(MockContSet):
            def compact_(self, method, tol):
                raise RuntimeError("Some other error")
        
        S = ErrorSet("ErrorSet", 2)
        with pytest.raises(RuntimeError, match="Some other error"):
            compact(S)


if __name__ == "__main__":
    pytest.main([__file__]) 