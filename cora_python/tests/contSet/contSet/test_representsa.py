"""
test_representsa - unit test function for representsa

Syntax:
    pytest test_representsa.py

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
from cora_python.contSet.contSet.representsa import representsa


class MockContSet:
    """Mock ContSet for testing representsa method"""
    
    def __init__(self, dim_val=2, can_represent=True):
        self._dim = dim_val
        self._can_represent = can_represent
        
    def dim(self):
        return self._dim
    
    def representsa_(self, set_type, tol, method, iter_val, splits):
        """Mock implementation of representsa_"""
        if self._can_represent:
            if set_type in ['interval', 'zonotope', 'emptySet']:
                return True
            elif set_type in ['origin', 'point']:
                return False
            else:
                return True
        else:
            return False


class TestRepresentsa:
    """Test class for representsa function"""
    
    def test_representsa_basic(self):
        """Test basic representsa functionality"""
        
        S = MockContSet(2, can_represent=True)
        
        # Test with valid type
        result = representsa(S, 'interval')
        assert result == True
        
        # Test with type that can't be represented
        result = representsa(S, 'origin')
        assert result == False
    
    def test_representsa_valid_types(self):
        """Test representsa with all valid set types"""
        
        S = MockContSet(2, can_represent=True)
        
        valid_types = [
            'capsule', 'conPolyZono', 'conHyperplane', 'conZonotope', 'ellipsoid',
            'halfspace', 'interval', 'levelSet', 'polygon', 'polytope', 'polyZonotope',
            'probZonotope', 'zonoBundle', 'zonotope',  # contSet classes
            'origin', 'point', 'hyperplane', 'parallelotope', 'convexSet',  # special types
            'emptySet', 'fullspace'
        ]
        
        for set_type in valid_types:
            result = representsa(S, set_type)
            assert isinstance(result, bool)
    
    def test_representsa_invalid_type(self):
        """Test representsa with invalid set type"""
        
        S = MockContSet(2)
        
        with pytest.raises(ValueError, match="Unknown set type"):
            representsa(S, 'invalid_type')
    
    def test_representsa_with_tolerance(self):
        """Test representsa with tolerance parameter"""
        
        class TolCheckSet(MockContSet):
            def representsa_(self, set_type, tol, method, iter_val, splits):
                assert tol == 1e-6, f"Expected tolerance 1e-6, got {tol}"
                return True
        
        S = TolCheckSet(2)
        result = representsa(S, 'interval', 1e-6)
        assert result == True
    
    def test_representsa_with_method(self):
        """Test representsa with method parameter"""
        
        class MethodCheckSet(MockContSet):
            def representsa_(self, set_type, tol, method, iter_val, splits):
                assert method == 'polynomial', f"Expected method 'polynomial', got {method}"
                return True
        
        S = MethodCheckSet(2)
        result = representsa(S, 'conPolyZono', 1e-12, 'polynomial')
        assert result == True
    
    def test_representsa_with_iter(self):
        """Test representsa with iter parameter"""
        
        class IterCheckSet(MockContSet):
            def representsa_(self, set_type, tol, method, iter_val, splits):
                assert iter_val == 10, f"Expected iter 10, got {iter_val}"
                return True
        
        S = IterCheckSet(2)
        result = representsa(S, 'conPolyZono', 1e-12, 'linearize', 10)
        assert result == True
    
    def test_representsa_with_splits(self):
        """Test representsa with splits parameter"""
        
        class SplitsCheckSet(MockContSet):
            def representsa_(self, set_type, tol, method, iter_val, splits):
                assert splits == 5, f"Expected splits 5, got {splits}"
                return True
        
        S = SplitsCheckSet(2)
        result = representsa(S, 'conPolyZono', 1e-12, 'linearize', 1, 5)
        assert result == True
    
    def test_representsa_with_kwargs(self):
        """Test representsa with keyword arguments"""
        
        class KwargsCheckSet(MockContSet):
            def representsa_(self, set_type, tol, method, iter_val, splits):
                assert tol == 1e-8, f"Expected tolerance 1e-8, got {tol}"
                assert method == 'interval', f"Expected method 'interval', got {method}"
                assert iter_val == 'fixpoint', f"Expected iter 'fixpoint', got {iter_val}"
                assert splits == 3, f"Expected splits 3, got {splits}"
                return True
        
        S = KwargsCheckSet(2)
        result = representsa(S, 'conPolyZono', tol=1e-8, method='interval', 
                           iter='fixpoint', splits=3)
        assert result == True
    
    def test_representsa_default_values(self):
        """Test representsa with default parameter values"""
        
        class DefaultCheckSet(MockContSet):
            def representsa_(self, set_type, tol, method, iter_val, splits):
                assert tol == 1e-12, f"Expected default tolerance 1e-12, got {tol}"
                assert method == 'linearize', f"Expected default method 'linearize', got {method}"
                assert iter_val == 1, f"Expected default iter 1, got {iter_val}"
                assert splits == 0, f"Expected default splits 0, got {splits}"
                return True
        
        S = DefaultCheckSet(2)
        result = representsa(S, 'interval')
        assert result == True
    
    def test_representsa_tolerance_validation(self):
        """Test representsa with tolerance validation"""
        
        S = MockContSet(2)
        
        # Valid tolerance
        result = representsa(S, 'interval', 1e-10)
        assert isinstance(result, bool)
        
        # Zero tolerance (valid)
        result = representsa(S, 'interval', 0)
        assert isinstance(result, bool)
        
        # Invalid negative tolerance
        with pytest.raises(ValueError, match="Tolerance must be a non-negative number"):
            representsa(S, 'interval', -1e-6)
        
        # Invalid non-numeric tolerance
        with pytest.raises(ValueError, match="Tolerance must be a non-negative number"):
            representsa(S, 'interval', 'invalid')
    
    def test_representsa_mixed_args_kwargs(self):
        """Test representsa with mixed positional and keyword arguments"""
        
        class MixedCheckSet(MockContSet):
            def representsa_(self, set_type, tol, method, iter_val, splits):
                assert tol == 1e-5, f"Expected tolerance 1e-5, got {tol}"
                assert method == 'forwardBackward', f"Expected method 'forwardBackward', got {method}"
                assert iter_val == 15, f"Expected iter 15, got {iter_val}"
                assert splits == 8, f"Expected splits 8, got {splits}"
                return True
        
        S = MixedCheckSet(2)
        # Mix positional and keyword args (kwargs should override)
        result = representsa(S, 'conPolyZono', 1e-5, 'polynomial', iter=15, splits=8, 
                           method='forwardBackward')
        assert result == True
    
    def test_representsa_case_sensitivity(self):
        """Test that set type checking handles case properly"""
        
        S = MockContSet(2)
        
        # Test lowercase (should work)
        result = representsa(S, 'interval')
        assert isinstance(result, bool)
        
        # Test uppercase (should work due to admissible_types list)
        with pytest.raises(ValueError, match="Unknown set type"):
            representsa(S, 'INTERVAL')  # Not in admissible list


if __name__ == "__main__":
    pytest.main([__file__]) 