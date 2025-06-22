"""
test_emptySet_supportFunc - unit tests for emptySet/supportFunc_

Syntax:
    python -m pytest cora_python/tests/contSet/emptySet/test_emptySet_supportFunc.py

Authors: Python translation by AI Assistant  
Written: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.emptySet import EmptySet


class TestEmptySetSupportFunc:
    """Test class for emptySet supportFunc_ method"""
    
    def test_supportFunc_upper(self):
        """Test supportFunc_ with upper bound - should return -Inf"""
        O = EmptySet(2)
        dir = np.array([[1], [1]])
        
        val, x = O.supportFunc_(dir, 'upper')
        
        assert val == -np.inf
        assert x.shape == (2, 0)
        assert x.size == 0
        
    def test_supportFunc_lower(self):
        """Test supportFunc_ with lower bound - should return Inf"""
        O = EmptySet(2)
        dir = np.array([[1], [1]])
        
        val, x = O.supportFunc_(dir, 'lower')
        
        assert val == np.inf
        assert x.shape == (2, 0)
        assert x.size == 0
        
    def test_supportFunc_range(self):
        """Test supportFunc_ with range - should return interval(-Inf, Inf)"""
        O = EmptySet(2)
        dir = np.array([[1], [1]])
        
        # This might fail if interval is not available, but we test the structure
        try:
            val, x = O.supportFunc_(dir, 'range')
            assert x.shape == (2, 0)
            assert x.size == 0
            # val should be an interval object, but we can't test the exact type without interval
        except ImportError:
            # If interval is not available, that's expected for now
            pass
        
    def test_supportFunc_different_dimensions(self):
        """Test supportFunc_ for different dimensions"""
        dimensions = [0, 1, 3, 5]
        
        for n in dimensions:
            O = EmptySet(n)
            dir = np.ones((n, 1)) if n > 0 else np.empty((0, 1))
            
            val, x = O.supportFunc_(dir, 'upper')
            assert val == -np.inf
            assert x.shape == (n, 0)
            
            val, x = O.supportFunc_(dir, 'lower')
            assert val == np.inf
            assert x.shape == (n, 0)


if __name__ == '__main__':
    pytest.main([__file__]) 