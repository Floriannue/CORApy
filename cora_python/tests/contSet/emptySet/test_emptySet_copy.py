"""
test_emptySet_copy - unit tests for emptySet/copy

Syntax:
    python -m pytest cora_python/tests/contSet/emptySet/test_emptySet_copy.py

Authors: Python translation by AI Assistant  
Written: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.emptySet import EmptySet


class TestEmptySetCopy:
    """Test class for emptySet copy method"""
    
    def test_copy_2d(self):
        """Test copy method for 2D empty set"""
        O = EmptySet(2)
        O_copy = O.copy()
        
        # Should be different objects
        assert O is not O_copy
        # But with same properties
        assert O.dimension == O_copy.dimension
        assert type(O) == type(O_copy)
        
    def test_copy_different_dimensions(self):
        """Test copy method for different dimensions"""
        dimensions = [0, 1, 3, 5, 10]
        
        for n in dimensions:
            O = EmptySet(n)
            O_copy = O.copy()
            
            assert O is not O_copy
            assert O.dimension == O_copy.dimension
            assert O.dimension == n


if __name__ == '__main__':
    pytest.main([__file__]) 