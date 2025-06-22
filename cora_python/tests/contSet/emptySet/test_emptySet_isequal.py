"""
test_emptySet_isequal - unit tests for emptySet/isequal

Syntax:
    python -m pytest cora_python/tests/contSet/emptySet/test_emptySet_isequal.py

Authors: Python translation by AI Assistant  
Written: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.emptySet import EmptySet


class TestEmptySetIsequal:
    """Test class for emptySet isequal method"""
    
    def test_isequal_same_dimension(self):
        """Test isequal with same dimension empty sets"""
        O1 = EmptySet(2)
        O2 = EmptySet(2)
        
        # Same dimension empty sets should be equal
        assert O1.isequal(O2) == True
        assert O2.isequal(O1) == True
        
    def test_isequal_different_dimension(self):
        """Test isequal with different dimension empty sets"""
        O1 = EmptySet(2)
        O2 = EmptySet(3)
        
        # Different dimension empty sets should not be equal
        assert O1.isequal(O2) == False
        assert O2.isequal(O1) == False
        
    def test_isequal_self(self):
        """Test isequal with self"""
        O = EmptySet(2)
        
        # Empty set should be equal to itself
        assert O.isequal(O) == True
        
    def test_isequal_empty_array(self):
        """Test isequal with empty numpy arrays"""
        O = EmptySet(2)
        
        # Empty array with correct dimension
        empty_arr = np.empty((2, 0))
        assert O.isequal(empty_arr) == True
        
        # Empty array with wrong dimension
        empty_arr_wrong = np.empty((3, 0))
        assert O.isequal(empty_arr_wrong) == False
        
    def test_isequal_different_dimensions(self):
        """Test isequal for different dimensions"""
        dimensions = [0, 1, 3, 5]
        
        for n in dimensions:
            O1 = EmptySet(n)
            O2 = EmptySet(n)
            O3 = EmptySet(n + 1) if n < 5 else EmptySet(0)
            
            # Same dimension should be equal
            assert O1.isequal(O2) == True
            # Different dimension should not be equal
            assert O1.isequal(O3) == False


if __name__ == '__main__':
    pytest.main([__file__]) 