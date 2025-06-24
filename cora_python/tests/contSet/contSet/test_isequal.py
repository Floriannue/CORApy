"""
test_isequal - unit test function for isequal

Syntax:
    pytest test_isequal.py

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
from cora_python.contSet.contSet.isequal import isequal


class MockContSet:
    """Mock ContSet for testing isequal method"""
    
    def __init__(self, value=1):
        self.value = value
        self.array_attr = np.array([1, 2, 3])


class TestIsequal:
    """Test class for isequal function"""
    
    def test_isequal_same_instance(self):
        """Test isequal with same instance"""
        
        S1 = MockContSet(1)
        result = isequal(S1, S1)
        assert result == True
    
    def test_isequal_different_types(self):
        """Test isequal with different types"""
        
        S1 = MockContSet(1)
        S2 = "not a contSet"
        
        result = isequal(S1, S2)
        assert result == False
    
    def test_isequal_same_type_same_attributes(self):
        """Test isequal with same type and attributes"""
        
        S1 = MockContSet(1)
        S1.test_attr = np.array([1, 2, 3])
        
        S2 = MockContSet(1)
        S2.test_attr = np.array([1, 2, 3])
        
        result = isequal(S1, S2)
        assert result == True
    
    def test_isequal_same_type_different_attributes(self):
        """Test isequal with same type but different attributes"""
        
        S1 = MockContSet(1)
        S1.test_attr = np.array([1, 2, 3])
        
        S2 = MockContSet(1)
        S2.test_attr = np.array([1, 2, 4])
        
        result = isequal(S1, S2)
        assert result == False
    
    def test_isequal_with_tolerance(self):
        """Test isequal with tolerance"""
        
        S1 = MockContSet(1)
        S1.test_attr = np.array([1.0, 2.0, 3.0])
        
        S2 = MockContSet(1)
        S2.test_attr = np.array([1.001, 2.001, 3.001])
        
        # Without tolerance should be False
        result = isequal(S1, S2)
        assert result == False
        
        # With tolerance should be True
        result = isequal(S1, S2, tol=0.01)
        assert result == True
    
    def test_isequal_different_attribute_keys(self):
        """Test isequal with different attribute keys"""
        
        S1 = MockContSet(1)
        S1.attr1 = 1
        
        S2 = MockContSet(1)
        S2.attr2 = 1
        
        result = isequal(S1, S2)
        assert result == False
    
    def test_isequal_mixed_attribute_types(self):
        """Test isequal with mixed attribute types"""
        
        S1 = MockContSet(1)
        S1.scalar_attr = 5
        S1.array_attr = np.array([1, 2, 3])
        S1.string_attr = "test"
        
        S2 = MockContSet(1)
        S2.scalar_attr = 5
        S2.array_attr = np.array([1, 2, 3])
        S2.string_attr = "test"
        
        result = isequal(S1, S2)
        assert result == True
    
    def test_isequal_array_tolerance(self):
        """Test isequal with array tolerance"""
        
        S1 = MockContSet(1)
        S1.array_attr = np.array([1.0, 2.0, 3.0])
        
        S2 = MockContSet(1)
        S2.array_attr = np.array([1.1, 2.1, 3.1])
        
        # Should be False without tolerance
        result = isequal(S1, S2)
        assert result == False
        
        # Should be True with sufficient tolerance
        result = isequal(S1, S2, tol=0.2)
        assert result == True
    
    def test_isequal_array_shape_mismatch(self):
        """Test isequal with different array shapes"""
        
        S1 = MockContSet(1)
        S1.array_attr = np.array([1, 2, 3])
        
        S2 = MockContSet(1)
        S2.array_attr = np.array([[1, 2], [3, 4]])
        
        result = isequal(S1, S2)
        assert result == False


if __name__ == "__main__":
    pytest.main([__file__]) 