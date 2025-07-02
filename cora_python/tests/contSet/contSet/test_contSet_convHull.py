"""
test_convHull - unit test function for convHull

Syntax:
    pytest test_convHull.py

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
from unittest.mock import Mock, patch
from cora_python.contSet.contSet import ContSet
from cora_python.contSet.contSet.convHull import convHull


class MockContSet(ContSet):
    """Mock ContSet for testing convHull method"""
    
    def __init__(self, dim_val=2, empty=False, name="MockSet"):
        super().__init__()
        self._dim = dim_val
        self._empty = empty
        self._name = name
        
    def dim(self):
        return self._dim
    
    def isemptyobject(self):
        return self._empty
    
    def __repr__(self):
        return self._name
    
    def convHull_(self, S2=None, method='exact'):
        return mock_convHull_(self, S2, method)


def mock_convHull_(S1, S2=None, method='exact'):
    """Mock implementation of convHull_"""
    if S2 is None:
        # Single argument case
        return MockContSet(S1.dim(), S1.isemptyobject(), f"Hull({S1})")
    else:
        # Two argument case
        s1_empty = S1.isemptyobject() if hasattr(S1, 'isemptyobject') else False
        s2_empty = S2.isemptyobject() if hasattr(S2, 'isemptyobject') else False

        if s1_empty or s2_empty:
            return MockContSet(S1.dim(), empty=True, name="EmptyHull")
        else:
            return MockContSet(S1.dim(), empty=False, name=f"Hull({S1},{S2})")


def mock_reorder(S1, S2):
    """Mock implementation of reorder - just return as is"""
    return S1, S2


class TestConvHull:
    """Test class for convHull function"""
    
    def test_convHull_single_argument(self):
        """Test convHull with single argument"""
        
        with patch('cora_python.contSet.contSet.convHull.convHull_', mock_convHull_):
            S = MockContSet(2, name="Set1")
            result = convHull(S)
            
            assert result is not None
            assert result.dim() == 2
    
    def test_convHull_two_arguments(self):
        """Test convHull with two arguments"""
        
        with patch('cora_python.contSet.contSet.convHull.convHull_', mock_convHull_), \
             patch('cora_python.contSet.contSet.convHull.reorder', mock_reorder):
            
            S1 = MockContSet(2, name="Set1")
            S2 = MockContSet(2, name="Set2")
            
            result = convHull(S1, S2)
            
            assert result is not None
            assert result.dim() == 2
    
    def test_convHull_methods(self):
        """Test convHull with different methods"""
        
        with patch('cora_python.contSet.contSet.convHull.convHull_', mock_convHull_), \
             patch('cora_python.contSet.contSet.convHull.reorder', mock_reorder):
            
            S1 = MockContSet(2, name="Set1")
            S2 = MockContSet(2, name="Set2")
            
            valid_methods = ['exact', 'outer', 'inner']
            for method in valid_methods:
                result = convHull(S1, S2, method=method)
                assert result is not None
    
    def test_convHull_invalid_method(self):
        """Test convHull with invalid method"""
        
        S1 = MockContSet(2, name="Set1")
        S2 = MockContSet(2, name="Set2")
        
        with pytest.raises(ValueError, match="Invalid method"):
            convHull(S1, S2, method='invalid')
    
    def test_convHull_dimension_mismatch(self):
        """Test convHull with dimension mismatch"""
        
        with patch('cora_python.contSet.contSet.convHull.reorder', mock_reorder):
            S1 = MockContSet(2, name="Set1")
            S2 = MockContSet(3, name="Set2")
            
            with pytest.raises(ValueError, match="Dimension mismatch"):
                convHull(S1, S2)
    
    def test_convHull_empty_sets(self):
        """Test convHull with empty sets"""
        
        with patch('cora_python.contSet.contSet.convHull.convHull_', mock_convHull_), \
             patch('cora_python.contSet.contSet.convHull.reorder', mock_reorder):
            
            # First set empty
            S1 = MockContSet(2, empty=True, name="EmptySet1")
            S2 = MockContSet(2, name="Set2")
            
            result = convHull(S1, S2)
            assert result.isemptyobject()
            
            # Second set empty
            S1 = MockContSet(2, name="Set1")
            S2 = MockContSet(2, empty=True, name="EmptySet2")
            
            result = convHull(S1, S2)
            assert result.isemptyobject()
    
    def test_convHull_with_numpy_array(self):
        """Test convHull with numpy array as second argument"""
        
        with patch('cora_python.contSet.contSet.convHull.convHull_', mock_convHull_), \
             patch('cora_python.contSet.contSet.convHull.reorder', mock_reorder):
            
            S1 = MockContSet(2, name="Set1")
            S2 = np.array([1, 2])  # Point
            
            result = convHull(S1, S2)
            assert result is not None
    
    def test_convHull_with_list(self):
        """Test convHull with list as second argument"""
        
        with patch('cora_python.contSet.contSet.convHull.convHull_', mock_convHull_), \
             patch('cora_python.contSet.contSet.convHull.reorder', mock_reorder):
            
            S1 = MockContSet(2, name="Set1")
            S2 = [1, 2]  # Point as list
            
            result = convHull(S1, S2)
            assert result is not None
    
    def test_convHull_reorder_called(self):
        """Test that reorder is called for argument precedence"""
        
        with patch('cora_python.contSet.contSet.convHull.convHull_', mock_convHull_), \
             patch('cora_python.contSet.contSet.convHull.reorder') as mock_reorder_call:
            
            # Mock reorder to return swapped arguments
            def reorder_swap(S1, S2):
                return S2, S1
            
            mock_reorder_call.side_effect = reorder_swap
            
            S1 = MockContSet(2, name="Set1")
            S2 = MockContSet(2, name="Set2")
            
            result = convHull(S1, S2)
            
            # Verify reorder was called
            mock_reorder_call.assert_called_once_with(S1, S2)
    
    def test_convHull_no_dim_method(self):
        """Test convHull with objects that don't have dim method"""
        
        with patch('cora_python.contSet.contSet.convHull.convHull_', mock_convHull_), \
             patch('cora_python.contSet.contSet.convHull.reorder', mock_reorder):
            
            # Object without dim method
            class NoDimSet:
                pass
            
            S1 = MockContSet(2, name="Set1")
            S2 = NoDimSet()
            
            # Should not raise dimension mismatch error
            result = convHull(S1, S2)
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__]) 