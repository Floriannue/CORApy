"""
test_and_op - unit test function for and_op (intersection)

Syntax:
    pytest test_and_op.py

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
from cora_python.contSet.contSet.and_op import and_op


class MockContSet:
    """Mock ContSet for testing and_op method"""
    
    def __init__(self, dim_val=2, empty=False, name="MockSet"):
        self._dim = dim_val
        self._empty = empty
        self._name = name
        
    def dim(self):
        return self._dim
    
    def isemptyobject(self):
        return self._empty
    
    def and_(self, S2, varargin):
        """Mock implementation of intersection"""
        if self._empty or S2.isemptyobject():
            # Return empty set
            return MockContSet(self._dim, empty=True, name="EmptyIntersection")
        else:
            # Return non-empty intersection
            return MockContSet(self._dim, empty=False, name=f"Intersection({self._name},{S2._name})")


class TestAndOp:
    """Test class for and_op function"""
    
    def test_and_op_basic(self):
        """Test basic and_op functionality"""
        
        S1 = MockContSet(2, name="Set1")
        S2 = MockContSet(2, name="Set2")
        
        result = and_op(S1, S2)
        
        assert not result.isemptyobject()
        assert result.dim() == 2
    
    def test_and_op_empty_sets(self):
        """Test and_op with empty sets"""
        
        # First set empty
        S1 = MockContSet(2, empty=True, name="EmptySet1")
        S2 = MockContSet(2, name="Set2")
        
        result = and_op(S1, S2)
        assert result.isemptyobject()
        
        # Second set empty
        S1 = MockContSet(2, name="Set1")
        S2 = MockContSet(2, empty=True, name="EmptySet2")
        
        result = and_op(S1, S2)
        assert result.isemptyobject()
        
        # Both sets empty
        S1 = MockContSet(2, empty=True, name="EmptySet1")
        S2 = MockContSet(2, empty=True, name="EmptySet2")
        
        result = and_op(S1, S2)
        assert result.isemptyobject()
    
    def test_and_op_dimension_mismatch(self):
        """Test and_op with dimension mismatch"""
        
        S1 = MockContSet(2)
        S2 = MockContSet(3)
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            and_op(S1, S2)
    
    def test_and_op_with_method(self):
        """Test and_op with specific method"""
        
        S1 = MockContSet(2, name="Set1")
        S2 = MockContSet(2, name="Set2")
        
        # Test with method parameter
        result = and_op(S1, S2, method='exact')
        assert not result.isemptyobject()
    
    def test_and_op_self_intersection(self):
        """Test and_op with same set (self-intersection)"""
        
        S1 = MockContSet(2, name="Set1")
        
        result = and_op(S1, S1)
        assert not result.isemptyobject()
        assert result.dim() == 2
    
    def test_and_op_different_types(self):
        """Test and_op with different set types"""
        
        class MockInterval(MockContSet):
            def __init__(self, dim_val=2, empty=False):
                super().__init__(dim_val, empty, "Interval")
        
        class MockZonotope(MockContSet):
            def __init__(self, dim_val=2, empty=False):
                super().__init__(dim_val, empty, "Zonotope")
        
        S1 = MockInterval(2)
        S2 = MockZonotope(2)
        
        result = and_op(S1, S2)
        assert not result.isemptyobject()


if __name__ == "__main__":
    pytest.main([__file__]) 