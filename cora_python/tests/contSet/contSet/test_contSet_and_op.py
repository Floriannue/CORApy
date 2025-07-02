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
from unittest.mock import Mock, MagicMock
from cora_python.contSet.contSet.and_op import and_op
from cora_python.contSet.contSet.contSet import ContSet


def create_mock_class(name):
    class NewMock(MockContSet):
        pass
    NewMock.__name__ = name
    return NewMock


class MockContSet(ContSet):
    """Mock ContSet for testing and_op method"""
    
    def __init__(self, dim_val=2, empty=False, name="MockSet", precedence=100):
        super().__init__()
        self._dim = dim_val
        self._empty = empty
        self._name = name
        self.precedence = precedence
        # Mock the and_ method to allow for tracking calls
        self.and_ = MagicMock(name=f"{name}.and_", side_effect=Exception("Should not be called without a real implementation"))
        self.representsa_ = MagicMock(name=f"{name}.representsa_")
        
    def __repr__(self):
        return f"MockContSet(dim={self._dim}, empty={self._empty}, name='{self._name}')"
        
    def dim(self):
        return self._dim
    
    def isemptyobject(self):
        return self._empty


class TestAndOp:
    """Test class for and_op function"""
    
    def test_and_op_basic(self):
        """Test basic and_op functionality"""
        
        S1 = MockContSet(2, name="Set1", precedence=110)
        S2 = MockContSet(2, name="Set2", precedence=100)

        S1.representsa_.return_value = False
        S2.representsa_.return_value = False
        S2.and_.return_value = "Intersection"
        S2.and_.side_effect = None
        
        result = and_op(S1, S2)
        
        S2.and_.assert_called_once()
        # S2 has lower precedence, so it should be the first argument to and_
        # and S1 should be the second. The call inside and_op is S1.and_(S2, ...),
        # so after reordering S2 becomes S1.
        called_with_S2 = S2.and_.call_args[0][0]
        assert called_with_S2._name == "Set1"

        assert result == "Intersection"

    def test_and_op_empty_sets(self):
        """Test and_op with empty sets"""
        
        # First set empty
        S1 = MockContSet(2, empty=True, name="EmptySet1")
        S2 = MockContSet(2, name="Set2")
        S1.representsa_.return_value = True

        result = and_op(S1, S2)
        assert result == S1
        
        # Second set empty
        S1 = MockContSet(2, name="Set1")
        S2 = MockContSet(2, empty=True, name="EmptySet2")
        S1.representsa_.return_value = False
        S2.representsa_.return_value = True

        result = and_op(S1, S2)
        assert result == S2
        
        # Both sets empty
        S1 = MockContSet(2, empty=True, name="EmptySet1")
        S1.representsa_.return_value = True
        S2 = MockContSet(2, empty=True, name="EmptySet2")
        S2.representsa_.return_value = True
        
        result = and_op(S1, S2)
        assert result == S1
    
    def test_and_op_dimension_mismatch(self):
        """Test and_op with dimension mismatch"""
        
        S1 = MockContSet(2)
        S2 = MockContSet(3)
        
        from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
        with pytest.raises(CORAerror, match="Dimension mismatch"):
            and_op(S1, S2)
    
    def test_and_op_with_method(self):
        """Test and_op with specific method"""
        
        S1 = MockContSet(2, name="Set1")
        S2 = MockContSet(2, name="Set2")
        S1.and_.return_value = "Intersection"
        S1.and_.side_effect = None
        
        # Test with method parameter
        result = and_op(S1, S2, 'exact')
        S1.and_.assert_called_once_with(S2, 'exact')
        assert result == "Intersection"
    
    def test_and_op_self_intersection(self):
        """Test and_op with same set (self-intersection)"""
        
        S1 = MockContSet(2, name="Set1")
        S1.and_.return_value = "Self-Intersection"
        S1.and_.side_effect = None
        
        result = and_op(S1, S1)
        S1.and_.assert_called_once_with(S1, 'exact')
        assert result == "Self-Intersection"

    def test_and_op_different_types(self):
        """Test and_op with different set types"""
        
        MockInterval = create_mock_class("Interval")
        MockZonotope = create_mock_class("Zonotope")

        S1 = MockInterval(2, name="Interval", precedence=120)
        S2 = MockZonotope(2, name="Zonotope", precedence=110)
        
        # S2 has lower precedence, so it should be reordered to be S1
        S2.and_.return_value = "Intersection"
        S2.and_.side_effect = None

        result = and_op(S1, S2)
        
        S2.and_.assert_called_once_with(S1, 'conZonotope')
        assert result == "Intersection"


if __name__ == "__main__":
    pytest.main([__file__]) 