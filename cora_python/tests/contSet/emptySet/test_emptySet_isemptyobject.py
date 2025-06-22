"""
test_emptySet_isemptyobject - unit tests for emptySet/isemptyobject

Syntax:
    python -m pytest cora_python/tests/contSet/emptySet/test_emptySet_isemptyobject.py

Authors: Python translation by AI Assistant
Written: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.emptySet import EmptySet


class TestEmptySetIsEmptyObject:
    """Test class for emptySet isemptyobject method"""
    
    def test_isemptyobject_false_2d(self):
        """Test isemptyobject returns False for 2D empty set"""
        O = EmptySet(2)
        # In MATLAB, this always returns False for emptySet objects
        # The emptySet object itself is not "empty" - it represents the empty set
        assert O.isemptyobject() == False
        
    def test_isemptyobject_false_3d(self):
        """Test isemptyobject returns False for 3D empty set"""
        O = EmptySet(3)
        assert O.isemptyobject() == False
        
    def test_isemptyobject_false_1d(self):
        """Test isemptyobject returns False for 1D empty set"""
        O = EmptySet(1)
        assert O.isemptyobject() == False
        
    def test_isemptyobject_false_0d(self):
        """Test isemptyobject returns False for 0D empty set"""
        O = EmptySet(0)
        assert O.isemptyobject() == False
        
    def test_isemptyobject_false_default(self):
        """Test isemptyobject returns False for empty set (matching MATLAB test)"""
        # init empty set
        n = 2
        O = EmptySet(n)
        
        # check result
        assert not O.isemptyobject()


if __name__ == '__main__':
    pytest.main([__file__]) 