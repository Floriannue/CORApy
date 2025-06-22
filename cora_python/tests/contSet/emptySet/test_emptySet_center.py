"""
test_emptySet_center - unit tests for emptySet/center

Syntax:
    python -m pytest cora_python/tests/contSet/emptySet/test_emptySet_center.py

Authors: Python translation by AI Assistant  
Written: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.emptySet import EmptySet


class TestEmptySetCenter:
    """Test class for emptySet center method"""
    
    def test_center_2d(self):
        """Test center method for 2D empty set"""
        O = EmptySet(2)
        c = O.center()
        
        # Should return empty array with shape (dimension, 0)
        assert c.shape == (2, 0)
        assert c.size == 0
        
    def test_center_3d(self):
        """Test center method for 3D empty set"""
        O = EmptySet(3)
        c = O.center()
        
        assert c.shape == (3, 0)
        assert c.size == 0
        
    def test_center_1d(self):
        """Test center method for 1D empty set"""
        O = EmptySet(1)
        c = O.center()
        
        assert c.shape == (1, 0)
        assert c.size == 0
        
    def test_center_0d(self):
        """Test center method for 0D empty set"""
        O = EmptySet(0)
        c = O.center()
        
        assert c.shape == (0, 0)
        assert c.size == 0


if __name__ == '__main__':
    pytest.main([__file__]) 