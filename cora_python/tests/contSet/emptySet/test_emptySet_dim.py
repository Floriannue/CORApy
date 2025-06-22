"""
test_emptySet_dim - unit tests for emptySet/dim

Syntax:
    python -m pytest cora_python/tests/contSet/emptySet/test_emptySet_dim.py

Authors: Python translation by AI Assistant
Written: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.emptySet import EmptySet


class TestEmptySetDim:
    """Test class for emptySet dim method"""
    
    def test_dim_2d(self):
        """Test dimension of 2D empty set"""
        O = EmptySet(2)
        assert O.dim() == 2
        
    def test_dim_3d(self):
        """Test dimension of 3D empty set"""
        O = EmptySet(3)
        assert O.dim() == 3
        
    def test_dim_1d(self):
        """Test dimension of 1D empty set"""
        O = EmptySet(1)
        assert O.dim() == 1
        
    def test_dim_0d(self):
        """Test dimension of 0D empty set"""
        O = EmptySet(0)
        assert O.dim() == 0
        
    def test_dim_default(self):
        """Test dimension of default empty set"""
        O = EmptySet()
        assert O.dim() == 0


if __name__ == '__main__':
    pytest.main([__file__]) 