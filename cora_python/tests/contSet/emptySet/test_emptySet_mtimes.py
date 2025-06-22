"""
test_emptySet_mtimes - unit tests for emptySet/mtimes

Syntax:
    python -m pytest cora_python/tests/contSet/emptySet/test_emptySet_mtimes.py

Authors: Python translation by AI Assistant
Written: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.emptySet import EmptySet


class TestEmptySetMtimes:
    """Test class for emptySet mtimes method"""
    
    def test_mtimes_scalar(self):
        """Test mtimes with scalar"""
        O = EmptySet(2)
        result = O.mtimes(2)
        
        # Result should be an EmptySet with same dimension
        assert isinstance(result, EmptySet)
        assert result.dimension == 2
        
    def test_mtimes_matrix_projection(self):
        """Test mtimes with matrix for projection"""
        O = EmptySet(3)
        M = np.array([[1, 0, 1], [0, 1, 0]])  # 2x3 matrix
        
        result = O.mtimes(M)
        
        # Result should be an EmptySet with dimension equal to number of rows in M
        assert isinstance(result, EmptySet)
        assert result.dimension == 2
        
    def test_mtimes_square_matrix(self):
        """Test mtimes with square matrix"""
        O = EmptySet(2)
        M = np.array([[2, 1], [-1, 3]])
        
        result = O.mtimes(M)
        
        # Result should be an EmptySet with same dimension
        assert isinstance(result, EmptySet)
        assert result.dimension == 2


if __name__ == '__main__':
    pytest.main([__file__]) 