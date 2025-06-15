"""
test_full_fact - unit test function for full_fact

Tests the full_fact function for generating full factorial design matrices.

Authors: Python translation by AI Assistant
Date: 2025
"""

import pytest
import numpy as np
from cora_python.g.functions.matlab.init.full_fact import full_fact


class TestFullFact:
    def test_full_fact_basic(self):
        """Test basic full_fact functionality"""
        x1 = [-1, 1]
        x2 = [100, 200, 300]
        
        des_mat = full_fact(x1, x2)
        
        # Should have 2*3 = 6 rows and 2 columns
        assert des_mat.shape == (6, 2)
        
        # Check that all combinations are present
        expected_combinations = [
            [-1, 100], [-1, 200], [-1, 300],
            [1, 100], [1, 200], [1, 300]
        ]
        
        # Sort both arrays for comparison
        des_mat_sorted = des_mat[np.lexsort((des_mat[:, 1], des_mat[:, 0]))]
        expected_sorted = np.array(sorted(expected_combinations))
        
        assert np.allclose(des_mat_sorted, expected_sorted)
    
    def test_full_fact_three_variables(self):
        """Test with three variables"""
        x1 = [1, 2]
        x2 = [10, 20]
        x3 = [100, 200]
        
        des_mat = full_fact(x1, x2, x3)
        
        # Should have 2*2*2 = 8 rows and 3 columns
        assert des_mat.shape == (8, 3)
        
        # Check that we have all unique combinations
        unique_rows = np.unique(des_mat, axis=0)
        assert unique_rows.shape[0] == 8
    
    def test_full_fact_different_sizes(self):
        """Test with variables of different sizes"""
        x1 = [1, 2, 3]
        x2 = [10, 20]
        
        des_mat = full_fact(x1, x2)
        
        # Should have 3*2 = 6 rows and 2 columns
        assert des_mat.shape == (6, 2)
        
        # Check that all values from x1 and x2 appear
        assert set(des_mat[:, 0]) == {1, 2, 3}
        assert set(des_mat[:, 1]) == {10, 20}
    
    def test_full_fact_single_level_error(self):
        """Test error when variable has only one level"""
        x1 = [1]  # Only one level
        x2 = [10, 20]
        
        with pytest.raises(Exception):  # Should raise CORAerror
            full_fact(x1, x2)
    
    def test_full_fact_insufficient_variables(self):
        """Test error when less than 2 variables"""
        x1 = [1, 2]
        
        with pytest.raises(ValueError):
            full_fact(x1)


if __name__ == "__main__":
    pytest.main([__file__]) 