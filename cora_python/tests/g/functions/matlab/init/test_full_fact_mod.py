"""
test_full_fact_mod - unit test function for full_fact_mod

Tests the full_fact_mod function for generating full factorial design matrices with level indices.

Authors: Python translation by AI Assistant
Date: 2025
"""

import pytest
import numpy as np
from cora_python.g.functions.matlab.init.full_fact_mod import full_fact_mod


class TestFullFactMod:
    def test_full_fact_mod_basic(self):
        """Test basic full_fact_mod functionality"""
        levels = [2, 3]  # 2 levels for first variable, 3 for second
        
        des_mat = full_fact_mod(levels)
        
        # Should have 2*3 = 6 rows and 2 columns
        assert des_mat.shape == (6, 2)
        
        # First column should contain values 1, 2
        assert set(des_mat[:, 0]) == {1, 2}
        
        # Second column should contain values 1, 2, 3
        assert set(des_mat[:, 1]) == {1, 2, 3}
        
        # Check that we have all unique combinations
        unique_rows = np.unique(des_mat, axis=0)
        assert unique_rows.shape[0] == 6
    
    def test_full_fact_mod_three_variables(self):
        """Test with three variables"""
        levels = [2, 2, 2]  # 2 levels each
        
        des_mat = full_fact_mod(levels)
        
        # Should have 2*2*2 = 8 rows and 3 columns
        assert des_mat.shape == (8, 3)
        
        # All columns should contain values 1, 2
        for col in range(3):
            assert set(des_mat[:, col]) == {1, 2}
        
        # Check that we have all unique combinations
        unique_rows = np.unique(des_mat, axis=0)
        assert unique_rows.shape[0] == 8
    
    def test_full_fact_mod_different_levels(self):
        """Test with different number of levels"""
        levels = [3, 2, 4]  # 3, 2, 4 levels respectively
        
        des_mat = full_fact_mod(levels)
        
        # Should have 3*2*4 = 24 rows and 3 columns
        assert des_mat.shape == (24, 3)
        
        # Check level ranges
        assert set(des_mat[:, 0]) == {1, 2, 3}
        assert set(des_mat[:, 1]) == {1, 2}
        assert set(des_mat[:, 2]) == {1, 2, 3, 4}
    
    def test_full_fact_mod_single_level_error(self):
        """Test error when variable has only one level"""
        levels = [1, 2]  # First variable has only 1 level
        
        with pytest.raises(Exception):  # Should raise CORAerror
            full_fact_mod(levels)
    
    def test_full_fact_mod_single_variable(self):
        """Test with single variable"""
        levels = [3]  # Single variable with 3 levels
        
        des_mat = full_fact_mod(levels)
        
        # Should have 3 rows and 1 column
        assert des_mat.shape == (3, 1)
        assert np.array_equal(des_mat.flatten(), [1, 2, 3])


if __name__ == "__main__":
    pytest.main([__file__]) 