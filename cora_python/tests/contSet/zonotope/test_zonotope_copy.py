"""
test_zonotope_copy - unit test function of copy

Syntax:
    python -m pytest test_zonotope_copy.py

Inputs:
    -

Outputs:
    test results

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 02-October-2024 (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import pytest
import numpy as np

from cora_python.contSet.zonotope import Zonotope


class TestZonotopeCopy:
    """Test class for zonotope copy method"""
    
    def test_basic_copy(self):
        """Test basic copy of zonotope"""
        # 2D zonotope
        Z = Zonotope(np.array([2]), np.array([[4]]))
        Z_copy = Z.copy()
        
        assert Z.isequal(Z_copy)


if __name__ == "__main__":
    test_instance = TestZonotopeCopy()
    
    # Run all tests
    test_instance.test_basic_copy()
    
    print("All zonotope copy tests passed!") 