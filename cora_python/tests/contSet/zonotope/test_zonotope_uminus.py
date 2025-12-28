"""
test_zonotope_uminus - unit test function of uminus

This test file matches MATLAB's test_zonotope_uminus.m exactly.

Syntax:
    python -m pytest test_zonotope_uminus.py

Inputs:
    -

Outputs:
    test results

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       06-April-2023 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np

from cora_python.contSet.zonotope import Zonotope


class TestZonotopeUminus:
    """Test class for zonotope uminus method - matches MATLAB exactly"""
    
    def test_uminus_basic(self):
        """Test basic negation: -Z"""
        # MATLAB test case
        c = np.array([[0], [0]])
        G = np.array([[2, 0, 2], [0, 2, 2]])
        Z = Zonotope(c, G)
        
        # Negate
        nZ = -Z
        
        # Check that center and generators are negated
        expected_c = -c
        expected_G = -G
        np.testing.assert_array_equal(nZ.c, expected_c)
        np.testing.assert_array_equal(nZ.G, expected_G)
        
        # Compare with -1 * Z (MATLAB: assert(isequal(nZ, -1*Z)))
        Z_scaled = -1 * Z
        assert nZ.isequal(Z_scaled) or (np.allclose(nZ.c, Z_scaled.c) and np.allclose(nZ.G, Z_scaled.G))
    
    def test_uminus_empty(self):
        """Test negation of empty zonotope"""
        # MATLAB: assert(isemptyobject(-zonotope.empty(2)))
        Z_empty = Zonotope.empty(2)
        nZ_empty = -Z_empty
        assert nZ_empty.isemptyobject()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
