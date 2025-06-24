"""
test_removeRedundantExponents - unit test function for removeRedundantExponents

Syntax:
    pytest test_removeRedundantExponents.py

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
from cora_python.g.functions.helper.sets.polyZonotope.removeRedundantExponents import removeRedundantExponents


class TestRemoveRedundantExponents:
    """Test class for removeRedundantExponents function"""
    
    def test_removeRedundantExponents_basic(self):
        """Test basic removeRedundantExponents functionality"""
        
        # Test case with redundant exponents
        E = np.array([[1, 2, 1], [0, 1, 0]], dtype=float)
        G = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        
        Enew, Gnew = removeRedundantExponents(E, G)
        
        # Should combine columns with identical exponents
        # Columns 1 and 3 have same exponents [1,0]
        expected_E = np.array([[1, 2], [0, 1]], dtype=float)
        expected_G = np.array([[4, 2], [10, 5]], dtype=float)  # [1+3, 4+6]
        
        # Sort by first row for comparison
        order_new = np.argsort(Enew[0, :])
        order_exp = np.argsort(expected_E[0, :])
        
        assert np.allclose(Enew[:, order_new], expected_E[:, order_exp])
        assert np.allclose(Gnew[:, order_new], expected_G[:, order_exp])
    
    def test_removeRedundantExponents_no_redundancy(self):
        """Test removeRedundantExponents with no redundant exponents"""
        
        # Test case with all unique exponents
        E = np.array([[1, 2, 3], [0, 1, 2]], dtype=float)
        G = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        
        Enew, Gnew = removeRedundantExponents(E, G)
        
        # Should return original matrices (possibly reordered)
        assert Enew.shape == E.shape
        assert Gnew.shape == G.shape
        
        # Check that all original columns are preserved
        for i in range(E.shape[1]):
            found = False
            for j in range(Enew.shape[1]):
                if np.allclose(E[:, i], Enew[:, j]):
                    assert np.allclose(G[:, i], Gnew[:, j])
                    found = True
                    break
            assert found, f"Column {i} not found in result"
    
    def test_removeRedundantExponents_empty_matrices(self):
        """Test removeRedundantExponents with empty matrices"""
        
        # Empty generator matrix
        E = np.array([]).reshape(2, 0)
        G = np.array([]).reshape(3, 0)
        
        Enew, Gnew = removeRedundantExponents(E, G)
        
        assert Enew.shape == E.shape
        assert Gnew.shape == G.shape
        assert Enew.size == 0
        assert Gnew.size == 0
    
    def test_removeRedundantExponents_zero_generators(self):
        """Test removeRedundantExponents with zero generators"""
        
        # Test with some zero generators
        E = np.array([[1, 2, 3], [0, 1, 2]], dtype=float)
        G = np.array([[0, 2, 0], [0, 5, 0]], dtype=float)  # columns 1 and 3 are zero
        
        Enew, Gnew = removeRedundantExponents(E, G)
        
        # Should remove zero generators
        assert Gnew.shape[1] == 1  # Only one non-zero generator remains
        assert np.allclose(Enew, E[:, [1]])
        assert np.allclose(Gnew, G[:, [1]])
    
    def test_removeRedundantExponents_all_zero_generators(self):
        """Test removeRedundantExponents with all zero generators"""
        
        # All generators are zero
        E = np.array([[1, 2, 3], [0, 1, 2]], dtype=float)
        G = np.array([[0, 0, 0], [0, 0, 0]], dtype=float)
        
        Enew, Gnew = removeRedundantExponents(E, G)
        
        # Should return single zero column
        assert Enew.shape == (2, 1)
        assert Gnew.shape == (2, 1)
        assert np.allclose(Enew, np.zeros((2, 1)))
        assert np.allclose(Gnew, np.zeros((2, 1)))
    
    def test_removeRedundantExponents_multiple_redundancies(self):
        """Test removeRedundantExponents with multiple sets of redundant exponents"""
        
        # Multiple redundant groups
        E = np.array([[1, 2, 1, 2, 3], [0, 1, 0, 1, 2]], dtype=float)
        G = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=float)
        
        Enew, Gnew = removeRedundantExponents(E, G)
        
        # Should have 3 unique exponent vectors: [1,0], [2,1], [3,2]
        assert Enew.shape[1] == 3
        assert Gnew.shape[1] == 3
        
        # Check that redundancies are combined
        # Find where [1,0] ended up
        found_10 = False
        found_21 = False
        for j in range(Enew.shape[1]):
            if np.allclose(Enew[:, j], [1, 0]):
                assert np.allclose(Gnew[:, j], [4, 14])  # [1+3, 6+8]
                found_10 = True
            elif np.allclose(Enew[:, j], [2, 1]):
                assert np.allclose(Gnew[:, j], [6, 16])  # [2+4, 7+9]
                found_21 = True
        
        assert found_10 and found_21
    
    def test_removeRedundantExponents_single_column(self):
        """Test removeRedundantExponents with single column"""
        
        # Single exponent vector
        E = np.array([[1], [2]], dtype=float)
        G = np.array([[3], [4]], dtype=float)
        
        Enew, Gnew = removeRedundantExponents(E, G)
        
        # Should return the same matrices
        assert np.allclose(Enew, E)
        assert np.allclose(Gnew, G)
    
    def test_removeRedundantExponents_identical_columns(self):
        """Test removeRedundantExponents with all identical columns"""
        
        # All columns have same exponents
        E = np.array([[1, 1, 1], [2, 2, 2]], dtype=float)
        G = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        
        Enew, Gnew = removeRedundantExponents(E, G)
        
        # Should combine all into single column
        assert Enew.shape[1] == 1
        assert Gnew.shape[1] == 1
        assert np.allclose(Enew, [[1], [2]])
        assert np.allclose(Gnew, [[6], [15]])  # [1+2+3, 4+5+6]
    
    def test_removeRedundantExponents_large_matrix(self):
        """Test removeRedundantExponents with larger matrices"""
        
        # Create larger test case
        np.random.seed(42)
        n_exp = 5
        n_gen = 20
        n_dim = 10
        
        # Create some repeated exponent patterns
        E_base = np.random.randint(0, 3, (n_exp, 5))
        E = np.hstack([E_base, E_base, E_base, np.random.randint(0, 3, (n_exp, 5))])
        G = np.random.randn(n_dim, E.shape[1])
        
        Enew, Gnew = removeRedundantExponents(E, G)
        
        # Result should have fewer or equal columns
        assert Enew.shape[1] <= E.shape[1]
        assert Gnew.shape[1] == Enew.shape[1]
        
        # Check dimensions match
        assert Enew.shape[0] == E.shape[0]
        assert Gnew.shape[0] == G.shape[0]
    
    def test_removeRedundantExponents_edge_cases(self):
        """Test edge cases for removeRedundantExponents"""
        
        # Very large exponent values
        E = np.array([[1000, 1000], [2000, 2000]], dtype=float)
        G = np.array([[1, 2], [3, 4]], dtype=float)
        
        Enew, Gnew = removeRedundantExponents(E, G)
        
        # Should combine identical large exponents
        assert Enew.shape[1] == 1
        assert np.allclose(Enew, [[1000], [2000]])
        assert np.allclose(Gnew, [[3], [7]])
        
        # Very small values close to zero
        E = np.array([[1e-10, 1e-10], [2e-10, 2e-10]], dtype=float)
        G = np.array([[1, 2], [3, 4]], dtype=float)
        
        Enew, Gnew = removeRedundantExponents(E, G)
        
        # Should still work with small values
        assert Enew.shape[1] == 1
        assert Gnew.shape[1] == 1


if __name__ == "__main__":
    pytest.main([__file__]) 