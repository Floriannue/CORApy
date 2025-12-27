"""
test_polyZonotope_compact_ - unit test function for polyZonotope compact_

Tests the redundancy removal functionality for polyZonotopes.

Authors: MATLAB: (no specific test file found, based on compact_.m example)
         Python: AI Assistant
"""

import numpy as np
import pytest
from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope


class TestPolyZonotopeCompact:
    """Test class for polyZonotope compact_ method"""
    
    def test_compact_example(self):
        """Test compact_ with example from MATLAB documentation"""
        # Example from MATLAB compact_.m:
        # pZ = polyZonotope([1;2],[1 3 1 -1 0;0 1 1 1 2], [],[1 0 1 0 1;0 1 2 0 2])
        c = np.array([[1], [2]])
        G = np.array([[1, 3, 1, -1, 0], [0, 1, 1, 1, 2]])
        GI = np.array([]).reshape(2, 0)
        E = np.array([[1, 0, 1, 0, 1], [0, 1, 2, 0, 2]])
        
        pZ = PolyZonotope(c, G, GI, E)
        pZ_compact = pZ.compact_('all', np.finfo(float).eps)
        
        # Verify that compact_ doesn't crash and returns a PolyZonotope
        assert isinstance(pZ_compact, PolyZonotope)
        assert pZ_compact.dim() == 2
        
        # Verify that constant parts (columns where sum(E,1) == 0) are moved to center
        # Column 2 and 4 of original E have sum == 0 (constant parts)
        # After constructor's removeRedundantExponents, these may be combined
        # After compact_('all'), constant parts should be in center
        # Verify that E has no constant columns (all columns should have sum > 0)
        if pZ_compact.E.size > 0:
            col_sums = np.sum(pZ_compact.E, axis=0)
            assert np.all(col_sums > 0), "Constant parts should be moved to center"
        
        # Verify that E has no zero rows
        if pZ_compact.E.size > 0:
            row_sums = np.sum(pZ_compact.E, axis=1)
            assert np.all(row_sums > 0), "Zero rows should be removed"
    
    def test_compact_states_remove_zero_generators(self):
        """Test compact_('states') removes zero generators"""
        c = np.array([[1], [2]])
        G = np.array([[1, 0, 3], [0, 0, 1]])  # Column 2 is all zeros
        GI = np.array([[1, 0], [2, 0]])  # Column 2 is all zeros
        E = np.array([[1, 0, 2], [0, 0, 1]])  # Column 2 corresponds to zero generator
        
        pZ = PolyZonotope(c, G, GI, E)
        pZ_compact = pZ.compact_('states', np.finfo(float).eps)
        
        # Zero generators should be removed
        assert pZ_compact.G.shape[1] == 2  # Column 2 removed
        assert pZ_compact.GI.shape[1] == 1  # Column 2 removed
        assert pZ_compact.E.shape[1] == 2  # Column 2 removed
        
        # Verify non-zero generators are kept
        np.testing.assert_array_equal(pZ_compact.G, G[:, [0, 2]])
        np.testing.assert_array_equal(pZ_compact.GI, GI[:, [0]])
        np.testing.assert_array_equal(pZ_compact.E, E[:, [0, 2]])
    
    def test_compact_states_remove_zero_rows_E(self):
        """Test compact_('states') removes zero rows from E after removing zero generators"""
        # Note: Constructor calls removeRedundantExponents which removes zero columns,
        # so we need to create a case where zero columns remain after constructor
        # Actually, let's test the behavior: compact_('states') only removes zero rows
        # if zero generators were removed. Since constructor already removes zero generators,
        # we test that compact_('exponentMatrix') removes zero rows instead.
        c = np.array([[1], [2]])
        G = np.array([[1, 2], [3, 4]])
        GI = np.array([]).reshape(2, 0)
        # Create E with a zero row (row 2)
        # After constructor removes zero columns, zero rows may remain
        E = np.array([[1, 2], [0, 0], [3, 4]])  # Row 2 is all zeros
        
        pZ = PolyZonotope(c, G, GI, E)
        # compact_('states') won't remove zero rows if all generators are non-zero
        # compact_('exponentMatrix') will remove zero rows
        pZ_compact = pZ.compact_('exponentMatrix', np.finfo(float).eps)
        
        # Zero row should be removed by exponentMatrix method
        assert pZ_compact.E.shape[0] == 2  # Row 2 removed
        np.testing.assert_array_equal(pZ_compact.E, E[[0, 2], :])
    
    def test_compact_exponentMatrix_remove_constant_parts(self):
        """Test compact_('exponentMatrix') moves constant parts to center"""
        c = np.array([[1], [2]])
        G = np.array([[1, 2, 3], [4, 5, 6]])
        GI = np.array([]).reshape(2, 0)
        E = np.array([[1, 0, 2], [0, 0, 1]])  # Column 2 has sum(E,1) == 0 (constant)
        
        pZ = PolyZonotope(c, G, GI, E)
        pZ_compact = pZ.compact_('exponentMatrix', np.finfo(float).eps)
        
        # Constant part (column 2) should be added to center and removed from G/E
        assert pZ_compact.G.shape[1] == 2  # Column 2 removed
        assert pZ_compact.E.shape[1] == 2  # Column 2 removed
        # Center should be updated: c + G(:,2) = [1;2] + [2;5] = [3;7]
        expected_c = c + G[:, [1]]  # Column 2 (0-indexed: 1)
        np.testing.assert_array_almost_equal(pZ_compact.c, expected_c)
    
    def test_compact_exponentMatrix_remove_zero_rows(self):
        """Test compact_('exponentMatrix') removes zero rows from E"""
        c = np.array([[1], [2]])
        G = np.array([[1, 2], [3, 4]])
        GI = np.array([]).reshape(2, 0)
        E = np.array([[1, 2], [0, 0], [3, 4]])  # Row 2 is all zeros
        
        pZ = PolyZonotope(c, G, GI, E)
        pZ_compact = pZ.compact_('exponentMatrix', np.finfo(float).eps)
        
        # Zero row should be removed
        assert pZ_compact.E.shape[0] == 2  # Row 2 removed
        np.testing.assert_array_equal(pZ_compact.E, E[[0, 2], :])
    
    def test_compact_all(self):
        """Test compact_('all') performs both operations"""
        c = np.array([[1], [2]])
        G = np.array([[1, 0, 2], [3, 0, 4]])  # Column 2 is all zeros
        GI = np.array([[1, 0], [2, 0]])  # Column 2 is all zeros
        E = np.array([[1, 0, 0], [0, 0, 2], [0, 0, 0]])  # Row 3 is all zeros, column 2 has sum==0
        
        pZ = PolyZonotope(c, G, GI, E)
        pZ_compact = pZ.compact_('all', np.finfo(float).eps)
        
        # Both zero generators and zero rows should be removed
        # Column 2 (constant part) should be added to center
        assert pZ_compact.G.shape[1] == 2  # Column 2 removed
        assert pZ_compact.E.shape[1] == 2  # Column 2 removed
        assert pZ_compact.E.shape[0] == 2  # Row 3 removed
    
    def test_compact_empty_G(self):
        """Test compact_ with empty G"""
        c = np.array([[1], [2]])
        G = np.array([]).reshape(2, 0)
        GI = np.array([[1, 2], [3, 4]])
        E = np.array([]).reshape(0, 0)
        
        pZ = PolyZonotope(c, G, GI, E)
        pZ_compact = pZ.compact_('all', np.finfo(float).eps)
        
        # Should handle empty G/E gracefully
        assert pZ_compact.G.size == 0
        assert pZ_compact.E.size == 0
        np.testing.assert_array_equal(pZ_compact.c, c)
        np.testing.assert_array_equal(pZ_compact.GI, GI)
    
    def test_compact_empty_GI(self):
        """Test compact_ with empty GI"""
        c = np.array([[1], [2]])
        G = np.array([[1, 2], [3, 4]])
        GI = np.array([]).reshape(2, 0)
        E = np.array([[1, 2], [0, 1]])
        
        pZ = PolyZonotope(c, G, GI, E)
        pZ_compact = pZ.compact_('all', np.finfo(float).eps)
        
        # Should handle empty GI gracefully
        assert pZ_compact.GI.size == 0
        np.testing.assert_array_equal(pZ_compact.c, c)
        np.testing.assert_array_equal(pZ_compact.G, G)
        np.testing.assert_array_equal(pZ_compact.E, E)
    
    def test_compact_all_nonzero(self):
        """Test compact_ when all generators are non-zero"""
        c = np.array([[1], [2]])
        G = np.array([[1, 2], [3, 4]])
        GI = np.array([[1], [2]])
        E = np.array([[1, 2], [0, 1]])
        
        pZ = PolyZonotope(c, G, GI, E)
        pZ_compact = pZ.compact_('states', np.finfo(float).eps)
        
        # Should remain unchanged if all generators are non-zero
        np.testing.assert_array_equal(pZ_compact.G, G)
        np.testing.assert_array_equal(pZ_compact.GI, GI)
        np.testing.assert_array_equal(pZ_compact.E, E)
    
    def test_compact_invalid_method(self):
        """Test compact_ with invalid method"""
        c = np.array([[1], [2]])
        G = np.array([[1], [2]])
        GI = np.array([]).reshape(2, 0)
        E = np.array([[1]])
        
        pZ = PolyZonotope(c, G, GI, E)
        
        with pytest.raises(ValueError):
            pZ.compact_('invalid', np.finfo(float).eps)
    
    def test_compact_default_tol(self):
        """Test compact_ with default tolerance"""
        c = np.array([[1], [2]])
        G = np.array([[1, 0], [0, 1]])
        GI = np.array([]).reshape(2, 0)
        E = np.array([[1, 0], [0, 1]])
        
        pZ = PolyZonotope(c, G, GI, E)
        # Should work without specifying tol
        pZ_compact = pZ.compact_('all')
        assert isinstance(pZ_compact, PolyZonotope)
    
    def test_compact_redundant_exponents(self):
        """Test compact_ removes redundant exponent vectors"""
        c = np.array([[1], [2]])
        G = np.array([[1, 2, 3], [4, 5, 6]])
        GI = np.array([]).reshape(2, 0)
        # Columns 1 and 2 have identical exponents
        E = np.array([[1, 1, 2], [0, 0, 1]])
        
        pZ = PolyZonotope(c, G, GI, E)
        pZ_compact = pZ.compact_('exponentMatrix', np.finfo(float).eps)
        
        # Redundant exponents should be combined
        # The two identical exponent columns should be merged
        assert pZ_compact.E.shape[1] <= 2  # At most 2 columns after merging
    
    def test_compact_multiple_constant_parts(self):
        """Test compact_ handles multiple constant parts"""
        c = np.array([[1], [2]])
        G = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        GI = np.array([]).reshape(2, 0)
        # Columns 2 and 4 are constant (sum(E,1) == 0)
        E = np.array([[1, 0, 2, 0], [0, 0, 1, 0]])
        
        pZ = PolyZonotope(c, G, GI, E)
        pZ_compact = pZ.compact_('exponentMatrix', np.finfo(float).eps)
        
        # Constant parts should be moved to center
        # Center should be c + G[:,1] + G[:,3] = [1;2] + [2;6] + [4;8] = [7;16]
        expected_c = c + G[:, [1, 3]].sum(axis=1, keepdims=True)
        np.testing.assert_array_almost_equal(pZ_compact.c, expected_c)
        # Constant columns should be removed
        assert pZ_compact.E.shape[1] == 2  # Only non-constant columns remain
    
    def test_compact_all_zero_generators(self):
        """Test compact_ with all zero generators"""
        c = np.array([[1], [2]])
        G = np.array([[0, 0], [0, 0]])  # All zeros
        GI = np.array([]).reshape(2, 0)
        E = np.array([[1, 2], [0, 1]])
        
        pZ = PolyZonotope(c, G, GI, E)
        pZ_compact = pZ.compact_('states', np.finfo(float).eps)
        
        # All zero generators should be removed
        assert pZ_compact.G.shape[1] == 0 or pZ_compact.G.size == 0
    
    def test_compact_id_preservation(self):
        """Test that compact_ correctly handles id vector"""
        c = np.array([[1], [2]])
        G = np.array([[1, 2], [3, 4]])
        GI = np.array([]).reshape(2, 0)
        E = np.array([[1, 2], [0, 0], [3, 4]])  # Row 2 is all zeros
        id_vec = np.array([[1], [2], [3]])  # Corresponding id vector
        
        pZ = PolyZonotope(c, G, GI, E, id_vec)
        pZ_compact = pZ.compact_('exponentMatrix', np.finfo(float).eps)
        
        # Zero row should be removed from both E and id
        assert pZ_compact.E.shape[0] == pZ_compact.id.shape[0]
        assert pZ_compact.E.shape[0] == 2  # Row 2 removed
        # id should contain [1, 3] (row 2 removed)
        np.testing.assert_array_equal(pZ_compact.id.flatten(), [1, 3])

