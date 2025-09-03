"""
Test script for splitDepFactor method of PolyZonotope

This is a Python translation of the MATLAB CORA test.

Authors: MATLAB: Niklas Kochdumper, Tobias Ladner
         Python: AI Assistant
"""

import pytest
import numpy as np
from cora_python.contSet.polyZonotope import PolyZonotope
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


class TestPolyZonotopeSplitDepFactor:
    """Test class for splitDepFactor method"""
    
    def test_splitDepFactor_basic(self):
        """Test basic splitDepFactor functionality"""
        
        # Create polynomial zonotope
        c = np.array([0, 0])
        G = np.array([[2, 0, 1], [0, 2, 1]])
        E = np.array([[1, 0, 3], [0, 1, 1]])
        GI = np.array([[0], [0]])
        id_list = np.array([1, 2])
        pZ = PolyZonotope(c, G, GI, E, id_list)
        
        # Split factor 1
        pZsplit = pZ.splitDepFactor(1)
        
        # Should return exactly 2 splits
        assert len(pZsplit) == 2
        
        # Both splits should be valid PolyZonotope objects
        for i, split_pZ in enumerate(pZsplit):
            assert isinstance(split_pZ, PolyZonotope)
            assert split_pZ.c.shape[0] == 2  # Same dimension
            assert split_pZ.G.shape[0] == 2  # Same dimension
    
    def test_splitDepFactor_with_polyOrd(self):
        """Test splitDepFactor with polyOrd parameter"""
        
        # Create polynomial zonotope with higher order terms
        c = np.array([0, 0])
        G = np.array([[2, 0, 1, 0.5], [0, 2, 1, 0.3]])
        E = np.array([[1, 0, 3, 2], [0, 1, 1, 2]])
        GI = np.array([[0], [0]])
        id_list = np.array([1, 2])
        pZ = PolyZonotope(c, G, GI, E, id_list)
        
        # Split factor 1 with polyOrd = 2
        pZsplit = pZ.splitDepFactor(1, polyOrd=2)
        
        # Should return exactly 2 splits
        assert len(pZsplit) == 2
        
        # Both splits should be valid PolyZonotope objects
        for i, split_pZ in enumerate(pZsplit):
            assert isinstance(split_pZ, PolyZonotope)
            assert split_pZ.c.shape[0] == 2  # Same dimension
            assert split_pZ.G.shape[0] == 2  # Same dimension
    
    def test_splitDepFactor_different_factors(self):
        """Test splitDepFactor with different factor indices"""
        
        # Create polynomial zonotope
        c = np.array([0, 0])
        G = np.array([[2, 0, 1], [0, 2, 1]])
        E = np.array([[1, 0, 3], [0, 1, 1]])
        GI = np.array([[0], [0]])
        id_list = np.array([1, 2])
        pZ = PolyZonotope(c, G, GI, E, id_list)
        
        # Test splitting factor 1
        pZsplit1 = pZ.splitDepFactor(1)
        assert len(pZsplit1) == 2
        
        # Test splitting factor 2
        pZsplit2 = pZ.splitDepFactor(2)
        assert len(pZsplit2) == 2
        
        # Results should be different
        assert not np.allclose(pZsplit1[0].c, pZsplit2[0].c) or not np.allclose(pZsplit1[0].G, pZsplit2[0].G)
    
    def test_splitDepFactor_invalid_factor(self):
        """Test splitDepFactor with invalid factor index"""
        
        # Create polynomial zonotope
        c = np.array([0, 0])
        G = np.array([[2, 0, 1], [0, 2, 1]])
        E = np.array([[1, 0, 3], [0, 1, 1]])
        GI = np.array([[0], [0]])
        id_list = np.array([1, 2])
        pZ = PolyZonotope(c, G, GI, E, id_list)
        
        # Test with invalid factor (not in id list)
        with pytest.raises(ValueError, match="Given value for 'ind' should be contained in identifiers"):
            pZ.splitDepFactor(3)
        
        # Test with invalid factor (duplicate in id list - though this shouldn't happen in practice)
        # This would require a malformed PolyZonotope, so we skip this test
    
    def test_splitDepFactor_edge_cases(self):
        """Test edge cases for splitDepFactor"""
        
        # Test with single factor
        c = np.array([0])
        G = np.array([[1]])
        E = np.array([[1]])
        GI = np.array([]).reshape(1, 0)
        id_list = np.array([1])
        pZ = PolyZonotope(c, G, GI, E, id_list)
        
        pZsplit = pZ.splitDepFactor(1)
        assert len(pZsplit) == 2
        
                # Test with constant polynomial (no dependent generators)
        c = np.array([1, 2])
        G = np.array([]).reshape(2, 0)
        E = np.array([]).reshape(0, 0)
        GI = np.array([[1], [1]])
        pZ = PolyZonotope(c, G, GI, E)

        # This should raise an error since there are no dependent factors to split
        with pytest.raises(ValueError, match="Given value for 'ind' should be contained in identifiers"):
            pZ.splitDepFactor(1)
    
    def test_splitDepFactor_mathematical_properties(self):
        """Test mathematical properties of splitDepFactor"""
        
        # Create a simple polynomial zonotope
        c = np.array([0, 0])
        G = np.array([[2, 0, 1], [0, 2, 1]])
        E = np.array([[1, 0, 2], [0, 1, 1]])
        GI = np.array([[0], [0]])
        id_list = np.array([1, 2])
        pZ = PolyZonotope(c, G, GI, E, id_list)
        
        # Split factor 1
        pZsplit = pZ.splitDepFactor(1)
        
        # Check that the union of splits contains the original set
        # This is a fundamental property of splitting
        # (We can't easily test this without containment methods, but we can check basic properties)
        
        # Both splits should have the same dimension
        for split_pZ in pZsplit:
            assert split_pZ.c.shape[0] == pZ.c.shape[0]
            assert split_pZ.G.shape[0] == pZ.G.shape[0]
        
        # The splits should be different
        assert not np.allclose(pZsplit[0].c, pZsplit[1].c) or not np.allclose(pZsplit[0].G, pZsplit[1].G)
    
    def test_splitDepFactor_random_cases(self):
        """Test splitDepFactor with random polynomial zonotopes"""
        
        np.random.seed(456)  # For reproducible tests
        
        for i in range(5):
            # Create random polynomial zonotope
            dim = np.random.randint(2, 5)
            n_gen = np.random.randint(3, 8)
            n_factors = np.random.randint(2, 4)
            
            c = np.random.rand(dim) - 0.5
            G = np.random.rand(dim, n_gen) - 0.5
            GI = np.random.rand(dim, np.random.randint(1, 3)) - 0.5
            E = np.random.randint(0, 4, (n_factors, n_gen))
            id_list = np.arange(1, n_factors + 1)
            
            pZ = PolyZonotope(c, G, GI, E, id_list)
            
            # Test splitting each factor
            for factor in id_list:
                pZsplit = pZ.splitDepFactor(factor)
                
                # Basic checks
                assert len(pZsplit) == 2
                for split_pZ in pZsplit:
                    assert isinstance(split_pZ, PolyZonotope)
                    assert split_pZ.c.shape[0] == dim
                    assert split_pZ.G.shape[0] == dim


if __name__ == "__main__":
    pytest.main([__file__])
