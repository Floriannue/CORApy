"""
Test script for splitLongestGen method of PolyZonotope

This is a Python translation of the MATLAB CORA test.

Authors: MATLAB: Niklas Kochdumper
         Python: AI Assistant
"""

import pytest
import numpy as np
from cora_python.contSet.polyZonotope import PolyZonotope
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


class TestPolyZonotopeSplitLongestGen:
    """Test class for splitLongestGen method"""
    
    def test_splitLongestGen_basic(self):
        """Test basic splitLongestGen functionality - matches MATLAB test_polyZonotope_split"""
        
        # Create polynomial zonotope (same as MATLAB test)
        c = np.array([-1, 3])
        G = np.array([[2, 0, 1], [1, 2, 1]])
        E = np.array([[1, 0, 2], [0, 1, 1]])
        GI = np.array([]).reshape(2, 0)
        pZ = PolyZonotope(c, G, GI, E)
        
        # Split the polynomial zonotope at the longest generator
        pZsplit = pZ.splitLongestGen()
        
        # Define ground truth (from MATLAB test)
        c1 = np.array([0, 3.5])
        G1 = np.array([[1, 1/4, 1/2, 1/4], [1/2, 9/4, 1/2, 1/4]])
        E1 = np.array([[1, 0, 1, 2], [0, 1, 1, 1]])
        
        c2 = np.array([-2, 2.5])
        G2 = np.array([[1, 1/4, -1/2, 1/4], [1/2, 9/4, -1/2, 1/4]])
        E2 = np.array([[1, 0, 1, 2], [0, 1, 1, 1]])
        
        # Check centers
        assert np.all(withinTol(c1, pZsplit[0].c.flatten()))
        assert np.all(withinTol(c2, pZsplit[1].c.flatten()))
        
        # Check generators and exponents for first split
        for i in range(E1.shape[1]):
            # Find matching exponent vector
            found = False
            for j in range(pZsplit[0].E.shape[1]):
                if np.allclose(pZsplit[0].E[:, j], E1[:, i]):
                    assert np.allclose(pZsplit[0].G[:, j], G1[:, i])
                    found = True
                    break
            assert found, f"Could not find matching exponent vector for E1[:, {i}]"
        
        # Check generators and exponents for second split
        for i in range(E2.shape[1]):
            # Find matching exponent vector
            found = False
            for j in range(pZsplit[1].E.shape[1]):
                if np.allclose(pZsplit[1].E[:, j], E2[:, i]):
                    assert np.allclose(pZsplit[1].G[:, j], G2[:, i])
                    found = True
                    break
            assert found, f"Could not find matching exponent vector for E2[:, {i}]"
    
    def test_splitLongestGen_with_polyOrd(self):
        """Test splitLongestGen with polyOrd parameter"""
        
        # Create a simple polynomial zonotope
        c = np.array([0, 0])
        G = np.array([[2, 0, 1], [0, 2, 1]])
        E = np.array([[1, 0, 3], [0, 1, 1]])
        GI = np.array([[0], [0]])
        id_list = np.array([1, 2])
        pZ = PolyZonotope(c, G, GI, E, id_list)
        
        # Test with polyOrd = 2
        pZsplit = pZ.splitLongestGen(polyOrd=2)
        
        # Should return exactly 2 splits
        assert len(pZsplit) == 2
        
        # Both splits should be valid PolyZonotope objects
        for i, split_pZ in enumerate(pZsplit):
            assert isinstance(split_pZ, PolyZonotope)
            assert split_pZ.c.shape[0] == 2  # Same dimension
            assert split_pZ.G.shape[0] == 2  # Same dimension
    
    def test_splitLongestGen_edge_cases(self):
        """Test edge cases for splitLongestGen"""
        
        # Test with single generator
        c = np.array([0])
        G = np.array([[1]])
        E = np.array([[1]])
        GI = np.array([]).reshape(1, 0)
        id_list = np.array([1])
        pZ = PolyZonotope(c, G, GI, E, id_list)
        
        pZsplit = pZ.splitLongestGen()
        assert len(pZsplit) == 2
        
                # Test with constant polynomial (no dependent generators)
        c = np.array([1, 2])
        G = np.array([]).reshape(2, 0)
        E = np.array([]).reshape(0, 0)
        GI = np.array([[1], [1]])
        pZ = PolyZonotope(c, G, GI, E)

        # This should raise an error since there are no generators to split
        with pytest.raises(ValueError, match="Cannot split polynomial zonotope with no generators"):
            pZ.splitLongestGen()
    
    def test_splitLongestGen_random_2d(self):
        """Test splitLongestGen with random 2D polynomial zonotopes"""
        
        np.random.seed(42)  # For reproducible tests
        
        for i in range(3):
            # Create random 2D polynomial zonotope
            c = np.random.rand(2) - 0.5
            G = np.random.rand(2, 7) - 0.5
            # Make some generators smaller
            ind = np.random.choice(7, 4, replace=False)
            G[:, ind] = G[:, ind] / 10
            GI = np.random.rand(2, 1) - 0.5
            E = np.hstack([np.eye(2), np.random.randint(0, 6, (2, 5))])
            pZ = PolyZonotope(c, G, GI, E)
            
            # Split with and without polyOrd
            if i < 2:
                pZsplit = pZ.splitLongestGen()
            else:
                pZsplit = pZ.splitLongestGen(polyOrd=2)
            
            # Basic checks
            assert len(pZsplit) == 2
            for split_pZ in pZsplit:
                assert isinstance(split_pZ, PolyZonotope)
                assert split_pZ.c.shape[0] == 2
                assert split_pZ.G.shape[0] == 2
    
    def test_splitLongestGen_random_4d(self):
        """Test splitLongestGen with random 4D polynomial zonotopes"""
        
        np.random.seed(123)  # For reproducible tests
        
        for i in range(3):
            # Create random 4D polynomial zonotope
            c = np.random.rand(4) - 0.5
            G = np.random.rand(4, 6) - 0.5
            # Make some generators smaller
            ind = np.random.choice(6, 4, replace=False)
            G[:, ind] = G[:, ind] / 10
            GI = np.random.rand(4, 2) - 0.5
            E = np.hstack([np.eye(4), np.random.randint(0, 6, (4, 2))])
            pZ = PolyZonotope(c, G, GI, E)
            
            # Split with and without polyOrd
            if i < 2:
                pZsplit = pZ.splitLongestGen()
            else:
                pZsplit = pZ.splitLongestGen(polyOrd=2)
            
            # Basic checks
            assert len(pZsplit) == 2
            for split_pZ in pZsplit:
                assert isinstance(split_pZ, PolyZonotope)
                assert split_pZ.c.shape[0] == 4
                assert split_pZ.G.shape[0] == 4


if __name__ == "__main__":
    pytest.main([__file__])
