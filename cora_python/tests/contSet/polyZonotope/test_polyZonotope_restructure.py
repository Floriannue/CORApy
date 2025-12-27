"""
test_polyZonotope_restructure - unit test function for over-approximative
   polynomial zonotope restructuring

Tests the restructure functionality for polyZonotopes.

Authors: MATLAB: Niklas Kochdumper
         Python: AI Assistant
Written:       29-March-2018 (MATLAB)
Python translation: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope


class TestPolyZonotopeRestructure:
    """Test class for polyZonotope restructure method"""
    
    def test_restructure_reduceGirard(self):
        """Test restructure with reduceGirard method"""
        # Test case from MATLAB test_polyZonotope_restructure.m
        # pZ = polyZonotope([0;0],[0 4 1 -1 2; 1 2 -1 -1 1],[-7 1 1;15 1 -1],[1 0 0 0 1;0 1 0 3 2; 0 0 1 1 0]);
        c = np.array([[0], [0]])
        G = np.array([[0, 4, 1, -1, 2], [1, 2, -1, -1, 1]])
        GI = np.array([[-7, 1, 1], [15, 1, -1]])
        E = np.array([[1, 0, 0, 0, 1], [0, 1, 0, 3, 2], [0, 0, 1, 1, 0]])
        
        pZ = PolyZonotope(c, G, GI, E)
        pZres = pZ.restructure('reduceGirard', 2)
        
        # Check that restructure returns a PolyZonotope
        assert isinstance(pZres, PolyZonotope)
        
        # Check that GI is empty (restructure removes independent generators)
        assert pZres.GI.size == 0
        
        # Check dimensions are consistent
        assert pZres.c.shape[0] == 2
        assert pZres.G.shape[0] == 2
        assert pZres.E.shape[1] == pZres.G.shape[1]
        if pZres.E.size > 0:
            assert pZres.E.shape[0] == pZres.id.shape[0]
    
    def test_restructure_zonotopeGirard(self):
        """Test restructure with zonotopeGirard method"""
        c = np.array([[1], [2]])
        G = np.array([[1, 2], [3, 4]])
        GI = np.array([[1], [2]])
        E = np.array([[1, 0], [0, 1]])
        
        pZ = PolyZonotope(c, G, GI, E)
        pZres = pZ.restructure('zonotopeGirard', 2)
        
        # Check that restructure returns a PolyZonotope
        assert isinstance(pZres, PolyZonotope)
        
        # Check that GI is empty (restructure removes independent generators)
        assert pZres.GI.size == 0
        
        # Check dimensions are consistent
        assert pZres.c.shape[0] == 2
        assert pZres.G.shape[0] == 2
        assert pZres.E.shape[1] == pZres.G.shape[1]
    
    def test_restructure_empty_GI(self):
        """Test restructure when GI is already empty"""
        c = np.array([[1], [2]])
        G = np.array([[1, 2], [3, 4]])
        GI = np.array([]).reshape(2, 0)
        E = np.array([[1, 0], [0, 1]])
        
        pZ = PolyZonotope(c, G, GI, E)
        pZres = pZ.restructure('reduceGirard', 2)
        
        # Should still work and return a PolyZonotope
        assert isinstance(pZres, PolyZonotope)
        assert pZres.GI.size == 0
    
    def test_restructure_empty_G(self):
        """Test restructure when G is empty"""
        c = np.array([[1], [2]])
        G = np.array([]).reshape(2, 0)
        GI = np.array([[1, 2], [3, 4]])
        E = np.array([]).reshape(0, 0)
        
        pZ = PolyZonotope(c, G, GI, E)
        pZres = pZ.restructure('reduceGirard', 2)
        
        # Should still work and return a PolyZonotope
        assert isinstance(pZres, PolyZonotope)
        assert pZres.GI.size == 0  # Independent generators should be converted
    
    def test_restructure_with_genOrder(self):
        """Test restructure with genOrder parameter"""
        c = np.array([[1], [2]])
        G = np.array([[1, 2, 3], [4, 5, 6]])
        GI = np.array([[1], [2]])
        E = np.array([[1, 0, 1], [0, 1, 0]])
        
        pZ = PolyZonotope(c, G, GI, E)
        pZres = pZ.restructure('reduceGirard', 2, genOrder=3)
        
        # Should work with genOrder parameter
        assert isinstance(pZres, PolyZonotope)
        assert pZres.GI.size == 0
    
    def test_restructure_invalid_method(self):
        """Test restructure with invalid method"""
        c = np.array([[1], [2]])
        G = np.array([[1], [2]])
        GI = np.array([]).reshape(2, 0)
        E = np.array([[1]])
        
        pZ = PolyZonotope(c, G, GI, E)
        
        with pytest.raises(Exception):  # CORAerror or ValueError
            pZ.restructure('invalid', 2)
    
    def test_restructure_reduceFull(self):
        """Test restructure with reduceFull method"""
        c = np.array([[1], [2]])
        G = np.array([[1, 2], [3, 4]])
        GI = np.array([[1], [2]])
        E = np.array([[1, 0], [0, 1]])
        
        pZ = PolyZonotope(c, G, GI, E)
        pZres = pZ.restructure('reduceFullGirard', 2)
        
        # Should work
        assert isinstance(pZres, PolyZonotope)
        assert pZres.GI.size == 0
    
    def test_restructure_reducePart(self):
        """Test restructure with reducePart method"""
        c = np.array([[1], [2]])
        G = np.array([[1, 2], [3, 4]])
        GI = np.array([[1], [2]])
        E = np.array([[1, 0], [0, 1]])
        
        pZ = PolyZonotope(c, G, GI, E)
        pZres = pZ.restructure('reducePartGirard', 2)
        
        # Should work
        assert isinstance(pZres, PolyZonotope)
        assert pZres.GI.size == 0

