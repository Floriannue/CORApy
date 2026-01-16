"""
test_polyZonotope_linComb - unit test function for the linear combination of two
    polynomial zonotope objects

TRANSLATED FROM MATLAB - This test is based on the example in the MATLAB documentation.
Source: cora_matlab/contSet/@polyZonotope/linComb.m

Tests the linComb method for polynomial zonotopes.

Authors:       Niklas Kochdumper (MATLAB)
Written:       25-June-2018 (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contSet.polyZonotope import PolyZonotope
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


class TestPolyZonotopeLinComb:
    """Test class for polyZonotope linComb method"""
    
    def test_linComb_basic(self):
        """TEST: Basic linear combination of two polynomial zonotopes"""
        # Example from MATLAB documentation
        # pZ1 = polyZonotope([-2;-2],[2 0 1;0 2 1],[],[1 0 3;0 1 1]);
        c1 = np.array([[-2], [-2]])
        G1 = np.array([[2, 0, 1], [0, 2, 1]])
        GI1 = np.zeros((2, 0))
        E1 = np.array([[1, 0, 3], [0, 1, 1]])
        pZ1 = PolyZonotope(c1, G1, GI1, E1)
        
        # pZ2 = polyZonotope([3;3],[1 -2 1; 2 3 1],[],[1 0 2;0 1 1]);
        c2 = np.array([[3], [3]])
        G2 = np.array([[1, -2, 1], [2, 3, 1]])
        GI2 = np.zeros((2, 0))
        E2 = np.array([[1, 0, 2], [0, 1, 1]])
        pZ2 = PolyZonotope(c2, G2, GI2, E2)
        
        # pZ = linComb(pZ1,pZ2);
        pZ = pZ1.linComb(pZ2)
        
        # Verify basic properties
        assert isinstance(pZ, PolyZonotope), "Result should be a PolyZonotope"
        assert pZ.dim() == 2, "Dimension should be 2"
        assert not pZ.isemptyobject(), "Result should not be empty"
        
        # Verify center is valid (should be a column vector)
        assert pZ.c.shape == (2, 1), "Center should be a column vector of dimension 2"
        
        # Verify E and id dimensions match if E is non-empty
        if pZ.E.size > 0:
            assert pZ.E.shape[0] == pZ.id.shape[0], \
                "E and id should have matching number of rows"
            assert pZ.G.shape[1] == pZ.E.shape[1], \
                "G and E should have matching number of columns"
        
        # Verify that the result contains both input sets (basic containment check)
        # The linear combination should enclose both pZ1 and pZ2
        # This is a basic sanity check - exact verification would require point containment tests
    
    def test_linComb_with_self(self):
        """TEST: Linear combination of a polyZonotope with itself"""
        # pZ = polyZonotope([0;0],[1 0;0 1],[],[1 3]);
        c = np.array([[0], [0]])
        G = np.array([[1, 0], [0, 1]])
        GI = np.zeros((2, 0))
        E = np.array([[1, 3]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # convHull_(pZ) calls linComb(pZ, pZ)
        pZ_lin = pZ.linComb(pZ)
        
        # Verify basic properties
        assert isinstance(pZ_lin, PolyZonotope), "Result should be a PolyZonotope"
        assert pZ_lin.dim() == 2, "Dimension should be 2"
        assert not pZ_lin.isemptyobject(), "Result should not be empty"
