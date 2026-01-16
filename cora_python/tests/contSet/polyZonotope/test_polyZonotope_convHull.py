"""
test_polyZonotope_convHull - unit test function for the convex hull of a
    polynomial zonotope and another set representation

TRANSLATED FROM MATLAB - This test is based on the example in the MATLAB documentation
and usage in testLong_polyZonotope_contains.m.
Source: cora_matlab/contSet/@polyZonotope/convHull_.m

Tests the convHull_ method for polynomial zonotopes.

Authors:       Niklas Kochdumper (MATLAB)
Written:       25-June-2018 (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contSet.polyZonotope import PolyZonotope
from cora_python.contSet.interval import Interval
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


class TestPolyZonotopeConvHull:
    """Test class for polyZonotope convHull_ method"""
    
    def test_convHull_single(self):
        """TEST: Convex hull of a single polyZonotope (with itself)"""
        # Example from MATLAB documentation
        # pZ = polyZonotope([0;0],[1 0;0 1],[],[1 3]);
        c = np.array([[0], [0]])
        G = np.array([[1, 0], [0, 1]])
        GI = np.zeros((2, 0))
        E = np.array([[1, 3]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # S_out = convHull(pZ);
        S_out = pZ.convHull_()
        
        # Verify basic properties
        assert isinstance(S_out, PolyZonotope), "Result should be a PolyZonotope"
        assert S_out.dim() == 2, "Dimension should be 2"
        assert not S_out.isemptyobject(), "Result should not be empty"
        
        # Verify center is valid (should be a column vector)
        assert S_out.c.shape == (2, 1), "Center should be a column vector of dimension 2"
        
        # Verify E and id dimensions match if E is non-empty
        if S_out.E.size > 0:
            assert S_out.E.shape[0] == S_out.id.shape[0], \
                "E and id should have matching number of rows"
            assert S_out.G.shape[1] == S_out.E.shape[1], \
                "G and E should have matching number of columns"
    
    def test_convHull_two_polyZonotopes(self):
        """TEST: Convex hull of two polynomial zonotopes"""
        # Create two polynomial zonotopes
        c1 = np.array([[-1], [0]])
        G1 = np.array([[1, 0], [0, 1]])
        GI1 = np.zeros((2, 0))
        E1 = np.array([[1, 0], [0, 1]])
        pZ1 = PolyZonotope(c1, G1, GI1, E1)
        
        c2 = np.array([[1], [0]])
        G2 = np.array([[1, 0], [0, 1]])
        GI2 = np.zeros((2, 0))
        E2 = np.array([[1, 0], [0, 1]])
        pZ2 = PolyZonotope(c2, G2, GI2, E2)
        
        # S_out = convHull(pZ1, pZ2);
        S_out = pZ1.convHull_(pZ2)
        
        # Verify basic properties
        assert isinstance(S_out, PolyZonotope), "Result should be a PolyZonotope"
        assert S_out.dim() == 2, "Dimension should be 2"
        assert not S_out.isemptyobject(), "Result should not be empty"
        
        # Verify center is valid
        assert S_out.c.shape == (2, 1), "Center should be a column vector of dimension 2"
    
    def test_convHull_with_interval(self):
        """TEST: Convex hull of a polyZonotope and an interval"""
        # Based on usage in testLong_polyZonotope_contains.m
        # I = interval([-1;-1],[1;1]);
        I = Interval(np.array([[-1], [-1]]), np.array([[1], [1]]))
        
        # p = polyZonotope([2;0]);
        p = PolyZonotope(np.array([[2], [0]]), np.zeros((2, 0)), np.zeros((2, 0)), np.zeros((0, 0), dtype=int))
        
        # S = convHull(polyZonotope(I), p);
        # First convert interval to polyZonotope
        pZ_I = PolyZonotope(I)
        S = pZ_I.convHull_(p)
        
        # Verify basic properties
        assert isinstance(S, PolyZonotope), "Result should be a PolyZonotope"
        assert S.dim() == 2, "Dimension should be 2"
        assert not S.isemptyobject(), "Result should not be empty"
    
    def test_convHull_with_empty(self):
        """TEST: Convex hull with empty set should return the original set"""
        c = np.array([[0], [0]])
        G = np.array([[1, 0], [0, 1]])
        GI = np.zeros((2, 0))
        E = np.array([[1, 3]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Create an empty polyZonotope
        pZ_empty = PolyZonotope.empty(2)
        
        # convHull with empty set should return the original set
        S_out = pZ.convHull_(pZ_empty)
        
        # Verify that the result is the same as the original (or at least has same dimension)
        assert isinstance(S_out, PolyZonotope), "Result should be a PolyZonotope"
        assert S_out.dim() == 2, "Dimension should be 2"
        # The result should be equivalent to pZ (convex hull with empty set is the set itself)
