"""
test_polyZonotope_quadMap - unit test function of quadMap

TRANSLATED FROM MATLAB - This test is a direct translation from MATLAB.
Source: cora_matlab/unitTests/contSet/polyZonotope/test_polyZonotope_quadMap.m

Tests the quadMap method for polynomial zonotopes.

Authors:       Niklas Kochdumper (MATLAB)
Written:       23-March-2018 (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contSet.polyZonotope import PolyZonotope
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


class TestPolyZonotopeQuadMap:
    """Test class for polyZonotope quadMap method"""
    
    def test_quadMap_basic(self):
        """Test basic quadMap functionality"""
        # instantiate polynomial zonotope
        c = np.array([[1], [2]])
        G = np.array([[1, -2, 1], [2, 3, -1]])
        GI = np.array([[0], [0]])
        E = np.array([[1, 0, 2], [0, 1, 1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # create matrices of the quadratic map
        Q = [np.array([[1, 2], [-1, 2]]), np.array([[-3, 0], [1, 1]])]
        
        # calculate quadratic map
        pZres = pZ.quadMap(Q)
        
        # define ground truth
        G_expected = np.array([[22, 19, -5, 11, 19, -5, 16, -11, 2], 
                              [6, 23, -9, 3, 23, -9, -9, 11, -3]])
        c_expected = np.array([[11], [3]])
        E_expected = np.array([[1, 0, 2, 2, 1, 3, 0, 2, 4], 
                              [0, 1, 1, 0, 1, 1, 2, 2, 2]])
        
        # check for correctness
        assert np.all(withinTol(pZres.c, c_expected))
        
        # Check that each column of E_expected exists in pZres.E
        # MATLAB: for i = 1:size(E,2)
        # MATLAB:     ind = ismember(pZres.E',E(:,i)','rows');
        # MATLAB:     ind_ = find(ind > 0);
        # MATLAB:     assertLoop(~isempty(ind_),i);
        # MATLAB:     assertLoop(all(pZres.G(:,ind_(1)) == G(:,i)),i)
        for i in range(E_expected.shape[1]):
            E_col = E_expected[:, i]
            # Find matching columns in pZres.E (transpose for row comparison)
            if pZres.E.size > 0:
                # Check if this column exists in pZres.E
                matches = np.all(pZres.E.T == E_col, axis=1)
                ind_ = np.where(matches)[0]
                
                assert len(ind_) > 0, f"Column {i} of E_expected not found in pZres.E"
                # Check that the corresponding generator matches
                assert np.allclose(pZres.G[:, ind_[0]], G_expected[:, i]), \
                    f"Generator for column {i} does not match"


def test_polyZonotope_quadMap():
    """Standalone test function for compatibility"""
    test_class = TestPolyZonotopeQuadMap()
    test_class.test_quadMap_basic()
