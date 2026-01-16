"""
test_polyZonotope_exactPlus - unit test function for the exact addition of two
    polynomial zonotope objects

TRANSLATED FROM MATLAB - This test is a direct translation from MATLAB.
Source: cora_matlab/unitTests/contSet/polyZonotope/test_polyZonotope_plus.m

Tests the exactPlus method for polynomial zonotopes.

Authors:       Niklas Kochdumper (MATLAB)
Written:       26-June-2018 (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contSet.polyZonotope import PolyZonotope
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


class TestPolyZonotopeExactPlus:
    """Test class for polyZonotope exactPlus method"""
    
    def test_exactPlus_basic(self):
        """TEST: Basic exact addition of two polynomial zonotopes"""
        # create polynomial zonotopes
        c1 = np.array([[-1], [3]])
        G1 = np.array([[-1, -1, 3, 2], [2, -1, -1, 0]])
        E1 = np.array([[1, 0, 1, 5], [0, 1, 3, 0]])
        GI1 = np.zeros((2, 0))
        pZ1 = PolyZonotope(c1, G1, GI1, E1)
        
        c2 = np.array([[0], [2]])
        G2 = np.array([[-2, -2, -1], [-1, -2, -3]])
        E2 = np.array([[1, 0, 2], [0, 0, 0], [0, 1, 3]])
        GI2 = np.zeros((2, 0))
        pZ2 = PolyZonotope(c2, G2, GI2, E2)
        
        # exact addition of the two polynomial zonotopes
        pZres = pZ1.exactPlus(pZ2)
        
        # define ground truth
        c_expected = np.array([[-1], [5]])
        G_expected = np.array([[-3, -1, 3, 2, -2, -1], [1, -1, -1, 0, -2, -3]])
        E_expected = np.array([[1, 0, 1, 5, 0, 2], [0, 1, 3, 0, 0, 0], [0, 0, 0, 0, 1, 3]])
        
        # check for correctness
        assert np.all(withinTol(c_expected, pZres.c))
        
        # Check that each column in E_expected exists in pZres.E
        for i in range(E_expected.shape[1]):
            # Find matching column in pZres.E
            E_col = E_expected[:, i].reshape(-1, 1)
            matches = np.all(pZres.E == E_col, axis=0)
            ind_ = np.where(matches)[0]
            
            assert len(ind_) > 0, f"Column {i} from E_expected not found in pZres.E"
            # Check that the corresponding generator matches
            assert np.allclose(pZres.G[:, ind_[0]], G_expected[:, i]), \
                f"Generator for column {i} does not match"
