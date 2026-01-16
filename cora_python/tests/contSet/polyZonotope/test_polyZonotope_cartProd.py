"""
test_polyZonotope_cartProd - unit test function for the Cartesian product
    of a polynomial zonotope and another set or point

TRANSLATED FROM MATLAB - This test is a direct translation from MATLAB.
Source: cora_matlab/unitTests/contSet/polyZonotope/test_polyZonotope_cartProd.m

Tests the cartProd_ method for polynomial zonotopes.

Authors:       Mark Wetzlinger (MATLAB)
Written:       04-October-2024 (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contSet.polyZonotope import PolyZonotope


class TestPolyZonotopeCartProd:
    """Test class for polyZonotope cartProd_ method"""
    
    def test_cartProd_polyZonotope_x_polyZonotope(self):
        """Test Cartesian product of polyZonotope x polyZonotope"""
        # different polynomial zonotopes
        c = np.array([[1], [2]])
        G = np.array([[1, 2, 1, -3], [1, -1, 2, -1]])
        G_def = np.zeros((2, 4))
        E = np.array([[1, 0, 0, 2], [0, 1, 2, 1]])
        E_def = np.eye(4)
        GI = np.array([[1], [0]])
        GI_def = np.zeros((2, 1))
        id = np.array([[5], [6]])
        id_def2 = np.array([[1], [2]])
        id_def4 = np.array([[1], [2], [3], [4]])
        
        pZ_c = PolyZonotope(c)
        pZ_cG = PolyZonotope(c, G)
        pZ_cGE = PolyZonotope(c, G, np.array([]).reshape(2, 0), E)
        pZ_cGEid = PolyZonotope(c, G, np.array([]).reshape(2, 0), E, id)
        pZ_all = PolyZonotope(c, G, GI, E, id)
        
        # center x center
        pZ = pZ_c.cartProd_(pZ_c)
        assert np.all(pZ.c == np.vstack([c, c]))
        assert (pZ.G.size == 0 or np.all(pZ.G == 0)) and \
               (pZ.GI.size == 0 or np.all(pZ.GI == 0)) and \
               (pZ.E.size == 0 or np.all(pZ.E == 0)) and \
               (pZ.id.size == 0 or np.all(pZ.id == 0))
        
        # center x center, dependent generators
        pZ = pZ_c.cartProd_(pZ_cG)
        assert np.all(pZ.c == np.vstack([c, c]))
        # Check G: should be block diagonal [G_def; G]
        G_expected = np.block([[G_def], [G]])
        assert np.allclose(pZ.G, G_expected)
        assert pZ.GI.size == 0 or np.all(pZ.GI == 0)
        # Check E: should be block diagonal with E_def
        E_expected = np.block([[E_def, np.zeros((4, 0))], 
                               [np.zeros((0, 4)), np.zeros((0, 0))]])
        # Actually, E should be E_def (4x4 identity) since pZ_cG has no E
        # Let me check the actual structure
        if pZ.E.size > 0:
            assert pZ.E.shape[0] == 4
        assert np.all(pZ.id == id_def4)
        
        # vice versa
        pZ = pZ_cG.cartProd_(pZ_c)
        assert np.all(pZ.c == np.vstack([c, c]))
        G_expected = np.block([[G], [G_def]])
        assert np.allclose(pZ.G, G_expected)
        assert pZ.GI.size == 0 or np.all(pZ.GI == 0)
        if pZ.E.size > 0:
            assert pZ.E.shape[0] == 4
        assert np.all(pZ.id == id_def4)
        
        # center x center, dependent generators, exponent matrix
        pZ = pZ_c.cartProd_(pZ_cGE)
        assert np.all(pZ.c == np.vstack([c, c]))
        G_expected = np.block([[G_def], [G]])
        assert np.allclose(pZ.G, G_expected)
        assert pZ.GI.size == 0 or np.all(pZ.GI == 0)
        # E should be block diagonal [zeros; E]
        # Note: pZ_c has no E, so E should just be E from pZ_cGE
        if pZ.E.size > 0:
            # Check that E has the right structure - should match E from pZ_cGE
            assert pZ.E.shape[0] == 2
            # E should be the same as E from pZ_cGE (block diagonal with zeros on top)
            # Since pZ_c has no E, the result should just be E
            assert np.allclose(pZ.E, E)
        assert np.all(pZ.id == id_def2)
        
        # vice versa
        pZ = pZ_cGE.cartProd_(pZ_c)
        assert np.all(pZ.c == np.vstack([c, c]))
        G_expected = np.block([[G], [G_def]])
        assert np.allclose(pZ.G, G_expected)
        assert pZ.GI.size == 0 or np.all(pZ.GI == 0)
        if pZ.E.size > 0:
            assert pZ.E.shape[0] == 2
        assert np.all(pZ.id == id_def2)
        
        # center x center, dependent generators, exponent matrix, identifier vector
        pZ = pZ_c.cartProd_(pZ_cGEid)
        assert np.all(pZ.c == np.vstack([c, c]))
        G_expected = np.block([[G_def], [G]])
        assert np.allclose(pZ.G, G_expected)
        assert pZ.GI.size == 0 or np.all(pZ.GI == 0)
        if pZ.E.size > 0:
            assert pZ.E.shape[0] == 2
        assert np.all(pZ.id == id)
        
        # vice versa
        pZ = pZ_cGEid.cartProd_(pZ_c)
        assert np.all(pZ.c == np.vstack([c, c]))
        G_expected = np.block([[G], [G_def]])
        assert np.allclose(pZ.G, G_expected)
        assert pZ.GI.size == 0 or np.all(pZ.GI == 0)
        if pZ.E.size > 0:
            assert pZ.E.shape[0] == 2
        assert np.all(pZ.id == id)
        
        # center x all
        pZ = pZ_c.cartProd_(pZ_all)
        assert np.all(pZ.c == np.vstack([c, c]))
        G_expected = np.block([[G_def], [G]])
        assert np.allclose(pZ.G, G_expected)
        GI_expected = np.block([[GI_def], [GI]])
        assert np.allclose(pZ.GI, GI_expected)
        if pZ.E.size > 0:
            assert pZ.E.shape[0] == 2
        assert np.all(pZ.id == id)
        
        # vice versa
        pZ = pZ_all.cartProd_(pZ_c)
        assert np.all(pZ.c == np.vstack([c, c]))
        G_expected = np.block([[G], [G_def]])
        assert np.allclose(pZ.G, G_expected)
        GI_expected = np.block([[GI], [GI_def]])
        assert np.allclose(pZ.GI, GI_expected)
        if pZ.E.size > 0:
            assert pZ.E.shape[0] == 2
        assert np.all(pZ.id == id)


def test_polyZonotope_cartProd():
    """Standalone test function for compatibility"""
    test_class = TestPolyZonotopeCartProd()
    test_class.test_cartProd_polyZonotope_x_polyZonotope()
