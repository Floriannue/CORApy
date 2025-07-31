"""
testLong_zonotope_ellipsoid - unit test function of ellipsoid
    (and implicitly of MVEE, insc_ellipsoid, enc_ellipsoid, MVEE, MVIE)

Syntax:
    res = testLong_zonotope_ellipsoid

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: insc_ellipsoid,enc_ellipsoid,MVEE,MVIE

Authors:       Victor Gassmann, Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       11-October-2019 (MATLAB)
Last update:   06-June-2021 (MA, degenerate case added) (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


def testLong_zonotope_ellipsoid():
    """Test zonotope ellipsoid conversion with comprehensive test cases."""
    
    # do small dimensions if gurobi is not installed => norm takes much longer
    # w/o gurobi
    dims = [2, 4]  # 2:3:4 in MATLAB
    dGen = 5
    steps = 2
    rndDirs = 100
    
    for i in dims:
        n = i
        for j in range(1, steps + 1):
            m = n + j * dGen
            
            # random zonotopes
            # normal case
            c = np.random.randn(n, 1)
            G = np.random.randn(n, m)
            Z = Zonotope(c, G)
            
            # degenerate case
            T = np.random.randn(n + 1, n)
            Z_deg = T @ Z
            
            # overapproximations
            # normal case
            Eo_exact = Z.ellipsoid('outer:exact')
            Eo_n = Z.ellipsoid('outer:norm')
            Eo_nb = Z.ellipsoid('outer:norm_bnd')
            
            # degenerate case
            Eo_exact_deg = Z_deg.ellipsoid('outer:exact')
            Eo_n_deg = Z_deg.ellipsoid('outer:norm')
            Eo_nb_deg = Z_deg.ellipsoid('outer:norm_bnd')
            
            # underapproximations
            # normal case
            Eu_exact = Z.ellipsoid('inner:exact')
            Eu_n = Z.ellipsoid('inner:norm')
            # Eu_nb = Z.ellipsoid('inner:norm_bnd')  # Not implemented in MATLAB
            
            # degenerate case
            Eu_exact_deg = Z_deg.ellipsoid('inner:exact')
            Eu_n_deg = Z_deg.ellipsoid('inner:norm')
            # Eu_nb_deg = Z_deg.ellipsoid('inner:norm_bnd')  # Not implemented in MATLAB
            
            # compute rndDirs random unit directions
            # normal case
            cc = np.random.randn(n, rndDirs)
            nn = cc / np.sqrt(np.sum(cc**2, axis=0, keepdims=True))
            
            # degenerate case
            nn_deg = T @ nn
            
            # check if suppfnc(E)>=suppfnc(Z) (Z in E)
            for k in range(rndDirs):
                # overapproximations
                # normal case
                assert Z.supportFunc_(nn[:, k:k+1]) <= Eo_exact.supportFunc_(nn[:, k:k+1]), \
                    f"Overapproximation exact failed for dim={i}, step={j}, dir={k}"
                assert Z.supportFunc_(nn[:, k:k+1]) <= Eo_n.supportFunc_(nn[:, k:k+1]), \
                    f"Overapproximation norm failed for dim={i}, step={j}, dir={k}"
                assert Z.supportFunc_(nn[:, k:k+1]) <= Eo_nb.supportFunc_(nn[:, k:k+1]), \
                    f"Overapproximation norm_bnd failed for dim={i}, step={j}, dir={k}"
                
                # degenerate case
                assert Z_deg.supportFunc_(nn_deg[:, k:k+1]) <= Eo_exact_deg.supportFunc_(nn_deg[:, k:k+1]), \
                    f"Overapproximation exact degenerate failed for dim={i}, step={j}, dir={k}"
                assert Z_deg.supportFunc_(nn_deg[:, k:k+1]) <= Eo_n_deg.supportFunc_(nn_deg[:, k:k+1]), \
                    f"Overapproximation norm degenerate failed for dim={i}, step={j}, dir={k}"
                assert Z_deg.supportFunc_(nn_deg[:, k:k+1]) <= Eo_nb_deg.supportFunc_(nn_deg[:, k:k+1]), \
                    f"Overapproximation norm_bnd degenerate failed for dim={i}, step={j}, dir={k}"
                
                # underapproximations
                # normal case
                assert Eu_exact.supportFunc_(nn[:, k:k+1]) <= Z.supportFunc_(nn[:, k:k+1]), \
                    f"Underapproximation exact failed for dim={i}, step={j}, dir={k}"
                assert Eu_n.supportFunc_(nn[:, k:k+1]) <= Z.supportFunc_(nn[:, k:k+1]), \
                    f"Underapproximation norm failed for dim={i}, step={j}, dir={k}"
                
                # degenerate case
                assert Eu_exact_deg.supportFunc_(nn_deg[:, k:k+1]) <= Z_deg.supportFunc_(nn_deg[:, k:k+1]), \
                    f"Underapproximation exact degenerate failed for dim={i}, step={j}, dir={k}"
                assert Eu_n_deg.supportFunc_(nn_deg[:, k:k+1]) <= Z_deg.supportFunc_(nn_deg[:, k:k+1]), \
                    f"Underapproximation norm degenerate failed for dim={i}, step={j}, dir={k}"
    
    # gather results
    res = True
    assert res


def test_zonotope_ellipsoid_edge_cases():
    """Test edge cases for zonotope ellipsoid conversion."""
    
    # Test point zonotope
    c = np.array([[1], [2], [3]])
    Z_point = Zonotope(c)
    E_point = Z_point.ellipsoid()
    
    assert isinstance(E_point, Ellipsoid)
    np.testing.assert_array_almost_equal(E_point.q, c)
    np.testing.assert_array_almost_equal(E_point.Q, np.zeros((3, 3)))
    
    # Test parallelotope (square generator matrix)
    c = np.array([[0], [0]])
    G = np.array([[1, 0], [0, 1]])
    Z_para = Zonotope(c, G)
    E_para = Z_para.ellipsoid('outer:norm_bnd')
    
    assert isinstance(E_para, Ellipsoid)
    # For parallelotope, should get n * (G * G.T) as shape matrix
    expected_Q = 2 * (G @ G.T)  # n=2
    np.testing.assert_array_almost_equal(E_para.Q, expected_Q)
    
    # Test degenerate zonotope
    c = np.array([[0], [0]])
    G = np.array([[1, 1], [1, 1]])  # Rank 1 matrix
    Z_degen = Zonotope(c, G)
    E_degen = Z_degen.ellipsoid('outer:norm_bnd')
    
    assert isinstance(E_degen, Ellipsoid)
    
    # Test all modes
    modes = ['outer:exact', 'outer:norm', 'outer:norm_bnd', 'inner:exact', 'inner:norm']
    c = np.array([[0], [0]])
    G = np.array([[1, 2, 3], [4, 5, 6]])
    Z = Zonotope(c, G)
    
    for mode in modes:
        E = Z.ellipsoid(mode)
        assert isinstance(E, Ellipsoid)
        assert E.Q.shape == (2, 2)
        assert E.q.shape == (2, 1)


def test_zonotope_ellipsoid_invalid_modes():
    """Test that invalid modes raise appropriate errors."""
    
    c = np.array([[0], [0]])
    G = np.array([[1, 2], [3, 4]])
    Z = Zonotope(c, G)
    
    # Test invalid mode
    with pytest.raises(Exception):  # CORAerror
        Z.ellipsoid('invalid_mode')
    
    # Test inner:norm_bnd (not implemented in MATLAB)
    with pytest.raises(Exception):  # CORAerror
        Z.ellipsoid('inner:norm_bnd')


def test_zonotope_ellipsoid_containment_properties():
    """Test that ellipsoids properly contain/are contained in zonotopes."""
    
    # Test outer approximations contain the zonotope
    c = np.array([[0], [0]])
    G = np.array([[1, 2], [3, 4]])
    Z = Zonotope(c, G)
    
    # Test multiple random directions
    for _ in range(10):
        direction = np.random.randn(2, 1)
        direction = direction / np.linalg.norm(direction)
        
        # Outer approximations should have larger support function
        E_outer = Z.ellipsoid('outer:norm_bnd')
        assert E_outer.supportFunc_(direction) >= Z.supportFunc_(direction)
        
        # Inner approximations should have smaller support function
        E_inner = Z.ellipsoid('inner:norm')
        assert E_inner.supportFunc_(direction) <= Z.supportFunc_(direction) 