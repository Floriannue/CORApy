"""
testLong_zonotope_cartProd_ - unit test function of Cartesian product

This module performs extensive random testing of the cartProd_ method for zonotope objects.

Syntax:
    res = testLong_zonotope_cartProd_

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       03-January-2023 (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval


def testLong_zonotope_cartProd_():
    """Unit test function of cartProd_ - mirrors MATLAB testLong_zonotope_cartProd.m"""
    
    # number of tests
    nrTests = 1000
    res = True
    
    for i in range(nrTests):
        
        # Test 1: zonotope-zonotope case
        
        # random dimensions
        n1 = np.random.randint(1, 11)
        n2 = np.random.randint(1, 11)
        
        # random number of generators
        m1 = np.random.randint(1, n1 * 2 + 1)
        m2 = np.random.randint(1, n2 * 2 + 1)
        
        # random centers and generator matrices
        c1 = np.random.randn(n1, 1)
        c2 = np.random.randn(n2, 1)
        G1 = np.random.randn(n1, m1)
        G2 = np.random.randn(n2, m2)
        
        # instantiate zonotopes
        Z1 = Zonotope(c1, G1)
        Z2 = Zonotope(c2, G2)
        
        # compute Cartesian product
        Z_ = Z1.cartProd_(Z2)
        
        # obtain center and generator matrix
        c = Z_.c
        G = Z_.G
        
        # check result
        c_expected = np.vstack([c1, c2])
        G_expected = np.block([[G1, np.zeros((n1, m2))],
                               [np.zeros((n2, m1)), G2]])
        
        assert np.allclose(c, c_expected), f"Test {i}: Center mismatch in zonotope-zonotope case"
        assert np.allclose(G, G_expected), f"Test {i}: Generator mismatch in zonotope-zonotope case"
        
        # Test 2: zonotope-interval case
        
        # random dimensions
        n1 = np.random.randint(1, 11)
        n2 = np.random.randint(1, 11)
        
        # random number of generators
        m1 = np.random.randint(1, n1 * 2 + 1)
        
        # random center and generator matrix
        c1 = np.random.randn(n1, 1)
        G1 = np.random.randn(n1, m1)
        
        # random lower and upper bounds
        lb2 = -np.random.rand(n2, 1)
        ub2 = np.random.rand(n2, 1)
        
        # instantiate zonotope and interval
        Z1 = Zonotope(c1, G1)
        I2 = Interval(lb2, ub2)
        
        # compute Cartesian product
        Z_ = Z1.cartProd_(I2)
        
        # obtain center and generator matrix
        c = Z_.c
        G = Z_.G
        
        # check result
        c_expected = np.vstack([c1, 0.5 * (lb2 + ub2)])
        G_expected = np.block([[G1, np.zeros((n1, n2))],
                               [np.zeros((n2, m1)), 0.5 * np.diag((ub2 - lb2).flatten())]])
        
        assert np.allclose(c, c_expected), f"Test {i}: Center mismatch in zonotope-interval case"
        assert np.allclose(G, G_expected), f"Test {i}: Generator mismatch in zonotope-interval case"
        
        # Test 3: zonotope-numeric case
        
        # random dimensions
        n1 = np.random.randint(1, 11)
        n2 = np.random.randint(1, 11)
        
        # random number of generators
        m1 = np.random.randint(1, n1 * 2 + 1)
        
        # random center and generator matrix
        c1 = np.random.randn(n1, 1)
        G1 = np.random.randn(n1, m1)
        Z1 = Zonotope(c1, G1)
        
        # random numeric vector
        num = np.random.randn(n2, 1)
        
        # compute Cartesian product
        Z_ = Z1.cartProd_(num)
        
        # obtain center and generator matrix
        c = Z_.c
        G = Z_.G
        
        # check result
        c_expected = np.vstack([c1, num])
        G_expected = np.vstack([G1, np.zeros((n2, m1))])
        
        assert np.allclose(c, c_expected), f"Test {i}: Center mismatch in zonotope-numeric case"
        assert np.allclose(G, G_expected), f"Test {i}: Generator mismatch in zonotope-numeric case"
        
        # Test 4: numeric-zonotope case (using cartProd function)
        from cora_python.contSet.contSet import cartProd
        
        # compute Cartesian product
        Z_ = cartProd(num, Z1)
        
        # obtain center and generator matrix
        c = Z_.c
        G = Z_.G
        
        # check result
        c_expected = np.vstack([num, c1])
        G_expected = np.vstack([np.zeros((n2, m1)), G1])
        
        assert np.allclose(c, c_expected), f"Test {i}: Center mismatch in numeric-zonotope case"
        assert np.allclose(G, G_expected), f"Test {i}: Generator mismatch in numeric-zonotope case"
    
    # Test completed
    return res


def testLong_zonotope_cartProd_edge_cases():
    """Test edge cases with random data"""
    
    # Test with zero generators
    for i in range(100):
        n1 = np.random.randint(1, 6)
        n2 = np.random.randint(1, 6)
        
        c1 = np.random.randn(n1, 1)
        c2 = np.random.randn(n2, 1)
        
        # Z1 with no generators
        Z1 = Zonotope(c1, np.array([]).reshape(n1, 0))
        Z2 = Zonotope(c2, np.random.randn(n2, np.random.randint(1, 4)))
        
        Z_ = Z1.cartProd_(Z2)
        
        assert Z_.dim() == n1 + n2
        assert Z_.G.shape[1] == Z2.G.shape[1]
        
        # Z2 with no generators
        Z1 = Zonotope(c1, np.random.randn(n1, np.random.randint(1, 4)))
        Z2 = Zonotope(c2, np.array([]).reshape(n2, 0))
        
        Z_ = Z1.cartProd_(Z2)
        
        assert Z_.dim() == n1 + n2
        assert Z_.G.shape[1] == Z1.G.shape[1]
    
    # Test with scalar values
    for i in range(100):
        n1 = np.random.randint(1, 6)
        c1 = np.random.randn(n1, 1)
        G1 = np.random.randn(n1, np.random.randint(1, 4))
        Z1 = Zonotope(c1, G1)
        
        scalar = np.random.randn()
        Z_ = Z1.cartProd_(scalar)
        
        assert Z_.dim() == n1 + 1
        assert np.allclose(Z_.c[-1], scalar)
        assert Z_.G.shape[1] == G1.shape[1]


if __name__ == "__main__":
    pytest.main([__file__]) 