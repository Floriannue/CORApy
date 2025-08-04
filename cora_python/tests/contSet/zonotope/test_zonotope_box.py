import pytest
import numpy as np
from cora_python.contSet.zonotope import Zonotope


def test_zonotope_box():
    """
    Test box method for zonotope - computes an enclosing axis-aligned box 
    in generator representation according to manual Appendix A.1.
    
    该测试严格对应MATLAB的test_zonotope_box.m。
    """
    
    # Set tolerance (same as MATLAB)
    tol = 1e-9
    
    # Test 2D zonotope - example from MATLAB test
    c = np.array([[1], [0]]) 
    G = np.array([[2, -1], [4, 1]])
    Z = Zonotope(c, G)
    Z_box = Z.box()
    
    # Expected result from MATLAB test
    # MATLAB: Ztrue = zonotope([1;0],[3 0; 0 5]);
    expected_c = np.array([[1], [0]])
    expected_G = np.array([[3, 0], [0, 5]])
    
    # Check if axis-aligned box same as expected
    assert np.allclose(Z_box.c, expected_c, atol=tol)
    assert np.allclose(Z_box.G, expected_G, atol=tol)
    
    # Test that result is axis-aligned (diagonal matrix)
    assert Z_box.representsa_('interval')


def test_zonotope_box_long():
    """
    Long test for box method - matches MATLAB testLong_zonotope_box.m logic.
    
    Box has to be the same as conversion to interval.
    """
    
    # Number of tests (same as MATLAB)
    nr_tests = 100
    
    for i in range(nr_tests):
        # Random dimension (1 to 20, same as MATLAB)
        n = np.random.randint(1, 21)
        
        # Create a random zonotope (同MATLAB: zonotope(-1+2*rand(n,nrOfGens+1)))
        nr_of_gens = 5 * n
        random_matrix = -1 + 2 * np.random.rand(n, nr_of_gens + 1)
        Z = Zonotope(random_matrix)
        
        # Compute axis-aligned box
        Z_box = Z.box()
        c_box = Z_box.center()
        G_box = Z_box.generators()
        
        # Convert to interval and back to zonotope
        Z_int = Zonotope(Z.interval())
        c_int = Z_int.center()
        G_int = Z_int.generators()
        
        # Check if axis-aligned box same as interval (same tolerance as MATLAB)
        assert np.allclose(c_box, c_int, atol=1e-14), f"Test {i} failed: centers don't match"
        assert np.allclose(G_box, G_int, atol=1e-14), f"Test {i} failed: generators don't match"


if __name__ == "__main__":
    test_zonotope_box()
    test_zonotope_box_long()
    print("All zonotope box tests passed!") 