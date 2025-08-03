"""
test_zonotope_split - unit test function of split

Syntax:
    res = test_zonotope_split

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       26-July-2016 (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonotope.split import split
from cora_python.contSet.polytope import Polytope


def test_zonotope_split():
    """
    Test the split method for zonotopes with various input scenarios.
    
    The test verifies that the split method produces the expected results
    for different types of splits: all dimensions, specific dimension,
    direction split, and halfspace split.
    """
    # Tolerance for numerical comparisons
    tol = 1e-14
    
    # Create zonotope
    Z1 = Zonotope(np.array([[-4, -3, -2, -1], [1, 2, 3, 4]]))
    
    # Create halfspace
    hs = Polytope(np.array([[1, -1]]), np.array([[-2]]))
    
    # Test 1: Split all dimensions
    Zsplit_1 = split(Z1)
    
    # Test 2: Split specific dimension (dimension 2, which is index 1 in 0-based indexing)
    Zsplit_2 = split(Z1, 1)
    
    # Test 3: Split in direction [1; 1]
    Zsplit_3 = split(Z1, np.array([[1], [1]]))
    
    # Test 4: Split according to halfspace
    Zsplit_4 = split(Z1, hs)
    
    # Extract centers and generators for split 1 (all dimensions)
    c_1 = []
    G_1 = []
    for i in range(len(Zsplit_1)):
        c_1.append([])
        G_1.append([])
        for j in range(len(Zsplit_1[i])):
            c_1[i].append(Zsplit_1[i][j].c)
            G_1[i].append(Zsplit_1[i][j].G)
    
    # Extract centers and generators for split 2 (specific dimension)
    c_2 = []
    G_2 = []
    for i in range(len(Zsplit_2)):
        c_2.append(Zsplit_2[i].c)
        G_2.append(Zsplit_2[i].G)
    
    # Extract centers and generators for split 3 (direction)
    c_3 = []
    G_3 = []
    for i in range(len(Zsplit_3)):
        c_3.append(Zsplit_3[i].c)
        G_3.append(Zsplit_3[i].G)
    
    # Extract centers and generators for split 4 (halfspace)
    c_4 = []
    G_4 = []
    for i in range(len(Zsplit_4)):
        c_4.append(Zsplit_4[i].c)
        G_4.append(Zsplit_4[i].G)
    
    # Expected results for split 1 (all dimensions)
    true_c_1 = [
        [np.array([[-7], [1]]), np.array([[-1], [1]])],
        [np.array([[-4], [-3.5]]), np.array([[-4], [5.5]])]
    ]
    true_G_1 = [
        [np.array([[3, 0], [0, 9]]), np.array([[3, 0], [0, 9]])],
        [np.array([[6, 0], [0, 4.5]]), np.array([[6, 0], [0, 4.5]])]
    ]
    
    # Expected results for split 2 (specific dimension)
    true_c_2 = [
        np.array([[-4], [-3.5]]),
        np.array([[-4], [5.5]])
    ]
    true_G_2 = [
        np.array([[6, 0], [0, 4.5]]),
        np.array([[6, 0], [0, 4.5]])
    ]
    
    # Expected results for split 3 (direction)
    true_c_3 = [
        np.array([[-5.25], [-0.25]]),
        np.array([[-2.75], [2.25]])
    ]
    true_G_3 = [
        np.array([[1.25, -2.5, -2.5, -2.5], [1.25, 2.5, 2.5, 2.5]]),
        np.array([[-1.25, -2.5, -2.5, -2.5], [-1.25, 2.5, 2.5, 2.5]])
    ]
    
    # Expected results for split 4 (halfspace)
    true_c_4 = [
        np.array([[-7], [4]]),
        np.array([[0.5], [-3.5]])
    ]
    true_G_4 = [
        np.array([[4.5, -0.5, 0.5, 1.5], [-4.5, -0.5, 0.5, 1.5]]),
        np.array([[-3, -0.5, 0.5, 1.5], [3, -0.5, 0.5, 1.5]])
    ]
    
    # Check results for split 1
    res_1 = True
    for i in range(len(G_1)):
        for j in range(len(G_1[i])):
            res_1 = res_1 and compare_matrices(true_c_1[i][j], c_1[i][j], tol) \
                and compare_matrices(true_G_1[i][j], G_1[i][j], tol)
    
    # Check results for split 2
    res_2 = True
    for i in range(len(G_2)):
        res_2 = res_2 and compare_matrices(true_c_2[i], c_2[i], tol) \
            and compare_matrices(true_G_2[i], G_2[i], tol)
    
    # Check results for split 3
    res_3 = True
    for i in range(len(G_3)):
        res_3 = res_3 and compare_matrices(true_c_3[i], c_3[i], tol) \
            and compare_matrices(true_G_3[i], G_3[i], tol)
    
    # Check results for split 4
    res_4 = True
    for i in range(len(G_4)):
        res_4 = res_4 and compare_matrices(true_c_4[i], c_4[i], tol) \
            and compare_matrices(true_G_4[i], G_4[i], tol)
    
    # Combined check
    assert res_1 and res_2 and res_3 and res_4, \
        f"Split test failed: res_1={res_1}, res_2={res_2}, res_3={res_3}, res_4={res_4}"


def compare_matrices(A, B, tol):
    """
    Compare two matrices with tolerance.
    
    Args:
        A: First matrix
        B: Second matrix
        tol: Tolerance for comparison
        
    Returns:
        True if matrices are equal within tolerance, False otherwise
    """
    if A.shape != B.shape:
        return False
    
    return np.allclose(A, B, atol=tol, rtol=tol)


def test_zonotope_split_edge_cases():
    """
    Test edge cases for the split method.
    """
    # Test with None inputs
    Z = Zonotope(np.array([[1], [0]]), np.array([[1, 0], [0, 1]]))
    
    # Test with invalid dimension
    with pytest.raises(Exception):
        split(Z, 10)  # Invalid dimension
    
    # Test with invalid number of arguments
    with pytest.raises(Exception):
        split(Z, 1, 2, 3, 4)  # Too many arguments


def test_zonotope_split_bundle():
    """
    Test the bundle split functionality.
    """
    Z = Zonotope(np.array([[1], [0]]), np.array([[1, 0], [0, 1]]))
    dir_vec = np.array([[1], [1]])
    
    # Test bundle split
    Zsplit_bundle = split(Z, dir_vec, 'bundle')
    
    # Check that we get a list of lists (bundle structure)
    assert isinstance(Zsplit_bundle, list)
    assert len(Zsplit_bundle) == 2
    assert isinstance(Zsplit_bundle[0], list)
    assert isinstance(Zsplit_bundle[1], list)


def test_zonotope_split_with_reduction():
    """
    Test split with reduction.
    """
    Z = Zonotope(np.array([[1], [0]]), np.array([[1, 0, 0.5], [0, 1, 0.5]]))
    
    # Test split with reduction
    Zsplit_reduced = split(Z, 0, 2)  # Split dimension 0 with maxOrder 2
    
    # Check that we get a list of zonotopes
    assert isinstance(Zsplit_reduced, list)
    assert len(Zsplit_reduced) == 2
    assert all(isinstance(z, Zonotope) for z in Zsplit_reduced)


def test_zonotope_split_perpendicular():
    """
    Test split in perpendicular direction.
    """
    Z = Zonotope(np.array([[1], [0]]), np.array([[1, 0], [0, 1]]))
    orig_dir = np.array([[1], [0]])
    aux_dir = np.array([[0], [1]])
    
    # Test perpendicular split
    Zsplit_perp = split(Z, orig_dir, aux_dir)
    
    # Check that we get a list of zonotopes
    assert isinstance(Zsplit_perp, list)
    assert len(Zsplit_perp) == 2
    assert all(isinstance(z, Zonotope) for z in Zsplit_perp)


if __name__ == "__main__":
    # Run all tests
    test_zonotope_split()
    test_zonotope_split_edge_cases()
    test_zonotope_split_bundle()
    test_zonotope_split_with_reduction()
    test_zonotope_split_perpendicular()
    print("All zonotope split tests passed!") 