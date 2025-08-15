import numpy as np
import pytest

from cora_python.contSet.polytope.polytope import Polytope
# Assuming ContSet.compact and ContSet.fullspace will be available or mocked for tests.
# from cora_python.contSet.contSet.compact import compact as contset_compact
# from cora_python.contSet.fullspace import fullspace
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def test_minkdiff_basic_exact():
    """Test basic Minkowski difference with 'exact' method for 2D polytopes."""
    # Define two simple 2D polytopes (squares)
    # P1: [-1,1]x[-1,1]
    A1 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b1 = np.array([[1], [1], [1], [1]])
    P1 = Polytope(A1, b1)

    # P2: [-0.5,0.5]x[-0.5,0.5]
    A2 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b2 = np.array([[0.5], [0.5], [0.5], [0.5]])
    P2 = Polytope(A2, b2)

    # Expected result: P1 - P2 = [-0.5,0.5]x[-0.5,0.5]
    # A = A1, b = b1 - supportFunc(P2, A1[i,:])
    # For a symmetric interval polytope P2, supportFunc(P2, dir) = max(abs(dir)) * radius
    # For A1[0,:] = [1,0], dir = [1,0], supportFunc = 0.5
    # So, expected b will be [1-0.5, 1-0.5, 1-0.5, 1-0.5] = [0.5,0.5,0.5,0.5]
    expected_b = np.array([[0.5], [0.5], [0.5], [0.5]]) # Corrected expected b

    P_out = P1.minkDiff(P2, 'exact')

    print(f"DEBUG (test_basic_exact): P_out type: {type(P_out)}")
    print(f"DEBUG (test_basic_exact): P_out._dim_val: {P_out._dim_val}")
    print(f"DEBUG (test_basic_exact): P_out.isemptyobject(): {P_out.isemptyobject()}")

    assert isinstance(P_out, Polytope)
    print(f"DEBUG (test_basic_exact): ID of P_out in test: {id(P_out)}")
    assert P_out.dim() == 2

    # Sort rows of A and b matrices before comparison to ensure order-invariant assertion
    # This handles cases where the order of constraints might differ but the polytope is the same.
    # Combine A and b for sorting, then split them again.
    P_out_combined = np.hstack((P_out.A, P_out.b))
    A1_combined = np.hstack((A1, expected_b)) # Use A1 and the corrected expected_b

    # Sort rows based on the first column, then second, and so on.
    P_out_sorted_indices = np.lexsort(P_out_combined.T[::-1])
    A1_sorted_indices = np.lexsort(A1_combined.T[::-1])

    P_out_A_sorted = P_out_combined[P_out_sorted_indices, :P_out.A.shape[1]]
    P_out_b_sorted = P_out_combined[P_out_sorted_indices, P_out.A.shape[1]:]

    A1_A_sorted = A1_combined[A1_sorted_indices, :A1.shape[1]]
    A1_b_sorted = A1_combined[A1_sorted_indices, A1.shape[1]:]

    assert np.allclose(P_out_A_sorted, A1_A_sorted)
    assert np.allclose(P_out_b_sorted, A1_b_sorted) # Expected b should also be sorted

    assert P_out.Ae.size == 0 and P_out.be.size == 0 # No equality constraints
    assert not P_out.isemptyobject()

def test_minkdiff_numeric_subtrahend():
    """Test Minkowski difference with a numeric subtrahend (point)."""
    # P1: [-1,1]x[-1,1]
    A1 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b1 = np.array([[1], [1], [1], [1]])
    P1 = Polytope(A1, b1)

    # S: point [0.5, 0.5]
    S = np.array([[0.5], [0.5]])

    # Expected result: P1 - [0.5, 0.5] = [-1.5, 0.5]x[-1.5, 0.5]
    # This should be equivalent to P1 + (-S)
    # new_b = b1 - S = [[1-0.5],[1-0.5],[1-0.5],[1-0.5]] = [[0.5],[0.5],[0.5],[0.5]] (Incorrect logic in comment, should be [1-(-0.5), 1-(-0.5), 1-(-0.5), 1-(-0.5)])
    # P1 + (-S): A1 * x <= b1 + S -> A1 * x <= b1 - (-S) for each row of A1
    # [1,0]*x <= 1, then x0 <= 1 - 0.5 = 0.5 -> [-inf, 0.5]
    # [-1,0]*x <= 1, then -x0 <= 1 - 0.5 = 0.5 -> [-0.5, inf]
    # [0,1]*x <= 1, then x1 <= 1 - 0.5 = 0.5 -> [-inf, 0.5]
    # [0,-1]*x <= 1, then -x1 <= 1 - 0.5 = 0.5 -> [-0.5, inf]
    # The operation P1 - S with S as a point implies translating the polytope.
    # So, it's P.A * (x + S) <= P.b --> P.A * x <= P.b - P.A * S
    # A1 = [[1,0], [-1,0], [0,1], [0,-1]]
    # S = [[0.5], [0.5]]
    # A1 * S = [[0.5], [-0.5], [0.5], [-0.5]]
    # b1 - (A1 * S) = [[1-0.5], [1-(-0.5)], [1-0.5], [1-(-0.5)]] = [[0.5], [1.5], [0.5], [1.5]]
    expected_b = np.array([[0.5], [1.5], [0.5], [1.5]]) # Corrected expected b

    P_out = P1.minkDiff(S)

    assert isinstance(P_out, Polytope)
    assert P_out.dim() == 2
    # Sort rows of A and b matrices before comparison to ensure order-invariant assertion
    P_out_combined = np.hstack((P_out.A, P_out.b))
    A1_combined = np.hstack((A1, expected_b)) # Use A1 and the corrected expected_b

    P_out_sorted_indices = np.lexsort(P_out_combined.T[::-1])
    A1_sorted_indices = np.lexsort(A1_combined.T[::-1])

    P_out_A_sorted = P_out_combined[P_out_sorted_indices, :P_out.A.shape[1]]
    P_out_b_sorted = P_out_combined[P_out_sorted_indices, P_out.A.shape[1]:]

    A1_A_sorted = A1_combined[A1_sorted_indices, :A1.shape[1]]
    A1_b_sorted = A1_combined[A1_sorted_indices, A1.shape[1]:]

    assert np.allclose(P_out_A_sorted, A1_A_sorted)
    assert np.allclose(P_out_b_sorted, A1_b_sorted)

    assert P_out.Ae.size == 0 and P_out.be.size == 0
    assert not P_out.isemptyobject()

def test_minkdiff_1d_polytopes():
    """Test Minkowski difference for 1D polytopes."""
    # P1: interval [0, 5]
    A1 = np.array([[1], [-1]])
    b1 = np.array([[5], [0]]) # x <= 5, -x <= 0 -> x >= 0
    P1 = Polytope(A1, b1)

    # P2: interval [1, 2]
    A2 = np.array([[1], [-1]])
    b2 = np.array([[2], [-1]]) # x <= 2, -x <= -1 -> x >= 1
    P2 = Polytope(A2, b2)

    # Expected P1 - P2 = [0-2, 5-1] = [-2, 4]
    # In 1D, A matrix is normalized, so A_out = A1
    # b_out = b1 - b2 = [[5-2],[0-(-1)]] = [[3],[1]]
    expected_b = np.array([[3], [1]])

    P_out = P1.minkDiff(P2)

    assert isinstance(P_out, Polytope)
    assert P_out.dim() == 1
    assert np.allclose(P_out.A, A1)
    assert np.allclose(P_out.b, expected_b)
    assert P_out.Ae.size == 0 and P_out.be.size == 0
    assert not P_out.isemptyobject()

def test_minkdiff_exact_vertices_not_implemented():
    """Test that 'exact:vertices' method raises CORAerror as not implemented."""
    A1 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b1 = np.array([[1], [1], [1], [1]])
    P1 = Polytope(A1, b1)

    A2 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b2 = np.array([[0.5], [0.5], [0.5], [0.5]])
    P2 = Polytope(A2, b2)

    with pytest.raises(CORAerror) as excinfo:
        P1.minkDiff(P2, 'exact:vertices')
    assert "minkDiff with type 'exact:vertices' is not yet translated." == str(excinfo.value)
