import numpy as np
import pytest
from cora_python.g.classes.intKaucher import IntKaucher

def test_intkaucher_init():
    # Test scalar initialization
    ik = IntKaucher(1, 2)
    assert ik.inf == 1
    assert ik.sup == 2

    # Test array initialization
    inf = np.array([[1, 2], [3, 4]])
    sup = np.array([[5, 6], [7, 8]])
    ik = IntKaucher(inf, sup)
    np.testing.assert_array_equal(ik.inf, inf)
    np.testing.assert_array_equal(ik.sup, sup)

    # Test mismatched dimensions
    with pytest.raises(ValueError):
        IntKaucher(np.array([1, 2]), np.array([1, 2, 3]))

def test_intkaucher_add():
    ik1 = IntKaucher(1, 2)
    ik2 = IntKaucher(3, 4)
    res = ik1 + ik2
    assert res.inf == 4
    assert res.sup == 6

    # Add with scalar
    res = ik1 + 5
    assert res.inf == 6
    assert res.sup == 7
    
    # Radd
    res = 5 + ik1
    assert res.inf == 6
    assert res.sup == 7

def test_intkaucher_neg_sub():
    ik1 = IntKaucher(1, 2)
    neg_ik1 = -ik1
    assert neg_ik1.inf == -2
    assert neg_ik1.sup == -1

    ik2 = IntKaucher(3, 4)
    res = ik1 - ik2
    assert res.inf == -3
    assert res.sup == -2

def test_intkaucher_mul():
    # Test case from MATLAB doc: P * DualZ
    ik1 = IntKaucher(np.array([1]), np.array([2])) # in P
    ik2 = IntKaucher(np.array([4]), np.array([3])) # in DualZ
    res = ik1 * ik2
    assert res.inf == 1 * 4
    assert res.sup == 1 * 3
    
    # Test case: -P * DualZ
    ik1 = IntKaucher(np.array([-2]), np.array([-1])) # in -P
    res = ik1 * ik2
    assert res.inf == -1 * 3
    assert res.sup == -1 * 4

    # Test case: Z * DualZ
    ik1 = IntKaucher(np.array([-1]), np.array([1])) # in Z
    res = ik1 * ik2
    assert res.inf == 0
    assert res.sup == 0

    # Test with scalar
    ik = IntKaucher(2, 3)
    res = ik * 2
    assert res.inf == 4
    assert res.sup == 6
    res = ik * -2
    assert res.inf == -3 * 2 # (-ik) * 2 -> (-3, -2) * 2
    assert res.sup == -2 * 2

def test_intkaucher_matmul():
    # Scalar matmul
    ik1 = IntKaucher(1, 2)
    ik2 = IntKaucher(4, 3)
    res = ik1 @ ik2
    assert res.inf == 4
    assert res.sup == 3

    # Matrix matmul
    inf1 = np.array([[1, 2], [3, 4]])
    sup1 = np.array([[1, 2], [3, 4]])
    ik_mat1 = IntKaucher(inf1, sup1) # in P

    inf2 = np.array([[6, 8], [10, 12]])
    sup2 = np.array([[5, 7], [9, 11]])
    ik_mat2 = IntKaucher(inf2, sup2) # in DualZ

    res_mat = ik_mat1 @ ik_mat2
    
    # Manual calculation for one element res_mat[0,0]
    # (ik11*ik_a11) + (ik12*ik_a21)
    # (1,1)*(6,5) + (2,2)*(10,9) = (6,5) + (20,18) = (26, 23)
    expected_inf = np.array([[26, 38], [58, 86]])
    expected_sup = np.array([[23, 33], [51, 75]])

    np.testing.assert_array_equal(res_mat.inf, expected_inf)
    np.testing.assert_array_equal(res_mat.sup, expected_sup)

def test_intkaucher_additional_methods():
    ik = IntKaucher(3, 2) # Improper
    
    # is_prop and prop
    assert not ik.is_prop()
    prop_ik = ik.prop()
    assert prop_ik.is_prop()
    assert prop_ik.inf == 2
    assert prop_ik.sup == 3

    # to_interval
    from cora_python.contSet.interval.interval import interval
    iv = ik.to_interval()
    assert isinstance(iv, interval)
    assert iv.inf == 3
    assert iv.sup == 2

    # isscalar and shape
    assert IntKaucher(1,1).isscalar()
    assert not IntKaucher([1,2],[1,2]).isscalar()
    assert IntKaucher([1,2],[1,2]).shape() == (2,)
    
def test_intkaucher_display():
    ik = IntKaucher(1.23456, 2.34567)
    assert str(ik) == "[1.23457, 2.34567]"

    ik_mat = IntKaucher(np.eye(2), np.ones((2,2))*3)
    disp_str = str(ik_mat)
    expected_str = (
        "IntKaucher:\n"
        "[1.00000, 3.00000] [0.00000, 3.00000]\n"
        "[0.00000, 3.00000] [1.00000, 3.00000]\n"
    )
    assert disp_str == expected_str 