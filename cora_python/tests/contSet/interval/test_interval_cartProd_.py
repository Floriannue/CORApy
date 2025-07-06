import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval

def test_interval_cartProd():
    # 1. vertcat
    I1 = Interval(np.array([[-2], [-4], [-3]]), np.array([[2], [6], [1]]))
    I2 = Interval(np.array([[-1], [-5]]), np.array([[7], [9]]))
    I_cartProd = I1.cartProd(I2)
    I_true = Interval(np.array([[-2], [-4], [-3], [-1], [-5]]), np.array([[2], [6], [1], [7], [9]]))
    assert I_cartProd.isequal(I_true)

    # 2. horzcat
    I1 = Interval(np.array([[-1, -5]]), np.array([[7, 9]]))
    I2 = Interval(np.array([[-2, -4, -3]]), np.array([[2, 6, 1]]))
    I_cartProd = I1.cartProd(I2)
    I_true = Interval(np.array([[-1, -5, -2, -4, -3]]), np.array([[7, 9, 2, 6, 1]]))
    assert I_cartProd.isequal(I_true)

    # 3. interval-numeric case
    I1 = Interval(np.array([[-2], [-4], [-3]]), np.array([[2], [6], [1]]))
    num = np.array([[2], [1]])
    # ...I x num
    I_cartProd = I1.cartProd(num)
    I_true = Interval(np.array([[-2], [-4], [-3], [2], [1]]), np.array([[2], [6], [1], [2], [1]]))
    assert I_cartProd.isequal(I_true)
    # ...num x I
    I_cartProd_swapped = I1.cartProd(num)
    I_true_swapped = Interval(np.array([[-2], [-4], [-3], [2], [1]]), np.array([[2], [6], [1], [2], [1]]))
    assert I_cartProd_swapped.isequal(I_true_swapped)

    # 4. unbounded case
    I1 = Interval(np.array([[-np.inf], [-2], [1]]), np.array([[-2], [np.inf], [np.inf]]))
    I2 = Interval(np.array([[2], [-np.inf]]), np.array([[np.inf], [np.inf]]))
    I_cartProd = I1.cartProd(I2)
    I_true = Interval(np.array([[-np.inf], [-2], [1], [2], [-np.inf]]), np.array([[-2], [np.inf], [np.inf], [np.inf], [np.inf]]))
    assert I_cartProd.isequal(I_true) 