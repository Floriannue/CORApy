"""
test_linearReset_linearReset - test function for linearReset constructor

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/linearReset/test_linearReset_linearReset.m

Authors:       Mark Wetzlinger
Written:       08-October-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
from cora_python.contSet.zonotope.zonotope import Zonotope


def test_linearReset_01_empty():
    """
    TRANSLATED TEST - Empty constructor test
    
    Tests LinearReset with no arguments.
    """
    reset = LinearReset()
    
    assert reset.preStateDim == 0, "preStateDim should be 0"
    assert reset.inputDim == 1, "inputDim should be 1"
    assert reset.postStateDim == 0, "postStateDim should be 0"


def test_linearReset_02_A_only():
    """
    TRANSLATED TEST - A matrix only test
    
    Tests LinearReset with only A matrix.
    """
    A = np.array([[1, 0],
                  [0, 1]])
    reset = LinearReset(A)
    
    np.testing.assert_allclose(reset.A, A, err_msg="A should match input")
    np.testing.assert_allclose(reset.B, np.array([[0], [0]]), err_msg="B should be [0;0]")
    np.testing.assert_allclose(reset.c, np.array([[0], [0]]), err_msg="c should be [0;0]")
    assert reset.preStateDim == 2, "preStateDim should be 2"
    assert reset.inputDim == 1, "inputDim should be 1"
    assert reset.postStateDim == 2, "postStateDim should be 2"


def test_linearReset_03_A_B():
    """
    TRANSLATED TEST - A and B test
    
    Tests LinearReset with A and B.
    """
    A = np.array([[1, 0],
                  [0, 1]])
    B = np.array([[1], [-1]])
    reset = LinearReset(A, B)
    
    np.testing.assert_allclose(reset.A, A, err_msg="A should match input")
    np.testing.assert_allclose(reset.B, B, err_msg="B should match input")
    np.testing.assert_allclose(reset.c, np.array([[0], [0]]), err_msg="c should be [0;0]")
    assert reset.preStateDim == 2, "preStateDim should be 2"
    assert reset.inputDim == 1, "inputDim should be 1"
    assert reset.postStateDim == 2, "postStateDim should be 2"


def test_linearReset_04_A_c():
    """
    TRANSLATED TEST - A and c test
    
    Tests LinearReset with A and c.
    """
    A = np.array([[1, 0],
                  [0, 1]])
    c = np.array([[1], [-1]])
    reset = LinearReset(A, None, c)
    
    np.testing.assert_allclose(reset.A, A, err_msg="A should match input")
    np.testing.assert_allclose(reset.B, np.array([[0], [0]]), err_msg="B should be [0;0]")
    np.testing.assert_allclose(reset.c, c, err_msg="c should match input")
    assert reset.preStateDim == 2, "preStateDim should be 2"
    assert reset.inputDim == 1, "inputDim should be 1"
    assert reset.postStateDim == 2, "postStateDim should be 2"


def test_linearReset_05_A_B_c():
    """
    TRANSLATED TEST - A, B, c test
    
    Tests LinearReset with A, B, and c.
    """
    A = np.array([[1, 0],
                  [0, 1]])
    B = np.array([[0], [1]])
    c = np.array([[1], [-1]])
    reset = LinearReset(A, B, c)
    
    np.testing.assert_allclose(reset.A, A, err_msg="A should match input")
    np.testing.assert_allclose(reset.B, B, err_msg="B should match input")
    np.testing.assert_allclose(reset.c, c, err_msg="c should match input")
    assert reset.preStateDim == 2, "preStateDim should be 2"
    assert reset.inputDim == 1, "inputDim should be 1"
    assert reset.postStateDim == 2, "postStateDim should be 2"


def test_linearReset_04_evaluate_basic():
    """
    GENERATED TEST - Evaluate basic test
    
    Tests the evaluate method with basic inputs.
    """
    A = np.array([[1, 0],
                  [0, -0.75]])
    reset = LinearReset(A)
    
    x = np.array([[1], [1]])
    x_ = reset.evaluate(x)
    
    # Expected: x_ = A*x + B*u + c = A*x (since B*u and c are zero by default)
    expected = A @ x
    np.testing.assert_allclose(x_, expected, err_msg="evaluate should compute A*x")


def test_linearReset_05_evaluate_with_input():
    """
    GENERATED TEST - Evaluate with input test
    
    Tests the evaluate method with input u.
    """
    A = np.array([[1, 0],
                  [0, -0.75]])
    B = np.array([[1], [0]])
    c = np.array([[0], [0]])
    reset = LinearReset(A, B, c)
    
    x = np.array([[1], [1]])
    u = np.array([[0.5]])
    x_ = reset.evaluate(x, u)
    
    # Expected: x_ = A*x + B*u + c
    expected = A @ x + B @ u + c
    np.testing.assert_allclose(x_, expected, err_msg="evaluate should compute A*x + B*u + c")


def test_linearReset_06_evaluate_with_set():
    """
    GENERATED TEST - Evaluate with set test
    
    Tests the evaluate method with a zonotope set.
    """
    A = np.array([[1, 0],
                  [0, -0.75]])
    reset = LinearReset(A)
    
    x = Zonotope(np.array([[1], [1]]), 0.1 * np.eye(2))
    x_ = reset.evaluate(x)
    
    # Expected: x_ = A*x (zonotope multiplication)
    expected = A @ x
    assert isinstance(x_, Zonotope), "Result should be a Zonotope"
    np.testing.assert_allclose(x_.center(), expected.center(), 
                              err_msg="Center should match A*x center")


def test_linearReset_07_copy_constructor():
    """
    GENERATED TEST - Copy constructor test
    
    Tests LinearReset copy constructor.
    """
    A = np.array([[1, 0],
                  [0, -0.75]])
    B = np.array([[1], [0]])
    c = np.array([[0], [0]])
    reset1 = LinearReset(A, B, c)
    
    reset2 = LinearReset(reset1)
    
    np.testing.assert_allclose(reset2.A, reset1.A, err_msg="A should be copied")
    np.testing.assert_allclose(reset2.B, reset1.B, err_msg="B should be copied")
    np.testing.assert_allclose(reset2.c, reset1.c, err_msg="c should be copied")


def test_linearReset_08_dimensions():
    """
    GENERATED TEST - Dimensions test
    
    Tests that dimensions are computed correctly.
    """
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])  # 2x3 matrix
    B = np.array([[1], [2]])  # 2x1 matrix
    c = np.array([[0], [0]])  # 2x1 vector
    reset = LinearReset(A, B, c)
    
    assert reset.preStateDim == 3, "preStateDim should be 3"
    assert reset.postStateDim == 2, "postStateDim should be 2"
    assert reset.inputDim == 1, "inputDim should be 1"


