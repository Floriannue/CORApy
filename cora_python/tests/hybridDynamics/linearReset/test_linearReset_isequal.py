"""
test_linearReset_isequal - test function for equality check of
   linearReset objects

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/linearReset/test_linearReset_isequal.m

Authors:       Mark Wetzlinger
Written:       09-October-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.linearReset.linearReset import LinearReset


def test_linearReset_isequal_01_empty():
    """
    TRANSLATED TEST - Empty linearReset equality test
    """
    # empty cases
    linReset_empty = LinearReset()
    assert linReset_empty.isequal(linReset_empty), "Empty linearResets should be equal"
    linReset = LinearReset(None, None, None)
    assert linReset.isequal(linReset_empty), "Empty linearResets should be equal"
    assert linReset_empty.isequal(linReset), "Empty linearResets should be equal"


def test_linearReset_isequal_02_only_A():
    """
    TRANSLATED TEST - Only state matrix test
    """
    # only state matrix
    A = np.array([[1, 2], [0, -1]])
    linReset = LinearReset(A)
    assert linReset.isequal(linReset), "Same linearResets should be equal"
    A_tol = np.array([[1+1e-10, 2], [0, -1-1e-10]])
    linReset_tol = LinearReset(A_tol)
    assert not linReset.isequal(linReset_tol), "LinearResets with different A should not be equal"
    assert linReset.isequal(linReset_tol, 1e-8), "LinearResets should be equal within tolerance"


def test_linearReset_isequal_03_A_and_B():
    """
    TRANSLATED TEST - State and input matrix test
    """
    # state and input matrix
    A = np.array([[1, 2], [0, -1]])
    B = np.array([[2, 0, 1], [-1, 0, 0]])
    linReset = LinearReset(A, B)
    assert linReset.isequal(linReset), "Same linearResets should be equal"
    A_tol = np.array([[1+1e-10, 2], [0, -1-1e-10]])
    B_tol = np.array([[2, 0, 1], [-1, 1e-10, 0]])
    linReset_tol = LinearReset(A_tol, B_tol)
    assert not linReset.isequal(linReset_tol), "LinearResets with different A/B should not be equal"
    assert linReset.isequal(linReset_tol, 1e-8), "LinearResets should be equal within tolerance"


def test_linearReset_isequal_04_A_B_c():
    """
    TRANSLATED TEST - State matrix, input matrix, offset vector test
    """
    # state matrix, input matrix, offset vector
    A = np.array([[1, 2], [0, -1]])
    B = np.array([[2, 0, 1], [-1, 0, 0]])
    c = np.array([[-1], [0]])
    linReset = LinearReset(A, B, c)
    assert linReset.isequal(linReset), "Same linearResets should be equal"
    A_tol = np.array([[1+1e-10, 2], [0, -1-1e-10]])
    B_tol = np.array([[2, 0, 1], [-1, 1e-10, 0]])
    c_tol = np.array([[-1], [1e-10]])
    linReset_tol = LinearReset(A_tol, B_tol, c_tol)
    assert not linReset.isequal(linReset_tol), "LinearResets with different A/B/c should not be equal"
    assert linReset.isequal(linReset_tol, 1e-8), "LinearResets should be equal within tolerance"


def test_linearReset_isequal_05_default_cases():
    """
    TRANSLATED TEST - Default cases test
    """
    # default cases
    A = np.array([[1, 0], [0, -1]])
    B_def = np.array([[0], [0]])
    c_def = np.array([[0], [0]])
    linReset = LinearReset(A)
    linReset_def1 = LinearReset(A, B_def)
    linReset_def2 = LinearReset(A, None, c_def)
    linReset_def3 = LinearReset(A, B_def, c_def)
    assert linReset.isequal(linReset_def1), "LinearResets with default B should be equal"
    assert linReset.isequal(linReset_def2), "LinearResets with default c should be equal"
    assert linReset.isequal(linReset_def3), "LinearResets with default B and c should be equal"


def test_linearReset_isequal_06_all_zero():
    """
    TRANSLATED TEST - All-zero test
    """
    # all-zero
    A = np.zeros((3, 3))
    B = np.zeros((3, 1))
    c = np.zeros((3, 1))
    linReset = LinearReset(A, B, c)
    A_tol = np.array([[0, 0, 1e-10], [0, 0, 0], [-1e-10, 0, 0]])
    B_tol = np.array([[0], [0], [1e-10]])
    c_tol = np.array([[-1e-10], [0], [0]])
    linReset_tol = LinearReset(A_tol, B_tol, c_tol)
    assert not linReset.isequal(linReset_tol), "LinearResets with different values should not be equal"
    assert linReset.isequal(linReset_tol, 1e-10), "LinearResets should be equal within tolerance"

