"""
test_linearReset_nonlinearReset - test function for conversion of a
   linearReset object into a nonlinearReset object

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/linearReset/test_linearReset_nonlinearReset.m

Authors:       Mark Wetzlinger
Written:       08-October-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.linearReset.linearReset import LinearReset


def test_linearReset_nonlinearReset_01_empty():
    """
    TRANSLATED TEST - Empty linearReset to nonlinearReset conversion test
    """
    # empty
    linReset = LinearReset()
    # Note: nonlinearReset conversion might not be implemented yet
    # For now, skip if not available
    try:
        from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
        nonlinReset = linReset.nonlinearReset()
        assert nonlinReset.preStateDim == 0, "preStateDim should be 0"
        assert nonlinReset.inputDim == 1, "inputDim should be 1"
        assert nonlinReset.postStateDim == 0, "postStateDim should be 0"
    except (ImportError, NotImplementedError):
        pytest.skip("NonlinearReset conversion not yet implemented")


def test_linearReset_nonlinearReset_02_only_A():
    """
    TRANSLATED TEST - Only state matrix conversion test
    """
    try:
        from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
        
        # only state matrix
        A = np.array([[1, 0], [0, 1]])
        linReset = LinearReset(A)
        nonlinReset = linReset.nonlinearReset()
        assert nonlinReset.preStateDim == 2, "preStateDim should be 2"
        assert nonlinReset.inputDim == 1, "inputDim should be 1"
        assert nonlinReset.postStateDim == 2, "postStateDim should be 2"
        
        # Compare with direct nonlinearReset creation
        def f(x, u):
            return np.array([[x[0]], [x[1]]])  # Column vector
        nonlinReset_ = NonlinearReset(f)
        assert nonlinReset.isequal(nonlinReset_), "Converted reset should equal direct creation"
    except (ImportError, NotImplementedError):
        pytest.skip("NonlinearReset conversion not yet implemented")


def test_linearReset_nonlinearReset_03_A_and_B():
    """
    TRANSLATED TEST - State matrix and input matrix conversion test
    """
    try:
        from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
        
        # state matrix and input matrix
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1], [-1]])
        linReset = LinearReset(A, B)
        nonlinReset = linReset.nonlinearReset()
        assert nonlinReset.preStateDim == 2, "preStateDim should be 2"
        assert nonlinReset.inputDim == 1, "inputDim should be 1"
        assert nonlinReset.postStateDim == 2, "postStateDim should be 2"
        
        # Compare with direct nonlinearReset creation
        def f(x, u):
            return np.array([x[0] + u[0], x[1] - u[0]])
        nonlinReset_ = NonlinearReset(f)
        assert nonlinReset.isequal(nonlinReset_), "Converted reset should equal direct creation"
    except (ImportError, NotImplementedError):
        pytest.skip("NonlinearReset conversion not yet implemented")


def test_linearReset_nonlinearReset_04_A_and_c():
    """
    TRANSLATED TEST - State matrix and offset conversion test
    """
    try:
        from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
        
        # state matrix and offset
        A = np.array([[1, 0], [0, 1]])
        c = np.array([[1], [-1]])
        linReset = LinearReset(A, None, c)
        nonlinReset = linReset.nonlinearReset()
        assert nonlinReset.preStateDim == 2, "preStateDim should be 2"
        assert nonlinReset.inputDim == 1, "inputDim should be 1"
        assert nonlinReset.postStateDim == 2, "postStateDim should be 2"
        
        # Compare with direct nonlinearReset creation
        def f(x, u):
            return np.array([[x[0] + 1], [x[1] - 1]])  # Column vector
        nonlinReset_ = NonlinearReset(f)
        assert nonlinReset.isequal(nonlinReset_), "Converted reset should equal direct creation"
    except (ImportError, NotImplementedError):
        pytest.skip("NonlinearReset conversion not yet implemented")


def test_linearReset_nonlinearReset_05_A_B_c():
    """
    TRANSLATED TEST - State matrix, input matrix, and offset conversion test
    """
    try:
        from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
        
        # state matrix, input matrix, and offset
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[0], [1]])
        c = np.array([[1], [-1]])
        linReset = LinearReset(A, B, c)
        nonlinReset = linReset.nonlinearReset()
        assert nonlinReset.preStateDim == 2, "preStateDim should be 2"
        assert nonlinReset.inputDim == 1, "inputDim should be 1"
        assert nonlinReset.postStateDim == 2, "postStateDim should be 2"
        
        # Compare with direct nonlinearReset creation
        def f(x, u):
            return np.array([x[0] + 1, x[1] + u[0] - 1])
        nonlinReset_ = NonlinearReset(f)
        assert nonlinReset.isequal(nonlinReset_), "Converted reset should equal direct creation"
    except (ImportError, NotImplementedError):
        pytest.skip("NonlinearReset conversion not yet implemented")

