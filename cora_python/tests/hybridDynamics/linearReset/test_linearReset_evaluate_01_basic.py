"""
test_linearReset_evaluate - test function for the evaluation of a linear
   reset function

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/linearReset/test_linearReset_evaluate.m

Authors:       Mark Wetzlinger
Written:       10-October-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
from cora_python.hybridDynamics.linearReset.evaluate import evaluate
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def test_linearReset_evaluate_01_basic():
    """
    TRANSLATED TEST - Basic evaluate test
    
    Tests the evaluate method with basic inputs.
    """
    # tolerance
    tol = np.finfo(float).eps
    
    # init linear reset functions
    A = np.array([[1, 2],
                  [0, -1]])
    B = np.array([[2, 0, 1],
                  [-1, 0, 0]])
    c = np.array([[1], [-5]])
    linReset_A = LinearReset(A)
    linReset_AB = LinearReset(A, B)
    linReset_ABc = LinearReset(A, B, c)
    
    # vectors
    x = np.array([[5], [-2]])
    u = np.array([[1], [4], [-3]])
    
    # Basic case
    x_ = evaluate(linReset_A, x)
    x_true = A @ x
    assert np.all(withinTol(x_, x_true, tol)), "x_ should equal A*x"
    
    x_ = evaluate(linReset_AB, x)
    assert np.all(withinTol(x_, x_true, tol)), "x_ should equal A*x (no input)"
    
    # linear system with B matrix
    x_ = evaluate(linReset_AB, x)
    x_true = A @ x
    assert np.all(withinTol(x_, x_true, tol)), "x_ should equal A*x"
    
    x_ = evaluate(linReset_AB, x, u)
    x_true = A @ x + B @ u
    assert np.all(withinTol(x_, x_true, tol)), "x_ should equal A*x + B*u"
    
    # linear system with B and c
    x_ = evaluate(linReset_ABc, x)
    x_true = A @ x + c
    assert np.all(withinTol(x_, x_true, tol)), "x_ should equal A*x + c"
    
    x_ = evaluate(linReset_ABc, x, u)
    x_true = A @ x + B @ u + c
    assert np.all(withinTol(x_, x_true, tol)), "x_ should equal A*x + B*u + c"


def test_linearReset_evaluate_02_zonotopes():
    """
    TRANSLATED TEST - Evaluate with zonotopes test
    
    Tests the evaluate method with zonotopic sets.
    """
    # tolerance
    tol = np.finfo(float).eps
    
    # init linear reset functions
    A = np.array([[1, 2],
                  [0, -1]])
    B = np.array([[2, 0, 1],
                  [-1, 0, 0]])
    c = np.array([[1], [-5]])
    linReset_A = LinearReset(A)
    linReset_AB = LinearReset(A, B)
    linReset_ABc = LinearReset(A, B, c)
    
    # zonotopes
    Z_x = Zonotope(np.array([[1], [-1]]), np.array([[1, 0, -1],
                                                     [2, 1, 1]]))
    Z_u = Zonotope(np.array([[0], [1], [-1]]), np.array([[1, 0, 1],
                                                          [-1, -2, 1],
                                                          [2, 1, 3]]))
    
    # basic case with zonotopic output
    x_ = evaluate(linReset_A, Z_x)
    x_true = A @ Z_x
    assert x_.isequal(x_true, tol), "x_ should equal A*Z_x"
    
    x_ = evaluate(linReset_AB, Z_x)
    assert x_.isequal(x_true, tol), "x_ should equal A*Z_x"
    
    # zonotopic output with B matrix
    x_ = evaluate(linReset_AB, Z_x)
    x_true = A @ Z_x
    assert x_.isequal(x_true, tol), "x_ should equal A*Z_x"
    
    x_ = evaluate(linReset_AB, Z_x, Z_u)
    x_true = A @ Z_x + B @ Z_u
    assert x_.isequal(x_true, tol), "x_ should equal A*Z_x + B*Z_u"
    
    # zonotopic output with B and c
    x_ = evaluate(linReset_ABc, Z_x)
    x_true = A @ Z_x + c
    assert x_.isequal(x_true, tol), "x_ should equal A*Z_x + c"
    
    x_ = evaluate(linReset_ABc, Z_x, Z_u)
    x_true = A @ Z_x + B @ Z_u + c
    assert x_.isequal(x_true, tol), "x_ should equal A*Z_x + B*Z_u + c"

