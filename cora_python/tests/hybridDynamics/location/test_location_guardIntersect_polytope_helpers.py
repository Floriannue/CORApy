"""
test_location_guardIntersect_polytope_helpers - test function for guardIntersect_polytope helper methods

GENERATED TEST - No MATLAB test file exists
This test is generated based on MATLAB source code logic.

Source: cora_matlab/hybridDynamics/@location/guardIntersect_polytope.m (auxiliary functions)
Generated: 2025-01-XX
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.location.guardIntersect_polytope import (
    _aux_flowEnclosure, _aux_conv2polytope
)
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.polytope.polytope import Polytope


def test_aux_flowEnclosure_01_basic():
    """
    GENERATED TEST - Basic aux_flowEnclosure test
    
    Tests flow enclosure computation.
    """
    # create system
    sys = LinearSys('linearSys', np.array([[0, 1], [-1, 0]]), np.array([[1]]))
    
    # create velocity matrix (from reachable set)
    V = np.array([[1, 0], [0, 1]])
    
    # options
    options = {
        'enclose': ['box']
    }
    
    Z_flow = _aux_flowEnclosure(sys, V, options)
    
    assert Z_flow is not None, "Should return flow enclosure"
    assert hasattr(Z_flow, 'center') or hasattr(Z_flow, 'c'), \
        "Should be a set with center"
    assert Z_flow.dim() == sys.dim, "Should have correct dimension"


def test_aux_conv2polytope_01_basic():
    """
    GENERATED TEST - Basic aux_conv2polytope test
    
    Tests conversion to polytope.
    """
    # create zonotope
    Z = Zonotope(np.array([[0], [0]]), np.array([[1, 0.5], [0, 1]]))
    
    P = _aux_conv2polytope(Z)
    
    assert P is not None, "Should return polytope"
    assert isinstance(P, Polytope), "Should be a Polytope"
    # Polytope should contain the zonotope
    assert P.dim() == Z.dim(), "Should have same dimension"

