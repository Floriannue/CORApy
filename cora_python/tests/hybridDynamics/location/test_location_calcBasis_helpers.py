"""
test_location_calcBasis_helpers - test function for calcBasis helper methods

GENERATED TEST - No MATLAB test file exists
This test is generated based on MATLAB source code logic.

Source: cora_matlab/hybridDynamics/@location/calcBasis.m (auxiliary functions)
Generated: 2025-01-XX
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.location.calcBasis import (
    _aux_extractGenerators, _aux_extractCenter
)
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.polytope.polytope import Polytope


def test_aux_extractGenerators_01_basic():
    """
    GENERATED TEST - Basic aux_extractGenerators test
    
    Tests extraction of generators from reachable sets.
    """
    # create list of sets
    R = [
        Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]])),
        Zonotope(np.array([[1], [1]]), np.array([[0.5, -0.3], [0.3, 0.5]]))
    ]
    
    G = _aux_extractGenerators(R)
    
    assert G is not None, "Should return generator matrix"
    assert isinstance(G, np.ndarray), "Should return numpy array"
    assert G.shape[0] == R[0].dim(), "Should have correct number of rows"
    # Number of columns should match total generators
    total_generators = sum(r.G.shape[1] if hasattr(r, 'G') else 0 for r in R)
    assert G.shape[1] == total_generators, "Should have correct number of columns"


def test_aux_extractGenerators_02_empty():
    """
    GENERATED TEST - Empty list test
    
    Tests extraction with empty list.
    """
    R = []
    
    G = _aux_extractGenerators(R)
    
    # Should handle empty list gracefully
    assert G is not None, "Should return something (even if empty)"
    assert isinstance(G, np.ndarray), "Should return numpy array"


def test_aux_extractCenter_01_basic():
    """
    GENERATED TEST - Basic aux_extractCenter test
    
    Tests extraction of centers from reachable sets.
    """
    # create list of sets
    R = [
        Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]])),
        Zonotope(np.array([[1], [1]]), np.array([[0.5, -0.3], [0.3, 0.5]]))
    ]
    
    c = _aux_extractCenter(R)
    
    assert c is not None, "Should return center vector"
    assert isinstance(c, np.ndarray), "Should return numpy array"
    assert c.shape[0] == R[0].dim(), "Should have correct dimension"
    # Should be average or weighted average of centers
    assert len(c.shape) == 1 or c.shape[1] == 1, "Should be a vector"


def test_aux_extractCenter_02_single_set():
    """
    GENERATED TEST - Single set test
    
    Tests extraction with single set.
    """
    R = [Zonotope(np.array([[2], [3]]), np.array([[1, 0], [0, 1]]))]
    
    c = _aux_extractCenter(R)
    
    assert c is not None, "Should return center"
    assert np.allclose(c, np.array([[2], [3]]), atol=1e-6), \
        "Center should match single set center"

