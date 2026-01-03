"""
test_location_calcBasis - test function for computation of orthogonal
   basis for guard intersection

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/location/test_location_calcBasis.m

Authors:       Mark Wetzlinger
Written:       17-May-2023
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.location.location import Location
from cora_python.hybridDynamics.transition.transition import Transition
from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.polytope.polytope import Polytope


def test_calcBasis_01_box_method():
    """
    TRANSLATED TEST - Box method test
    
    Tests calcBasis with 'box' method.
    """
    # 2D location: flow moving to the right
    inv = Polytope(np.array([[1, 0]]), np.array([[1]]))
    guard = Polytope(np.array([]), np.array([]),
                     np.array([[1, 0]]), np.array([[1]]))
    reset = LinearReset(np.array([[1, 0],
                                  [0, 1]]),
                       np.zeros((2, 1)),
                       np.array([[5], [0]]))
    trans = Transition(guard, reset, 1)
    flow = LinearSys('linearSys', np.zeros((2, 2)), np.array([[0]]), np.array([[1], [0]]))
    loc = Location(inv, trans, flow)
    
    # intersecting reachable sets
    R = [
        Zonotope(np.array([[0], [1]]), np.array([[1, 0],
                                                   [0, 1]])),
        Zonotope(np.array([[1], [0]]), np.array([[1, 0],
                                                  [0, 1]])),
        Zonotope(np.array([[2], [-1]]), np.array([[1, 0],
                                                   [0, 1]]))
    ]
    
    # go through different methods
    options = {'enclose': ['box']}
    B = loc.calcBasis(R, guard, options)
    from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
    assert np.all(withinTol(B[0], np.eye(2))), "Box method should return identity matrix"


def test_calcBasis_02_pca_method():
    """
    TRANSLATED TEST - PCA method test
    
    Tests calcBasis with 'pca' method.
    """
    # 2D location: flow moving to the right
    inv = Polytope(np.array([[1, 0]]), np.array([[1]]))
    guard = Polytope(np.array([]), np.array([]),
                     np.array([[1, 0]]), np.array([[1]]))
    reset = LinearReset(np.array([[1, 0],
                                  [0, 1]]),
                       np.zeros((2, 1)),
                       np.array([[5], [0]]))
    trans = Transition(guard, reset, 1)
    flow = LinearSys('linearSys', np.zeros((2, 2)), np.array([[0]]), np.array([[1], [0]]))
    loc = Location(inv, trans, flow)
    
    # intersecting reachable sets
    R = [
        Zonotope(np.array([[0], [1]]), np.array([[1, 0],
                                                   [0, 1]])),
        Zonotope(np.array([[1], [0]]), np.array([[1, 0],
                                                  [0, 1]])),
        Zonotope(np.array([[2], [-1]]), np.array([[1, 0],
                                                   [0, 1]]))
    ]
    
    options = {'enclose': ['pca']}
    B = loc.calcBasis(R, guard, options)
    from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
    # PCA should return [0 1; 1 0] (swapped columns)
    expected = np.array([[0, 1],
                         [1, 0]])
    assert np.all(withinTol(B[0], expected)), "PCA method should return swapped identity"


def test_calcBasis_03_flow_method():
    """
    TRANSLATED TEST - Flow method test
    
    Tests calcBasis with 'flow' method.
    """
    # 2D location: flow moving to the right
    inv = Polytope(np.array([[1, 0]]), np.array([[1]]))
    guard = Polytope(np.array([]), np.array([]),
                     np.array([[1, 0]]), np.array([[1]]))
    reset = LinearReset(np.array([[1, 0],
                                  [0, 1]]),
                       np.zeros((2, 1)),
                       np.array([[5], [0]]))
    trans = Transition(guard, reset, 1)
    flow = LinearSys('linearSys', np.zeros((2, 2)), np.array([[0]]), np.array([[1], [0]]))
    loc = Location(inv, trans, flow)
    
    # intersecting reachable sets
    R = [
        Zonotope(np.array([[0], [1]]), np.array([[1, 0],
                                                   [0, 1]])),
        Zonotope(np.array([[1], [0]]), np.array([[1, 0],
                                                  [0, 1]])),
        Zonotope(np.array([[2], [-1]]), np.array([[1, 0],
                                                   [0, 1]]))
    ]
    
    options = {'enclose': ['flow']}
    params = {
        'uTrans': np.array([[0], [0]]),
        'w': 0
    }
    B = loc.calcBasis(R, guard, options, params)
    from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
    assert np.all(withinTol(B[0], np.eye(2))), "Flow method should return identity matrix"


