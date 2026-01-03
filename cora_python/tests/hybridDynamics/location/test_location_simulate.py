"""
test_location_simulate - test function for simulate

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/location/test_location_simulate.m

Authors:       Mark Wetzlinger
Written:       19-May-2023
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
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.g.functions.matlab.validate.check.compareMatrices import compareMatrices


def test_location_simulate_01_basic():
    """
    TRANSLATED TEST - Basic location simulate test
    
    Tests simulation of location with guard intersections.
    """
    # init location with simple dynamics
    inv = Interval(np.array([[-1], [-1]]), np.array([[1], [1]]))
    guard = Polytope(np.array([]), np.array([]),
                     np.array([[1, 0]]), np.array([[1]]))
    reset = LinearReset(np.zeros((2, 2)), np.zeros((2, 1)), np.array([[-0.9], [-0.9]]))
    trans1 = Transition(guard, reset, 2)
    guard2 = Polytope(np.array([]), np.array([]),
                      np.array([[0, 1]]), np.array([[1]]))
    reset2 = LinearReset(np.zeros((2, 2)), np.zeros((2, 1)), np.array([[-0.5], [0]]))
    trans2 = Transition(guard2, reset2, 3)
    flow = LinearSys('linearSys', np.zeros((2, 2)), np.array([[0], [0]]), np.array([[1], [1]]))
    loc = Location(inv, [trans1, trans2], flow)
    
    # model parameters
    params = {
        'loc': 1,
        'x0': np.array([[-0.5], [0]]),
        'tFinal': 2
    }
    
    # simulate
    t, x, nextLoc, xJump = loc.simulate(params)
    
    # takes 1 second to hit guard
    assert withinTol(t[-1], 1), "Should hit guard at t=1"
    # hits guard at x = [0.5;1]
    assert compareMatrices(x[-1, :].reshape(-1, 1), np.array([[0.5], [1]])), \
        "Should hit guard at x=[0.5;1]"
    # all points must be in invariant
    assert all(inv.contains_(x[i, :].reshape(-1, 1)) for i in range(len(x))), \
        "All points should be in invariant"
    # next location must be 3 (since second transition hit)
    assert nextLoc == 3, "Next location should be 3"
    # point after reset function
    assert compareMatrices(xJump, trans2.reset.c), "xJump should match reset.c"

