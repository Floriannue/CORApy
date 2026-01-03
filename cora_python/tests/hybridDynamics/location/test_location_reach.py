"""
test_location_reach - test function for reach

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/location/test_location_reach.m

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
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.interval.interval import Interval
from cora_python.g.classes.reachSet.reachSet import ReachSet
from cora_python.specification.specification.specification import Specification


def test_location_reach_01_basic():
    """
    TRANSLATED TEST - Basic location reach test
    
    Tests location.reach with guard intersection.
    """
    # init location
    inv = Polytope(np.array([[1, 0]]), np.array([[2]]))
    guard = Polytope(np.array([]), np.array([]),
                     np.array([[1, 0]]), np.array([[2]]))
    reset = LinearReset(np.eye(2), np.zeros((2, 1)), np.array([[3], [-2]]))
    trans = Transition(guard, reset, 2)
    flow = LinearSys('linearSys', np.zeros((2, 2)), np.zeros((2, 1)), np.array([[1], [-0.2]]))
    loc = Location(inv, trans, flow)
    
    # start set and time
    R0 = Zonotope(np.array([[0.5], [3]]), 0.5 * np.eye(2))
    tStart = Interval([2], [2.5])
    
    # set parameters and options
    params = {
        'R0': R0,
        'U': Zonotope(np.zeros((2, 1)), np.array([]).reshape(2, 0)),
        'tStart': tStart,
        'tFinal': 10,
        'finalLoc': 4
    }
    
    options = {
        'timeStep': 1,
        'taylorTerms': 2,
        'zonotopeOrder': 10,
        'specification': Specification(),
        'guardIntersect': 'polytope',
        'enclose': ['box'],
        'intersectInvariant': False
    }
    
    # call reach
    R, Rjump, res_ = loc.reach(params, options)
    
    # no specification violated
    assert res_, "No specification should be violated"
    # reachable set needs to consist of four sets
    assert len(R.timeInterval['set']) == 4, "R.timeInterval.set should have 4 sets"
    # all Rjump - reset.c need to intersect the guard set
    from cora_python.contSet.zonotope.isIntersecting_ import isIntersecting_
    for rjump in Rjump:
        assert 'set' in rjump, "Rjump should have 'set' field"
        # Check intersection: (rjump.set - reset.c) should intersect guard
        rjump_set_minus_c = rjump['set'] - reset.c
        assert isIntersecting_(rjump_set_minus_c, guard), \
               "rjump.set - reset.c should intersect guard"

