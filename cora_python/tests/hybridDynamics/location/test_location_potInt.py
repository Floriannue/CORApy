"""
test_location_potInt - test function for finding potential intersections
   of the reachable set and the guard set

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/location/test_location_potInt.m

Authors:       Mark Wetzlinger
Written:       19-May-2023
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.location.location import Location
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.g.classes.reachSet.reachSet import ReachSet
from cora_python.contSet.interval.interval import Interval
from cora_python.hybridDynamics.transition.transition import Transition
from cora_python.hybridDynamics.linearReset.linearReset import LinearReset


def test_potInt_01_basic():
    """
    TRANSLATED TEST - Basic potInt test
    
    Tests potInt with multiple reachable sets and guards.
    """
    # init location
    inv = Interval(np.array([[-2], [-1]]), np.array([[3], [5]]))
    guard = Polytope(np.array([]), np.array([]),
                     np.array([[1, 0]]), np.array([[3]]))
    reset = LinearReset(np.eye(2), np.zeros((2, 1)), np.array([[2], [0]]))
    trans = [Transition(guard, reset, 2)]
    guard2 = Polytope(np.array([]), np.array([]),
                      np.array([[0, 1]]), np.array([[5]]))
    reset2 = LinearReset(np.eye(2), np.zeros((2, 1)), np.array([[0], [2]]))
    trans.append(Transition(guard2, reset2, 2))
    flow = LinearSys('linearSys', np.zeros((2, 2)), np.array([[0]]), np.array([[1], [1]]))
    loc = Location(inv, trans, flow)
    
    # reachSet object
    # -- not intersecting any guard
    sets = {
        'set': [
            Zonotope(np.array([[1], [0]]), 0.05 * np.eye(2)),
            Zonotope(np.array([[2], [0]]), 0.05 * np.eye(2)),
            # -- only intersecting guard of first transition
            Zonotope(np.array([[3], [0]]), 0.05 * np.eye(2)),
            Zonotope(np.array([[3], [2]]), 0.05 * np.eye(2)),
            Zonotope(np.array([[3], [4]]), 0.05 * np.eye(2)),
            # -- intersecting both guards
            Zonotope(np.array([[3], [5]]), 0.05 * np.eye(2)),
            # -- only intersecting guard of second transition
            Zonotope(np.array([[2], [5]]), 0.05 * np.eye(2)),
            Zonotope(np.array([[1], [5]]), 0.05 * np.eye(2)),
            # -- not intersecting any guard
            Zonotope(np.array([[1], [4]]), 0.05 * np.eye(2))
        ]
    }
    # time is irrelevant
    time_list = []
    for i in range(9):
        time_list.append(Interval([i], [i+1]))
    sets['time'] = time_list
    # init object (no time-point solutions)
    R = ReachSet(timePoint=None, timeInterval=sets)
    
    # set final location (just a dummy value here...)
    finalLoc = 4
    
    # check for potential intersections
    guards, setIndices, setType = loc.potInt(R, finalLoc)
    
    # total intersections: 7
    assert len(guards) == 7, "Should have 7 intersections"
    # correct guards for each intersection
    expected_guards = np.array([1, 1, 1, 1, 2, 2, 2])
    np.testing.assert_array_equal(guards, expected_guards, 
                                  err_msg="Guards should match expected")
    # correct indices of intersecting sets (MATLAB 1-based, Python 0-based)
    # MATLAB: [3,4,5,6,6,7,8] -> Python: [2,3,4,5,5,6,7]
    expected_indices = np.array([2, 3, 4, 5, 5, 6, 7])
    np.testing.assert_array_equal(setIndices, expected_indices,
                                  err_msg="Set indices should match expected")



