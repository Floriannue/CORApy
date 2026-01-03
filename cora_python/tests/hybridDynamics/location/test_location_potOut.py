"""
test_location_potOut - test function for potOut

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/location/test_location_potOut.m

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


def test_potOut_01_basic():
    """
    TRANSLATED TEST - Basic potOut test
    
    Tests potOut with multiple reachable sets.
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
    
    # cell-array of reachable sets
    # -- not intersecting any guard
    timePoint = {
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
        ],
        'time': [1, 2, 3, 4, 5, 6, 7, 8, 9]
    }
    
    # Create time-interval sets using convHull
    timeInt = {
        'set': [],
        'time': []
    }
    for i in range(len(timePoint['set']) - 1):
        timeInt['set'].append(timePoint['set'][i].convHull_(timePoint['set'][i+1]))
        timeInt['time'].append([Interval([timePoint['time'][i]], [timePoint['time'][i+1]])])
    
    R = ReachSet(timePoint=timePoint, timeInterval=timeInt)
    
    # indices (MATLAB 1-based, potOut expects 1-based indices)
    minInd = np.array([3, 6])  # MATLAB: [3;6] (1-based)
    maxInd = np.array([6, 8])  # MATLAB: [6;8] (1-based)
    
    # evaluate function
    R = loc.potOut(R, minInd, maxInd)
    
    # sets 3 to 8 intersect and have been altered (MATLAB 1-based: 3-8, Python 0-based: 2-7)
    Rtp = R.timePoint['set'][2:8]  # Python 0-based indexing (sets 3-8 in MATLAB)
    Rti = R.timeInterval['set'][2:8]  # Python 0-based indexing (sets 3-8 in MATLAB)
    
    # all intersecting sets must be polytopes
    assert all(isinstance(x, Polytope) for x in Rtp), "All time-point sets should be polytopes"
    assert all(isinstance(x, Polytope) for x in Rti), "All time-interval sets should be polytopes"
    # all intersecting sets must be contained in invariant
    assert all(inv.contains_(x) for x in Rtp), "All time-point sets should be contained in invariant"
    assert all(inv.contains_(x) for x in Rti), "All time-interval sets should be contained in invariant"




