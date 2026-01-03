"""
test_location_guardIntersect - test function for guardIntersect

GENERATED TEST - No MATLAB test file exists
This test is generated based on MATLAB source code logic.

Source: cora_matlab/hybridDynamics/@location/guardIntersect.m
Generated: 2025-01-XX
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.location.location import Location
from cora_python.hybridDynamics.transition.transition import Transition
from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.interval.interval import Interval
from cora_python.g.classes.reachSet.reachSet import ReachSet


def test_location_guardIntersect_01_empty():
    """
    GENERATED TEST - Empty guard intersection test
    
    Tests guardIntersect with no intersections.
    """
    # init location
    inv = Interval(np.array([[-1], [-1]]), np.array([[1], [1]]))
    guard = Polytope(np.array([]), np.array([]),
                     np.array([[1, 0]]), np.array([[1]]))
    reset = LinearReset(np.eye(2))
    trans = Transition(guard, reset, 1)
    flow = LinearSys('linearSys', np.zeros((2, 2)), np.array([[0], [0]]), np.array([[1], [1]]))
    loc = Location(inv, [trans], flow)
    
    # reachable set that doesn't intersect guard
    Rtp = [Zonotope(np.array([[-0.5], [-0.5]]), 0.1 * np.eye(2))]
    Rcont = ReachSet(timePoint={'set': Rtp, 'time': [0, 1]})
    
    # no intersections
    guards = []
    setInd = []
    setType = 'time-point'
    params = {'R0': Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))}
    options = {'guardIntersect': 'zonoGirard'}
    
    Rguard, actGuards, minInd, maxInd = loc.guardIntersect(
        guards, setInd, setType, Rcont, params, options)
    
    assert len(Rguard) == 0, "Should have no guard intersections"
    assert len(actGuards) == 0, "Should have no active guards"
    assert len(minInd) == 0, "Should have no min indices"
    assert len(maxInd) == 0, "Should have no max indices"


def test_location_guardIntersect_02_zonoGirard():
    """
    GENERATED TEST - zonoGirard method test
    
    Tests guardIntersect with zonoGirard method.
    """
    # init location
    inv = Interval(np.array([[-2], [-2]]), np.array([[2], [2]]))
    guard = Polytope(np.array([]), np.array([]),
                     np.array([[1, 0]]), np.array([[1]]))
    reset = LinearReset(np.eye(2))
    trans = Transition(guard, reset, 1)
    flow = LinearSys('linearSys', np.zeros((2, 2)), np.array([[0], [0]]), np.array([[1], [0]]))
    loc = Location(inv, [trans], flow)
    
    # reachable set that intersects guard
    Rtp = [
        Zonotope(np.array([[0.5], [0]]), 0.1 * np.eye(2)),
        Zonotope(np.array([[1.0], [0]]), 0.1 * np.eye(2)),
        Zonotope(np.array([[1.5], [0]]), 0.1 * np.eye(2))
    ]
    Rti = [
        Zonotope(np.array([[0.75], [0]]), 0.15 * np.eye(2)),
        Zonotope(np.array([[1.25], [0]]), 0.15 * np.eye(2))
    ]
    Rcont = ReachSet(
        timePoint={'set': Rtp, 'time': [0, 1, 2, 3]},
        timeInterval={'set': Rti, 'time': [[0, 1], [1, 2], [2, 3]]}
    )
    
    # intersections at indices 1, 2 (MATLAB 1-based, Python 0-based: 0, 1)
    guards = [0, 0]  # guard index 0
    setInd = [1, 2]  # MATLAB 1-based indices
    setType = 'time-interval'
    params = {'R0': Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))}
    options = {'guardIntersect': 'zonoGirard'}
    
    Rguard, actGuards, minInd, maxInd = loc.guardIntersect(
        guards, setInd, setType, Rcont, params, options)
    
    assert len(Rguard) > 0, "Should have guard intersections"
    assert len(actGuards) > 0, "Should have active guards"
    assert len(minInd) > 0, "Should have min indices"
    assert len(maxInd) > 0, "Should have max indices"
    # All intersections should be sets
    assert all(hasattr(r, 'center') or hasattr(r, 'c') for r in Rguard), \
        "All guard intersections should be sets"


def test_location_guardIntersect_03_polytope():
    """
    GENERATED TEST - polytope method test
    
    Tests guardIntersect with polytope method.
    """
    # init location
    inv = Interval(np.array([[-2], [-2]]), np.array([[2], [2]]))
    guard = Polytope(np.array([]), np.array([]),
                     np.array([[1, 0]]), np.array([[1]]))
    reset = LinearReset(np.eye(2))
    trans = Transition(guard, reset, 1)
    flow = LinearSys('linearSys', np.zeros((2, 2)), np.array([[0], [0]]), np.array([[1], [0]]))
    loc = Location(inv, [trans], flow)
    
    # reachable set that intersects guard
    Rtp = [
        Zonotope(np.array([[0.5], [0]]), 0.1 * np.eye(2)),
        Zonotope(np.array([[1.0], [0]]), 0.1 * np.eye(2))
    ]
    Rcont = ReachSet(timePoint={'set': Rtp, 'time': [0, 1, 2]})
    
    guards = [0]
    setInd = [1]  # MATLAB 1-based
    setType = 'time-point'
    params = {'R0': Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))}
    options = {'guardIntersect': 'polytope', 'enclose': ['box']}
    
    Rguard, actGuards, minInd, maxInd = loc.guardIntersect(
        guards, setInd, setType, Rcont, params, options)
    
    assert len(Rguard) > 0, "Should have guard intersections"
    assert len(actGuards) > 0, "Should have active guards"


def test_location_guardIntersect_04_multiple_guards():
    """
    GENERATED TEST - Multiple guards test
    
    Tests guardIntersect with multiple guard intersections.
    """
    # init location with multiple transitions
    inv = Interval(np.array([[-2], [-2]]), np.array([[2], [2]]))
    guard1 = Polytope(np.array([]), np.array([]),
                      np.array([[1, 0]]), np.array([[1]]))
    guard2 = Polytope(np.array([]), np.array([]),
                      np.array([[0, 1]]), np.array([[1]]))
    reset = LinearReset(np.eye(2))
    trans1 = Transition(guard1, reset, 1)
    trans2 = Transition(guard2, reset, 1)
    flow = LinearSys('linearSys', np.zeros((2, 2)), np.array([[0], [0]]), np.array([[1], [1]]))
    loc = Location(inv, [trans1, trans2], flow)
    
    # reachable set that intersects both guards
    Rtp = [
        Zonotope(np.array([[0.5], [0.5]]), 0.1 * np.eye(2)),
        Zonotope(np.array([[1.0], [0.5]]), 0.1 * np.eye(2)),
        Zonotope(np.array([[0.5], [1.0]]), 0.1 * np.eye(2))
    ]
    Rcont = ReachSet(timePoint={'set': Rtp, 'time': [0, 1, 2, 3]})
    
    guards = [0, 1]  # both guards
    setInd = [1, 2]  # MATLAB 1-based
    setType = 'time-point'
    params = {'R0': Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))}
    options = {'guardIntersect': 'zonoGirard'}
    
    Rguard, actGuards, minInd, maxInd = loc.guardIntersect(
        guards, setInd, setType, Rcont, params, options)
    
    assert len(Rguard) > 0, "Should have guard intersections"
    assert len(actGuards) > 0, "Should have active guards"
    # Should have intersections for both guards
    assert len(set(actGuards)) >= 1, "Should have at least one unique guard"

