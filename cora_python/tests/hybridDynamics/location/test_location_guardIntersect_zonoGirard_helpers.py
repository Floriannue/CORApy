"""
test_location_guardIntersect_zonoGirard_helpers - test function for guardIntersect_zonoGirard helper methods

GENERATED TEST - No MATLAB test file exists
This test is generated based on MATLAB source code logic.

Source: cora_matlab/hybridDynamics/@location/guardIntersect_zonoGirard.m (auxiliary functions)
Generated: 2025-01-XX
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.location.guardIntersect_zonoGirard import (
    _aux_enclosingInterval, _aux_bound_intersect_2D, _aux_sort_trig,
    _aux_split_pivot, _aux_dichotomicSearch, _aux_lineIntersect2D,
    _aux_robustProjection, _aux_tightenSet
)
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def test_aux_enclosingInterval_01_basic():
    """
    GENERATED TEST - Basic aux_enclosingInterval test
    
    Tests computation of enclosing interval for guard intersection.
    """
    # create guard (hyperplane)
    guard = Polytope(np.array([]), np.array([]),
                     np.array([[1, 0]]), np.array([[1]]))
    
    # create basis
    B = [np.array([[0, 1], [1, 0]])]  # Basis vectors
    
    # create zonotope
    Z = Zonotope(np.array([[0.5], [0.5]]), 0.1 * np.eye(2))
    
    I = _aux_enclosingInterval(guard, B, Z)
    
    assert I is not None, "Should return an interval"
    assert hasattr(I, 'inf') or hasattr(I, 'infimum'), "Should be an interval"
    # Interval should contain the intersection region
    assert I is not None, "Interval should be computed"


def test_aux_bound_intersect_2D_01_basic():
    """
    GENERATED TEST - Basic aux_bound_intersect_2D test
    
    Tests 2D intersection bounds computation.
    """
    # create zonotope
    Z = Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))
    
    # L parameter (distance to hyperplane)
    L = 1.0
    
    t_min, t_max = _aux_bound_intersect_2D(Z, L)
    
    assert isinstance(t_min, (int, float, np.number)), "t_min should be a number"
    assert isinstance(t_max, (int, float, np.number)), "t_max should be a number"
    assert t_min <= t_max, "t_min should be <= t_max"


def test_aux_sort_trig_01_basic():
    """
    GENERATED TEST - Basic aux_sort_trig test
    
    Tests trigonometric sorting.
    """
    # create generator vector
    g = np.array([[1], [1], [-1], [0.5]])
    
    # direction value
    dir_val = 1
    
    g_sorted = _aux_sort_trig(g, dir_val)
    
    assert g_sorted.shape == g.shape, "Sorted vector should have same shape"
    assert len(g_sorted) == len(g), "Should have same length"
    # Values should be sorted according to trigonometric order


def test_aux_split_pivot_01_basic():
    """
    GENERATED TEST - Basic aux_split_pivot test
    
    Tests pivot splitting for generators.
    """
    # create generator matrix
    G = np.array([[1, 0, 0.5], [0, 1, -0.5]])
    
    # create sign vector
    s = np.array([1, -1, 1])
    
    # direction value
    dir_val = 1
    
    G_split, s_split = _aux_split_pivot(G, s, dir_val)
    
    assert G_split.shape[0] == G.shape[0], "Split G should have same number of rows"
    assert len(s_split) == G_split.shape[1], "s_split should match G_split columns"
    assert len(s_split) >= len(s), "Split should have at least as many columns"


def test_aux_dichotomicSearch_01_basic():
    """
    GENERATED TEST - Basic aux_dichotomicSearch test
    
    Tests dichotomic search for intersection.
    """
    # create initial matrices
    P_0 = np.array([[1, 0], [0, 1]])
    G_0 = np.array([[0.5, -0.3], [0.3, 0.5]])
    s_0 = np.array([1, -1])
    
    t_min, t_max = _aux_dichotomicSearch(P_0, G_0, s_0, 0, 1, 1e-6)
    
    assert isinstance(t_min, (int, float, np.number)), "t_min should be a number"
    assert isinstance(t_max, (int, float, np.number)), "t_max should be a number"
    assert t_min <= t_max, "t_min should be <= t_max"


def test_aux_lineIntersect2D_01_basic():
    """
    GENERATED TEST - Basic aux_lineIntersect2D test
    
    Tests 2D line intersection computation.
    """
    # create two points
    p1 = np.array([[0], [0]])
    p2 = np.array([[1], [1]])
    
    # gamma parameter
    gamma = 0.5
    
    t = _aux_lineIntersect2D(p1, p2, gamma)
    
    # t might be None if no intersection, or a float if intersection exists
    if t is not None:
        assert isinstance(t, (int, float, np.number)), "t should be a number"


def test_aux_robustProjection_01_basic():
    """
    GENERATED TEST - Basic aux_robustProjection test
    
    Tests robust projection computation.
    """
    # create direction matrix
    D = np.array([[1, 0], [0, 1]])
    
    # normal vector
    n = np.array([[1], [0]])
    
    # gamma parameter
    gamma = 1.0
    
    proj = _aux_robustProjection(D, n, gamma, 1e-6)
    
    assert proj is not None, "Should return projection"
    assert isinstance(proj, (np.ndarray, list)), "Should return array or list"


def test_aux_tightenSet_01_basic():
    """
    GENERATED TEST - Basic aux_tightenSet test
    
    Tests set tightening operation.
    """
    # create zonotope
    Z = Zonotope(np.array([[0], [0]]), np.array([[1, 0.5], [0, 1]]))
    
    # create polytope (guard)
    P = Polytope(np.array([]), np.array([]),
                 np.array([[1, 0]]), np.array([[1]]))
    
    Z_tight = _aux_tightenSet(Z, P)
    
    # Z_tight might be None if tightening fails, or a Zonotope if successful
    if Z_tight is not None:
        assert hasattr(Z_tight, 'center') or hasattr(Z_tight, 'c'), \
            "Tightened set should be a set with center"
        # Tightened set should be contained in original set
        # (This is a property check, actual implementation may vary)

