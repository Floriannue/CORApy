"""
test_location_guardIntersect_helpers - test function for guardIntersect helper methods

GENERATED TEST - No MATLAB test file exists
This test is generated based on MATLAB source code logic.

Source: cora_matlab/hybridDynamics/@location/guardIntersect.m (auxiliary functions)
Generated: 2025-01-XX
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.location.guardIntersect import (
    _aux_groupSets, _aux_removeEmptySets, _aux_getInitialSet, _aux_removeGaps
)
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.polytope.polytope import Polytope


def test_aux_groupSets_01_basic():
    """
    GENERATED TEST - Basic aux_groupSets test
    
    Tests grouping of sets that intersect guards.
    """
    # create sets
    Pset = [
        Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2)),
        Zonotope(np.array([[1], [0]]), 0.1 * np.eye(2)),
        Zonotope(np.array([[2], [0]]), 0.1 * np.eye(2)),
        Zonotope(np.array([[3], [0]]), 0.1 * np.eye(2))
    ]
    
    guards = [0, 0, 1, 1]  # first two sets hit guard 0, last two hit guard 1
    setIndices = [1, 2, 3, 4]  # MATLAB 1-based indices
    
    minInd, maxInd, P, guards_out = _aux_groupSets(Pset, guards, setIndices)
    
    assert len(minInd) > 0, "Should have grouped sets"
    assert len(maxInd) > 0, "Should have max indices"
    assert len(P) > 0, "Should have grouped sets"
    assert len(guards_out) > 0, "Should have guard indices"
    # minInd should be less than or equal to maxInd
    assert all(mi <= ma for mi, ma in zip(minInd, maxInd)), \
        "minInd should be <= maxInd"


def test_aux_groupSets_02_consecutive():
    """
    GENERATED TEST - Consecutive sets grouping test
    
    Tests grouping of consecutive sets.
    """
    Pset = [
        Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2)),
        Zonotope(np.array([[1], [0]]), 0.1 * np.eye(2)),
        Zonotope(np.array([[2], [0]]), 0.1 * np.eye(2))
    ]
    
    guards = [0, 0, 0]  # all same guard
    setIndices = [1, 2, 3]  # consecutive
    
    minInd, maxInd, P, guards_out = _aux_groupSets(Pset, guards, setIndices)
    
    # Should group consecutive sets together
    assert len(minInd) == 1, "Consecutive sets should be grouped"
    assert minInd[0] == 1, "minInd should be first index"
    assert maxInd[0] == 3, "maxInd should be last index"


def test_aux_removeEmptySets_01_basic():
    """
    GENERATED TEST - Basic aux_removeEmptySets test
    
    Tests removal of empty sets from guard intersections.
    """
    # create some sets (some empty, some not)
    # Note: We'll use None to represent empty sets, as the function checks for None
    R = [
        Zonotope(np.array([[1], [0]]), 0.1 * np.eye(2)),  # non-empty
        None,  # empty (None)
        Zonotope(np.array([[2], [0]]), 0.1 * np.eye(2))   # non-empty
    ]
    minInd = [1, 2, 3]
    maxInd = [1, 2, 3]
    actGuards = [0, 0, 0]
    
    R_out, minInd_out, maxInd_out, actGuards_out = _aux_removeEmptySets(
        R, minInd, maxInd, actGuards)
    
    # Should remove None/empty set
    assert len(R_out) < len(R), "Should remove empty sets"
    assert len(minInd_out) == len(R_out), "minInd should match R length"
    assert len(maxInd_out) == len(R_out), "maxInd should match R length"
    assert len(actGuards_out) == len(R_out), "actGuards should match R length"
    # All remaining sets should be non-empty
    assert all(r is not None for r in R_out), \
        "All remaining sets should be non-empty"


def test_aux_removeEmptySets_02_all_empty():
    """
    GENERATED TEST - All empty sets test
    
    Tests removal when all sets are empty.
    """
    R = [None, None]
    minInd = [1, 2]
    maxInd = [1, 2]
    actGuards = [0, 0]
    
    R_out, minInd_out, maxInd_out, actGuards_out = _aux_removeEmptySets(
        R, minInd, maxInd, actGuards)
    
    assert len(R_out) == 0, "Should remove all empty sets"
    assert len(minInd_out) == 0, "Should have no minInd"
    assert len(maxInd_out) == 0, "Should have no maxInd"
    assert len(actGuards_out) == 0, "Should have no actGuards"


def test_aux_getInitialSet_01_basic():
    """
    GENERATED TEST - Basic aux_getInitialSet test
    
    Tests retrieval of initial set for guard intersection.
    """
    Rtp = [
        Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2)),
        Zonotope(np.array([[1], [0]]), 0.1 * np.eye(2)),
        Zonotope(np.array([[2], [0]]), 0.1 * np.eye(2))
    ]
    
    minInd = 2  # MATLAB 1-based, Python 0-based: 1
    
    R0 = _aux_getInitialSet(Rtp, minInd)
    
    assert R0 is not None, "Should return initial set"
    # R0 should be the set at index minInd-1 (0-based)
    assert hasattr(R0, 'center') or hasattr(R0, 'c'), \
        "R0 should be a set with center or c property"
    # For minInd=2 (1-based), should get Rtp[1] (0-based)
    expected_idx = minInd - 1
    if expected_idx < len(Rtp):
        assert R0 is Rtp[expected_idx] or \
               (hasattr(R0, 'center') and hasattr(Rtp[expected_idx], 'center') and
                np.allclose(R0.center, Rtp[expected_idx].center)), \
            "R0 should match set at minInd-1"


def test_aux_getInitialSet_02_first_set():
    """
    GENERATED TEST - First set test
    
    Tests retrieval when minInd is 1 (first set).
    """
    Rtp = [
        Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2)),
        Zonotope(np.array([[1], [0]]), 0.1 * np.eye(2))
    ]
    
    minInd = 1  # MATLAB 1-based, first set
    
    R0 = _aux_getInitialSet(Rtp, minInd)
    
    assert R0 is not None, "Should return initial set"
    # Should get first set (index 0)
    assert R0 is Rtp[0] or \
           (hasattr(R0, 'center') and hasattr(Rtp[0], 'center') and
            np.allclose(R0.center, Rtp[0].center)), \
        "R0 should be first set"


def test_aux_removeGaps_01_basic():
    """
    GENERATED TEST - Basic aux_removeGaps test
    
    Tests removal of gaps in set indices.
    """
    # create sets with gaps
    Pset = [
        Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2)),
        Zonotope(np.array([[1], [0]]), 0.1 * np.eye(2)),
        Zonotope(np.array([[2], [0]]), 0.1 * np.eye(2)),
        Zonotope(np.array([[3], [0]]), 0.1 * np.eye(2))
    ]
    
    # indices with gap: [1, 2, 4] (gap between 2 and 4)
    setIndicesGuards = [[1, 2, 4]]
    guards = np.array([0])
    Pint = [[Pset[0], Pset[1], Pset[3]]]  # skip Pset[2] to create gap
    
    minInd, maxInd, Pint_out, guards_out = _aux_removeGaps(
        setIndicesGuards, guards, Pint)
    
    # Should split into two groups: [1, 2] and [4]
    assert len(minInd) == 2, "Should split into two groups due to gap"
    assert minInd[0] == 1, "First group should start at 1"
    assert maxInd[0] == 2, "First group should end at 2"
    assert minInd[1] == 4, "Second group should start at 4"
    assert maxInd[1] == 4, "Second group should end at 4"


def test_aux_removeGaps_02_no_gaps():
    """
    GENERATED TEST - No gaps test
    
    Tests when there are no gaps in indices.
    """
    Pset = [
        Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2)),
        Zonotope(np.array([[1], [0]]), 0.1 * np.eye(2)),
        Zonotope(np.array([[2], [0]]), 0.1 * np.eye(2))
    ]
    
    # consecutive indices: [1, 2, 3]
    setIndicesGuards = [[1, 2, 3]]
    guards = np.array([0])
    Pint = [[Pset[0], Pset[1], Pset[2]]]
    
    minInd, maxInd, Pint_out, guards_out = _aux_removeGaps(
        setIndicesGuards, guards, Pint)
    
    # Should remain as one group
    assert len(minInd) == 1, "Should remain as one group"
    assert minInd[0] == 1, "minInd should be 1"
    assert maxInd[0] == 3, "maxInd should be 3"

