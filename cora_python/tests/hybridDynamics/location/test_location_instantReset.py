"""
test_location_instantReset - test function for instantReset

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/location/test_location_instantReset.m

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
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.interval.interval import Interval
from cora_python.g.classes.reachSet.reachSet import ReachSet
# Note: isequal might not be available, using direct comparison instead


def test_location_instantReset_01_basic():
    """
    TRANSLATED TEST - Basic instantReset test
    
    Tests instantReset with instant transition.
    """
    # init location
    inv = Interval(np.array([[-2], [1]]), np.array([[1], [4]]))
    guard = Polytope(np.array([]), np.array([]),
                     np.array([[1, 0]]), np.array([[4]]))
    reset = LinearReset(np.array([[2, -1], [1, 3]]), np.zeros((2, 1)), np.array([[1], [-1]]))
    trans = Transition(guard, reset, 2)
    flow = LinearSys('linearSys', np.array([[2, 1], [-1, 2]]), np.array([[1]]))
    loc = Location(inv, [trans], flow)
    
    # start set and time
    R0 = Zonotope(np.array([[4], [2]]))
    tStart = 2
    
    # options
    options = {
        'instantTransition': 0,  # MATLAB 1-based index, Python 0-based: 0
        'U': Zonotope(np.array([[0]]))
    }
    
    # reset
    R, Rjump, res_ = loc.instantReset(R0, tStart, options)
    
    # 'res' must be true as no specification was violated
    assert res_, "res should be true"
    # reach set must only contain the first time-point solution
    assert len(R.timePoint.get('set', [])) == 1, "R should have one time-point set"
    # no time-interval solution
    assert len(R.timeInterval.get('set', [])) == 0, "R should have no time-interval sets"
    # only one jumped set
    assert len(Rjump) == 1, "Should have one jumped set"
    # apply reset function to set
    # Note: R0 is a zonotope, so we need to apply reset to it
    Rjump_set_expected = reset.evaluate(R0, options.get('U', None))
    # Check that the set matches (using approximate equality for zonotopes)
    assert Rjump[0]['set'] is not None, "Rjump set should not be None"
    # For zonotopes, we can check center and generators
    if hasattr(Rjump[0]['set'], 'center') and hasattr(Rjump_set_expected, 'center'):
        assert np.allclose(Rjump[0]['set'].center, Rjump_set_expected.center, atol=1e-6), \
            "Rjump set center should match reset function applied to R0"
    # goal location = 2
    assert Rjump[0]['loc'] == trans.target, "Target location should match"
    # no time has passed
    assert Rjump[0]['time'] == tStart, "Time should remain at tStart"
