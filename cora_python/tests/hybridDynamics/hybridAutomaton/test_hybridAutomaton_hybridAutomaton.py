"""
test_hybridAutomaton_hybridAutomaton - unit test function for constructor

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/hybridAutomaton/test_hybridAutomaton_hybridAutomaton.m

Authors:       Mark Wetzlinger
Written:       26-November-2022
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.hybridAutomaton.hybridAutomaton import HybridAutomaton
from cora_python.hybridDynamics.location.location import Location
from cora_python.hybridDynamics.transition.transition import Transition
from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def test_hybridAutomaton_hybridAutomaton_01_empty():
    """
    TRANSLATED TEST - Empty hybridAutomaton constructor test
    """
    # empty object
    HA = HybridAutomaton()
    assert HA is not None, "Empty hybridAutomaton should be created"


def test_hybridAutomaton_hybridAutomaton_02_standard():
    """
    TRANSLATED TEST - Standard hybridAutomaton constructor test
    """
    # invariant
    inv_2D = Polytope(np.array([[-1, 0]]), np.array([[0]]))
    inv_3D = Polytope(np.array([[-1, 0, 0]]), np.array([[0]]))
    
    # transition
    guard = Polytope(np.array([[0, 1]]), np.array([[0]]),
                     np.array([[-1, 0]]), np.array([[0]]))
    reset = LinearReset(np.array([[1, 0], [0, -0.75]]))
    trans_2D = Transition(guard, reset, 2)
    guard3D = Polytope(np.array([[0, 0, 1]]), np.array([[0]]),
                       np.array([[-1, 0, 0]]), np.array([[0]]))
    reset3D = LinearReset.eye(3)
    trans_3D = Transition(guard3D, reset3D, 1)
    
    # flow equation
    dynamics_2D = LinearSys('linearSys', np.array([[0, 1], [0, 0]]), 
                           np.array([[0], [0]]), np.array([[0], [-9.81]]))
    # MATLAB: linearSys([0,1,0;0,0,1;0,0,1],1) - scalar B=1 means 3x1 matrix of ones
    # In Python, we need to pass a proper B matrix with correct dimensions
    dynamics_3D = LinearSys('linearSys', np.array([[0, 1, 0], [0, 0, 1], [0, 0, 1]]), 
                           np.ones((3, 1)))
    
    # define location
    loc_1 = Location('S1', inv_2D, [trans_2D], dynamics_2D)
    loc_2 = Location('S2', inv_3D, [trans_3D], dynamics_3D)
    
    # init hybrid automaton
    HA = HybridAutomaton([loc_1, loc_1])
    # init with name
    HA = HybridAutomaton('HA', [loc_1, loc_1])
    
    assert HA is not None, "HybridAutomaton should be created"


def test_hybridAutomaton_hybridAutomaton_03_wrong_initializations():
    """
    TRANSLATED TEST - Wrong initializations test
    """
    # invariant
    inv_2D = Polytope(np.array([[-1, 0]]), np.array([[0]]))
    inv_3D = Polytope(np.array([[-1, 0, 0]]), np.array([[0]]))
    
    # transition
    guard = Polytope(np.array([[0, 1]]), np.array([[0]]),
                     np.array([[-1, 0]]), np.array([[0]]))
    reset = LinearReset(np.array([[1, 0], [0, -0.75]]))
    trans_2D = Transition(guard, reset, 2)
    guard3D = Polytope(np.array([[0, 0, 1]]), np.array([[0]]),
                       np.array([[-1, 0, 0]]), np.array([[0]]))
    reset3D = LinearReset.eye(3)
    trans_3D = Transition(guard3D, reset3D, 1)
    
    # flow equation
    dynamics_2D = LinearSys('linearSys', np.array([[0, 1], [0, 0]]), 
                           np.array([[0], [0]]), np.array([[0], [-9.81]]))
    # MATLAB: linearSys([0,1,0;0,0,1;0,0,1],1) - scalar B=1 means 3x1 matrix of ones
    # In Python, we need to pass a proper B matrix with correct dimensions
    dynamics_3D = LinearSys('linearSys', np.array([[0, 1, 0], [0, 0, 1], [0, 0, 1]]), 
                           np.ones((3, 1)))
    
    # define location
    loc_1 = Location('S1', inv_2D, [trans_2D], dynamics_2D)
    loc_2 = Location('S2', inv_3D, [trans_3D], dynamics_3D)
    
    # wrong name
    with pytest.raises((CORAerror, ValueError, TypeError)):
        HybridAutomaton(2, [loc_1, loc_1])  # wrong name type
    
    # dimensions of reset functions do not match
    with pytest.raises((CORAerror, ValueError)):
        HybridAutomaton([loc_1, loc_2])  # different dimensions
    
    # too many input arguments
    with pytest.raises((CORAerror, ValueError, TypeError)):
        HybridAutomaton('HA', loc_1, loc_1)  # too many args
    
    # not a location array
    with pytest.raises((CORAerror, ValueError, TypeError)):
        HybridAutomaton({'loc1': loc_1, 'loc2': loc_2})  # dict instead of list

