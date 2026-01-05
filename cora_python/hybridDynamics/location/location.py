"""
location - constructor of class location

Syntax:
    loc = location()
    loc = location(invSet,trans,sys)
    loc = location(name,invSet,trans,sys)

Inputs:
    name - name of the location
    invSet - invariant set
    trans - object-array containing all transitions
    sys - continuous dynamics

Outputs:
    loc - generated location object

Example:
    % name of location
    name = 'S1';

    % invariant
    inv = polytope([-1,0],0);
    
    % transition: guard set, reset function, target
    guard = polytope([0,1],0,[-1,0],0);
    reset = linearReset([1,0;0,-0.75]);
    trans = transition(guard,reset,2);

    % flow equation
    dynamics = linearSys([0,1;0,0],[0;0],[0;-9.81]);

    % define location
    loc = location(name,inv,trans,dynamics);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: hybridAutomaton, transition

Authors:       Matthias Althoff
Written:       02-May-2007 
Last update:   ---
Last revision: 14-October-2024 (MW, update to current constructor structure)
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, Optional, List, Union
import numpy as np
from cora_python.g.functions.matlab.validate.check.assertNarginConstructor import assertNarginConstructor
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.contSet.polytope.polytope import Polytope


class Location:
    """
    Location class for hybrid automata
    
    A location represents a discrete state in a hybrid automaton with:
    - An invariant set (where the system can remain)
    - Transitions to other locations
    - Continuous dynamics (flow equation)
    
    Properties:
        name: Name of the location
        invariant: Invariant set (contSet)
        transition: Array of transition objects
        contDynamics: Continuous dynamics (contDynamics object)
    """
    
    def __init__(self, *args, **kwargs):
        """
        Constructor for location
        
        Args:
            *args: Variable arguments:
                - location(): Empty location
                - location(invSet, trans, sys): Location with invariant, transitions, and dynamics
                - location(name, invSet, trans, sys): Location with name, invariant, transitions, and dynamics
                - location(other_location): Copy constructor
        """
        # 0. empty
        assertNarginConstructor([0, 1, 3, 4], len(args))
        if len(args) == 0:
            self.name = 'location'
            # MATLAB: invariant = []; transition = transition(); contDynamics = contDynamics();
            # MATLAB [] is a 2D empty array (0x0), not None
            self.invariant = np.empty((0, 0))  # MATLAB: invariant = [] (empty array)
            self.transition = []  # MATLAB: transition = transition() (empty array)
            self.contDynamics = None  # MATLAB: contDynamics = contDynamics() - object, None is OK
            return
        
        # 1. copy constructor
        if len(args) == 1 and isinstance(args[0], Location):
            other = args[0]
            self.name = other.name
            self.invariant = other.invariant
            self.transition = other.transition
            self.contDynamics = other.contDynamics
            return
        
        # 2. parse input arguments: varargin -> vars
        name, inv, trans, sys = _aux_parseInputArgs(*args)
        
        # 3. check correctness of input arguments
        _aux_checkInputArgs(name, inv, trans, sys, len(args))
        
        # 4. compute dependent properties
        name, inv, trans, sys = _aux_computeProperties(name, inv, trans, sys)
        
        # 5. assign properties
        self.name = name
        self.invariant = inv
        self.transition = trans
        self.contDynamics = sys


# Auxiliary functions -----------------------------------------------------

def _aux_parseInputArgs(*varargin):
    """Parse input arguments for location constructor"""
    from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
    
    # default properties (MATLAB: inv = [], trans = transition(), sys = contDynamics())
    name = 'location'
    # MATLAB [] is a 2D matrix (0x0), not 1D array
    inv = np.empty((0, 0))  # MATLAB: inv = []
    trans = []  # MATLAB: trans = transition() - will be transition array
    sys = None  # MATLAB: sys = contDynamics() - object, None is OK
    
    # parse arguments
    if len(varargin) == 3:
        inv, trans, sys = setDefaultValues([inv, trans, sys], list(varargin))
    elif len(varargin) == 4:
        name, inv, trans, sys = setDefaultValues([name, inv, trans, sys], list(varargin))
    
    # Convert None to empty arrays (MATLAB uses [] not None)
    # This handles cases where None is explicitly passed
    # MATLAB [] is a 2D matrix (0x0), not 1D array
    if inv is None:
        inv = np.empty((0, 0))
    
    return name, inv, trans, sys


def _aux_checkInputArgs(name: str, inv: Any, trans: Any, sys: Any, n_in: int) -> None:
    """Check correctness of input arguments"""
    
    # Import here to avoid circular dependencies
    from cora_python.g.macros.CHECKS_ENABLED import CHECKS_ENABLED
    
    if CHECKS_ENABLED and n_in > 0:
        inputArgsCheck([
            [name, 'att', ['char', 'string']],
            [inv, 'att', 'contSet'],
            [trans, 'att', ['transition', 'list']],  # list for Python array
            [sys, 'att', 'contDynamics']
        ])


def _aux_computeProperties(name: str, inv: Any, trans: Any, sys: Any) -> tuple:
    """Compute dependent properties"""
    
    # Import here to avoid circular dependencies
    from cora_python.hybridDynamics.transition.transition import Transition
    
    # loc.transition has to be an array of transition objects
    # MATLAB allows single transition object, which is treated as array
    if isinstance(trans, Transition):
        # Single transition - convert to list
        trans = [trans]
    elif not isinstance(trans, list):
        if isinstance(trans, (tuple, np.ndarray)):
            trans = list(trans)
        else:
            raise CORAerror('CORA:wrongInputInConstructor',
                          'Transitions have to be a transition object array.')
    
    # convert invariant sets to polytopes if possible (keep polytopes,
    # levelSet, and fullspace/emptySet invariants)
    from cora_python.contSet.fullspace.fullspace import Fullspace
    from cora_python.contSet.emptySet.emptySet import EmptySet
    from cora_python.contSet.levelSet.levelSet import LevelSet
    
    if inv is not None:
        if not (isinstance(inv, Fullspace) or isinstance(inv, Polytope) or
                isinstance(inv, LevelSet) or isinstance(inv, EmptySet)):
            inv = Polytope(inv)
    
    return name, inv, trans, sys

