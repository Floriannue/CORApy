"""
hybridAutomaton - constructor for class hybridAutomaton

Syntax:
    HA = hybridAutomaton()
    HA = hybridAutomaton(loc)
    HA = hybridAutomaton(name,loc)

Inputs:
    loc - location-array storing location objects

Outputs:
    name - name of automaton
    HA - generated hybridAutomaton object

Example:
    % invariant
    inv = polytope([-1,0],0);
 
    % transition
    guard = polytope([0,1],0,[-1,0],0);
    reset = linearReset([1,0;0,-0.75]);
    trans(1) = transition(guard,reset,1);

    % flow equation
    dynamics = linearSys([0,1;0,0],[0;0],[0;-9.81]);

    % define location
    loc(1) = location('S1',inv,trans,dynamics);

    % instantiate hybrid automaton
    HA = hybridAutomaton('bouncingball',loc);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: parallelHybridAutomaton, location, transition

Authors:       Matthias Althoff, Mark Wetzlinger
Written:       03-May-2007 
Last update:   16-June-2022 (MW, add checks for object properties)
               21-June-2023 (MW, add internal properties, restructure)
               15-October-2024 (MW, add name property)
               16-October-2024 (TL, renames dim to nrOfStates)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, Optional, List, Union
import numpy as np
from cora_python.hybridDynamics.hybridDynamics.hybridDynamics import HybridDynamics
from cora_python.g.functions.matlab.validate.check.assertNarginConstructor import assertNarginConstructor
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class HybridAutomaton(HybridDynamics):
    """
    Hybrid automaton class
    
    A hybrid automaton consists of multiple locations, each with:
    - An invariant set
    - Transitions to other locations
    - Continuous dynamics
    
    Properties:
        name: Name of automaton
        location: Array of location objects
        nrOfDims: Number of states of each location
        nrOfInputs: Number of inputs for each location
        nrOfOutputs: Number of outputs for each location
        nrOfDisturbances: Number of disturbances for each location
        nrOfNoises: Number of noises for each location
    """
    
    def __init__(self, *args, **kwargs):
        """
        Constructor for hybridAutomaton
        
        Args:
            *args: Variable arguments:
                - hybridAutomaton(): Empty automaton
                - hybridAutomaton(loc): Automaton with locations
                - hybridAutomaton(name, loc): Automaton with name and locations
                - hybridAutomaton(other_HA): Copy constructor
        """
        super().__init__()
        
        # 0. empty
        assertNarginConstructor([0, 1, 2], len(args))
        if len(args) == 0:
            self.name = ''
            self.location = []
            # MATLAB: numeric properties default to [] (empty array)
            self.nrOfDims = np.array([])
            self.nrOfInputs = np.array([])
            self.nrOfOutputs = np.array([])
            self.nrOfDisturbances = np.array([])
            self.nrOfNoises = np.array([])
            # Legacy properties
            self.dim = np.array([])
            self.nrOfStates = np.array([])
            return
        
        # 1. copy constructor
        if len(args) == 1 and isinstance(args[0], HybridAutomaton):
            other = args[0]
            self.name = other.name
            self.location = other.location
            self.nrOfDims = other.nrOfDims
            self.nrOfInputs = other.nrOfInputs
            self.nrOfOutputs = other.nrOfOutputs
            self.nrOfDisturbances = other.nrOfDisturbances
            self.nrOfNoises = other.nrOfNoises
            # Legacy properties
            self.dim = other.dim if hasattr(other, 'dim') else other.nrOfDims
            self.nrOfStates = other.nrOfStates if hasattr(other, 'nrOfStates') else other.nrOfDims
            return
        
        # 2. parse input arguments: varargin -> vars
        name, locs = _aux_parseInputArgs(*args)
        
        # 3. check correctness of input arguments
        _aux_checkInputArgs(name, locs, len(args))
        
        # 4. compute internal properties
        states, inputs, outputs, dists, noises = _aux_computeProperties(locs)
        
        # 5. assign properties
        self.name = name
        # Reshape to column vector (list of locations)
        if isinstance(locs, (list, tuple, np.ndarray)):
            self.location = list(locs) if not isinstance(locs, np.ndarray) else locs.tolist()
        else:
            self.location = [locs]
        self.nrOfDims = states
        self.nrOfInputs = inputs
        self.nrOfOutputs = outputs
        self.nrOfDisturbances = dists
        self.nrOfNoises = noises
        # Legacy properties
        self.dim = states
        self.nrOfStates = states


# Auxiliary functions -----------------------------------------------------

def _aux_parseInputArgs(*varargin):
    """Parse input arguments for hybridAutomaton constructor"""
    
    name = 'hybridAutomaton'
    if len(varargin) == 1:
        locs = varargin[0]
    elif len(varargin) == 2:
        name, locs = varargin
    
    return name, locs


def _aux_checkInputArgs(name: str, locs: Any, n_in: int) -> None:
    """Check correctness of input arguments"""
    
    # Import here to avoid circular dependencies
    from cora_python.g.macros.CHECKS_ENABLED import CHECKS_ENABLED
    from cora_python.hybridDynamics.location.location import Location
    from cora_python.contSet.contSet.contSet import ContSet
    
    if CHECKS_ENABLED and n_in > 0:
        inputArgsCheck([
            [name, 'att', ['char', 'string']],
            [locs, 'att', ['location', 'list']]  # list for Python array
        ])
        
        if not isinstance(locs, (list, tuple, np.ndarray)) and not isinstance(locs, Location):
            if isinstance(locs, list):
                # Check if it's a list of locations
                if len(locs) > 0 and not isinstance(locs[0], Location):
                    raise CORAerror('CORA:wrongInputInConstructor',
                                  'Locations have to be a location object array.')
            else:
                raise CORAerror('CORA:wrongInputInConstructor',
                              'Locations have to be a location object array.')
        
        # Convert to list if single location
        if isinstance(locs, Location):
            locs = [locs]
        elif isinstance(locs, np.ndarray):
            locs = locs.tolist()
        elif isinstance(locs, tuple):
            locs = list(locs)
        
        # number of locations
        numLoc = len(locs)
        
        for i in range(numLoc):
            loc = locs[i]
            
            # 1. invariant of each location must have same dimension as
            # flow equation of that same location (unless empty)
            if loc.invariant is not None:
                inv_dim = loc.invariant.dim() if hasattr(loc.invariant, 'dim') else 0
                flow_dim = loc.contDynamics.nr_of_dims if hasattr(loc.contDynamics, 'nr_of_dims') else loc.contDynamics.nrOfDims
                if inv_dim != flow_dim:
                    raise CORAerror('CORA:wrongInputInConstructor',
                                  'Ambient dimension of invariant has to match state dimension of flow.')
            
            # 2. guard set of each transition of a location must have
            # same dimension as flow equation of that same location
            for j in range(len(loc.transition)):
                trans = loc.transition[j]
                if hasattr(trans, 'guard') and trans.guard is not None:
                    if not isinstance(trans.guard, (int, float, np.ndarray)):
                        guard_dim = trans.guard.dim() if hasattr(trans.guard, 'dim') else 0
                        flow_dim = loc.contDynamics.nr_of_dims if hasattr(loc.contDynamics, 'nr_of_dims') else loc.contDynamics.nrOfDims
                        if guard_dim != flow_dim:
                            raise CORAerror('CORA:wrongInputInConstructor',
                                          'Ambient dimension of guard set has to match state dimension of flow.')
            
            # 3. target of each transition must be <= number of locations
            for j in range(len(loc.transition)):
                trans = loc.transition[j]
                if hasattr(trans, 'target'):
                    target = trans.target
                    if isinstance(target, (list, np.ndarray)):
                        if np.any(np.array(target) > numLoc):
                            raise CORAerror('CORA:wrongInputInConstructor',
                                          'Targets exceed number of locations.')
                    elif isinstance(target, (int, float)) and target > numLoc:
                        raise CORAerror('CORA:wrongInputInConstructor',
                                      'Targets exceed number of locations.')
            
            # 4. output dimension of reset function must have same
            # dimension as flow equation of target dimension
            for j in range(len(loc.transition)):
                trans = loc.transition[j]
                if hasattr(trans, 'reset') and trans.reset is not None:
                    if hasattr(trans.reset, 'postStateDim'):
                        reset_post_dim = trans.reset.postStateDim
                        target = trans.target
                        if isinstance(target, (list, np.ndarray)):
                            target_idx = target[0] - 1  # Convert to 0-based
                        else:
                            target_idx = int(target) - 1  # Convert to 0-based
                        if 0 <= target_idx < numLoc:
                            target_loc = locs[target_idx]
                            target_flow_dim = target_loc.contDynamics.nr_of_dims if hasattr(target_loc.contDynamics, 'nr_of_dims') else target_loc.contDynamics.nrOfDims
                            if reset_post_dim != target_flow_dim:
                                raise CORAerror('CORA:wrongInputInConstructor',
                                              'Output dimension of reset function has to match the state dimension of the flow equation of the target location.')
            
            # 5. no duplicates in synchronization labels of a location
            syncLabelList = []
            for j in range(len(loc.transition)):
                trans = loc.transition[j]
                if hasattr(trans, 'syncLabel') and trans.syncLabel:
                    syncLabel = trans.syncLabel
                    if syncLabel in syncLabelList:
                        raise CORAerror('CORA:wrongInputInConstructor',
                                      'Each synchronization label may only be used in one outgoing transition per location.')
                    syncLabelList.append(syncLabel)
            
            # 6. unless synchronization labels differ, no more than one
            # instant transition per location allowed
            from cora_python.contSet.fullspace.fullspace import Fullspace
            emptyGuardSets = 0
            for j in range(len(loc.transition)):
                trans = loc.transition[j]
                if hasattr(trans, 'guard') and isinstance(trans.guard, Fullspace):
                    if not (hasattr(trans, 'syncLabel') and trans.syncLabel):
                        emptyGuardSets += 1
                if emptyGuardSets > 1:
                    raise CORAerror('CORA:wrongInputInConstructor',
                                  'Only one instant transition (guard = []) per location allowed, unless synchronization labels differ.')


def _aux_computeProperties(locs: List[Any]) -> tuple:
    """
    Compute the number of states, inputs, outputs, disturbances, and noises
    for each location
    """
    
    # loop over flows of all locations
    states = []
    inputs = []
    outputs = []
    dists = []
    noises = []
    
    for loc in locs:
        sys = loc.contDynamics
        states.append(sys.nr_of_dims if hasattr(sys, 'nr_of_dims') else sys.nrOfDims)
        inputs.append(sys.nr_of_inputs if hasattr(sys, 'nr_of_inputs') else sys.nrOfInputs)
        outputs.append(sys.nr_of_outputs if hasattr(sys, 'nr_of_outputs') else sys.nrOfOutputs)
        dists.append(sys.nr_of_disturbances if hasattr(sys, 'nr_of_disturbances') else sys.nrOfDisturbances)
        noises.append(sys.nr_of_noises if hasattr(sys, 'nr_of_noises') else sys.nrOfNoises)
    
    return np.array(states), np.array(inputs), np.array(outputs), np.array(dists), np.array(noises)

