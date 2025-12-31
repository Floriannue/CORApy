"""
getDefaultValueParams - contains list of default values for params

Syntax:
    defValue = getDefaultValueParams(field,sys,params,options)

Inputs:
    field - struct field in params / options
    sys - object of system class
    params - struct containing model parameters
    options - struct containing algorithm parameters

Outputs:
    defValue - default value for given field

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: getDefaultValues

Authors:       Mark Wetzlinger
Written:       26-January-2021
Last update:   09-October-2023 (TL, split options/params)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Dict, List
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval


def getDefaultValueParams(field: str, sys: Any, params: Dict[str, Any], 
                         options: Dict[str, Any]) -> Any:
    """
    Get default value for a field in params
    
    Args:
        field: struct field in params
        sys: object of system class
        params: struct containing model parameters
        options: struct containing algorithm parameters
        
    Returns:
        defValue: default value for given field
    """
    
    # search for default value in params.<field>
    # MATLAB: switch field
    if field == 'tStart':
        defValue = 0
    elif field == 'finalLoc':
        defValue = aux_def_finalLoc(sys, params, options)
    elif field == 'U':
        defValue = aux_def_U(sys, params, options)
    elif field == 'u':
        defValue = aux_def_u(sys, params, options)
    elif field == 'tu':
        defValue = aux_def_tu(sys, params, options)
    elif field == 'y':
        defValue = aux_def_y(sys, params, options)
    elif field == 'W':
        defValue = aux_def_W(sys, params, options)
    elif field == 'V':
        defValue = aux_def_V(sys, params, options)
    elif field == 'inputCompMap':
        defValue = aux_def_inputCompMap(sys, params, options)
    else:
        # MATLAB: throw(CORAerror('CORA:specialError',...))
        raise CORAerror('CORA:specialError', f"There is no default value for params.{field}.")
    
    return defValue


# Auxiliary functions -----------------------------------------------------

def aux_def_finalLoc(sys: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Any:
    """
    Auxiliary function to get default finalLoc
    """
    
    val = None
    # MATLAB: if isa(sys,'hybridAutomaton')
    if hasattr(sys, '__class__') and 'hybridAutomaton' in sys.__class__.__name__.lower():
        val = 0
    # MATLAB: elseif isa(sys,'parallelHybridAutomaton')
    elif hasattr(sys, '__class__') and 'parallelHybridAutomaton' in sys.__class__.__name__.lower():
        # MATLAB: val = zeros(length(sys.components),1);
        val = np.zeros((len(sys.components), 1))
    # no assignment for contDynamics
    
    return val


def aux_def_U(sys: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Any:
    """
    Auxiliary function to get default U
    """
    
    val = None
    # MATLAB: if isa(sys,'contDynamics')
    if hasattr(sys, '__class__') and 'contDynamics' in sys.__class__.__name__.lower():
        # MATLAB: val = zonotope(zeros(sys.nrOfInputs,1));
        val = Zonotope(np.zeros((sys.nrOfInputs, 1)))
    # MATLAB: elseif isa(sys,'hybridAutomaton')
    elif hasattr(sys, '__class__') and 'hybridAutomaton' in sys.__class__.__name__.lower():
        # define for each location
        # MATLAB: locations = sys.location;
        locations = sys.location
        # MATLAB: numLoc = length(locations);
        numLoc = len(locations)
        # MATLAB: val = cell(numLoc,1);
        val = []
        # MATLAB: for i = 1:numLoc
        for i in range(numLoc):
            # MATLAB: nrInputs = locations(i).contDynamics.nrOfInputs;
            nrInputs = locations[i].contDynamics.nrOfInputs
            # MATLAB: val{i} = zonotope(zeros(max(1,nrInputs),1));
            val.append(Zonotope(np.zeros((max(1, nrInputs), 1))))
    # MATLAB: elseif isa(sys,'parallelHybridAutomaton')
    elif hasattr(sys, '__class__') and 'parallelHybridAutomaton' in sys.__class__.__name__.lower():
        # define for each component
        # MATLAB: numComps = length(sys.components);
        numComps = len(sys.components)
        # MATLAB: val = cell(numComps,1);
        val = []
        # MATLAB: for i=1:numComps
        for i in range(numComps):
            # define for each location
            # MATLAB: numLoc = length(sys.components(i).location);
            numLoc = len(sys.components[i].location)
            # MATLAB: val{i} = cell(numLoc,1);
            val_i = []
            # MATLAB: for j = 1:numLoc
            for j in range(numLoc):
                # MATLAB: nrInputs = sys.components(i).location(1).contDynamics.nrOfInputs;
                nrInputs = sys.components[i].location[0].contDynamics.nrOfInputs
                # MATLAB: val{i}{j} = zonotope(zeros(nrInputs,1));
                val_i.append(Zonotope(np.zeros((nrInputs, 1))))
            val.append(val_i)
    
    return val


def aux_def_u(sys: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Any:
    """
    Auxiliary function to get default u
    """
    
    val = None
    # MATLAB: if isa(sys,'contDynamics')
    if hasattr(sys, '__class__') and 'contDynamics' in sys.__class__.__name__.lower():
        # MATLAB: val = zeros(sys.nrOfInputs,1);
        val = np.zeros((sys.nrOfInputs, 1))
    # MATLAB: elseif isa(sys,'hybridAutomaton')
    elif hasattr(sys, '__class__') and 'hybridAutomaton' in sys.__class__.__name__.lower():
        # define for each location
        # MATLAB: locations = sys.location;
        locations = sys.location
        # MATLAB: numLoc = length(locations);
        numLoc = len(locations)
        # MATLAB: val = cell(numLoc,1);
        val = []
        # MATLAB: for i = 1:numLoc
        for i in range(numLoc):
            # MATLAB: subsys = locations(i).contDynamics;
            subsys = locations[i].contDynamics
            # MATLAB: val{i} = zeros(max(1,subsys.nrOfInputs),1);
            val.append(np.zeros((max(1, subsys.nrOfInputs), 1)))
    # MATLAB: elseif isa(sys,'parallelHybridAutomaton')
    elif hasattr(sys, '__class__') and 'parallelHybridAutomaton' in sys.__class__.__name__.lower():
        # define for each component
        # MATLAB: numComps = length(sys.components);
        numComps = len(sys.components)
        # MATLAB: val = cell(numComps,1);
        val = []
        # MATLAB: for i=1:numComps
        for i in range(numComps):
            # define for each location
            # MATLAB: numLoc = length(sys.components(i).location);
            numLoc = len(sys.components[i].location)
            # MATLAB: val{i} = cell(numLoc,1);
            val_i = []
            # MATLAB: for j = 1:numLoc
            for j in range(numLoc):
                # MATLAB: val{i}{j} = zeros(sys.nrOfInputs,1);
                val_i.append(np.zeros((sys.nrOfInputs, 1)))
            val.append(val_i)
    
    return val


def aux_def_tu(sys: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Any:
    """
    Auxiliary function to get default tu
    """
    
    val = None
    # MATLAB: if isa(sys,'contDynamics')
    if hasattr(sys, '__class__') and 'contDynamics' in sys.__class__.__name__.lower():
        # MATLAB: if isa(sys,'linearSysDT') || isa(sys,'nonlinearSysDT')
        is_linearSysDT = hasattr(sys, '__class__') and 'linearSysDT' in sys.__class__.__name__.lower()
        is_nonlinearSysDT = hasattr(sys, '__class__') and 'nonlinearSysDT' in sys.__class__.__name__.lower()
        
        if is_linearSysDT or is_nonlinearSysDT:
            # MATLAB: if size(params.u,2) == 1
            u = params.get('u', None)
            if u is not None and isinstance(u, np.ndarray) and u.shape[1] == 1:
                # constant input, only one step
                # MATLAB: val = params.tStart;
                val = params['tStart']
            else:
                # input trajectory, create tArray
                # MATLAB: val = (params.tStart:sys.dt:params.tFinal-sys.dt)';
                val = np.arange(params['tStart'], params['tFinal'] - sys.dt + sys.dt, sys.dt).reshape(-1, 1)
                # MATLAB: if isa(sys,'linearSysDT') && any(any(sys.D))
                if is_linearSysDT and hasattr(sys, 'D') and np.any(sys.D):
                    # MATLAB: val = (params.tStart:sys.dt:params.tFinal)';
                    val = np.arange(params['tStart'], params['tFinal'] + sys.dt, sys.dt).reshape(-1, 1)
        # MATLAB: elseif isa(sys,'linearSys')
        elif hasattr(sys, '__class__') and 'linearSys' in sys.__class__.__name__.lower():
            # create tArray based on input trajectory
            # MATLAB: steps = size(params.u,2);
            u = params.get('u', None)
            if u is not None:
                steps = u.shape[1]
                # MATLAB: if any(any(sys.D)) && steps > 1
                if hasattr(sys, 'D') and np.any(sys.D) and steps > 1:
                    steps = steps - 1
                # MATLAB: stepsize = (params.tFinal-params.tStart) / steps;
                stepsize = (params['tFinal'] - params['tStart']) / steps
                # MATLAB: val = (params.tStart:stepsize:params.tFinal-stepsize)';
                val = np.arange(params['tStart'], params['tFinal'] - stepsize + stepsize, stepsize).reshape(-1, 1)
                # MATLAB: if steps > 1 && any(any(sys.D))
                if steps > 1 and hasattr(sys, 'D') and np.any(sys.D):
                    # MATLAB: val = (params.tStart:stepsize:params.tFinal)';
                    val = np.arange(params['tStart'], params['tFinal'] + stepsize, stepsize).reshape(-1, 1)
        # handle all other system types
        else:  # isa(sys,'linParamSys') || isa(sys,'linProbSys')
            # MATLAB: if size(params.u,2) == 1
            u = params.get('u', None)
            if u is not None and isinstance(u, np.ndarray) and u.shape[1] == 1:
                # MATLAB: val = params.tStart;
                val = params['tStart']
            else:
                # MATLAB: steps = size(params.u,2);
                steps = u.shape[1]
                # MATLAB: stepsize = (params.tFinal-params.tStart) / steps;
                stepsize = (params['tFinal'] - params['tStart']) / steps
                # MATLAB: val = (params.tStart:stepsize:params.tFinal-stepsize)';
                val = np.arange(params['tStart'], params['tFinal'] - stepsize + stepsize, stepsize).reshape(-1, 1)
    
    # no assignment for hybridAutomaton / parallelHybridAutomaton
    
    return val


def aux_def_y(sys: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Any:
    """
    Auxiliary function to get default y
    """
    
    val = None
    # MATLAB: if isa(sys,'contDynamics')
    if hasattr(sys, '__class__') and 'contDynamics' in sys.__class__.__name__.lower():
        # MATLAB: val = zeros(sys.nrOfOutputs,1);
        val = np.zeros((sys.nrOfOutputs, 1))
    # no assignment for hybridAutomaton / parallelHybridAutomaton
    
    return val


def aux_def_W(sys: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Any:
    """
    Auxiliary function to get default W
    """
    
    val = None
    # MATLAB: if isa(sys,'contDynamics')
    if hasattr(sys, '__class__') and 'contDynamics' in sys.__class__.__name__.lower():
        # MATLAB: val = interval(zeros(sys.nrOfDisturbances,1));
        val = Interval(np.zeros((sys.nrOfDisturbances, 1)))
    # MATLAB: elseif isa(sys,'hybridAutomaton')
    elif hasattr(sys, '__class__') and 'hybridAutomaton' in sys.__class__.__name__.lower():
        # set for all locations
        # MATLAB: locations = sys.location;
        locations = sys.location
        # MATLAB: numLoc = length(locations);
        numLoc = len(locations)
        # MATLAB: val = cell(numLoc,1);
        val = []
        # MATLAB: for i = 1:numLoc
        for i in range(numLoc):
            # MATLAB: nrDists = locations(i).contDynamics.nrOfDisturbances;
            nrDists = locations[i].contDynamics.nrOfDisturbances
            # MATLAB: val{i} = interval(zeros(max(1,nrDists),1));
            val.append(Interval(np.zeros((max(1, nrDists), 1))))
    # MATLAB: elseif isa(sys,'parallelHybridAutomaton')
    elif hasattr(sys, '__class__') and 'parallelHybridAutomaton' in sys.__class__.__name__.lower():
        # set for all components
        # MATLAB: numComps = length(sys.components);
        numComps = len(sys.components)
        # MATLAB: val = cell(numComps,1);
        val = []
        # MATLAB: for i=1:numComps
        for i in range(numComps):
            # set for all locations
            # MATLAB: numLoc = length(sys.components(i).location);
            numLoc = len(sys.components[i].location)
            # MATLAB: val{i} = cell(numLoc,1);
            val_i = []
            # MATLAB: for j = 1:numLoc
            for j in range(numLoc):
                # MATLAB: nrDists = sys.components(i).location(1).contDynamics.nrOfDisturbances;
                nrDists = sys.components[i].location[0].contDynamics.nrOfDisturbances
                # MATLAB: val{i}{j} = interval(zeros(nrDists,1));
                val_i.append(Interval(np.zeros((nrDists, 1))))
            val.append(val_i)
    
    return val


def aux_def_V(sys: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Any:
    """
    Auxiliary function to get default V
    """
    
    val = None
    # MATLAB: if isa(sys,'contDynamics')
    if hasattr(sys, '__class__') and 'contDynamics' in sys.__class__.__name__.lower():
        # MATLAB: val = interval(zeros(sys.nrOfNoises,1));
        val = Interval(np.zeros((sys.nrOfNoises, 1)))
    # MATLAB: elseif isa(sys,'hybridAutomaton')
    elif hasattr(sys, '__class__') and 'hybridAutomaton' in sys.__class__.__name__.lower():
        # get for each location
        # MATLAB: locations = sys.location;
        locations = sys.location
        # MATLAB: numLoc = length(locations);
        numLoc = len(locations)
        # MATLAB: val = cell(numLoc,1);
        val = []
        # MATLAB: for i = 1:numLoc
        for i in range(numLoc):
            # MATLAB: nrNoises = locations(i).contDynamics.nrOfNoises;
            nrNoises = locations[i].contDynamics.nrOfNoises
            # MATLAB: val{i} = interval(zeros(max(1,nrNoises),1));
            val.append(Interval(np.zeros((max(1, nrNoises), 1))))
    # MATLAB: elseif isa(sys,'parallelHybridAutomaton')
    elif hasattr(sys, '__class__') and 'parallelHybridAutomaton' in sys.__class__.__name__.lower():
        # get for each component
        # MATLAB: numComps = length(sys.components);
        numComps = len(sys.components)
        # MATLAB: val = cell(numComps,1);
        val = []
        # MATLAB: for i=1:numComps
        for i in range(numComps):
            # get for each location
            # MATLAB: numLoc = length(sys.components(i).location);
            numLoc = len(sys.components[i].location)
            # MATLAB: val{i} = cell(numLoc,1);
            val_i = []
            # MATLAB: for j = 1:numLoc
            for j in range(numLoc):
                # MATLAB: nrNoises = sys.components(i).location(1).contDynamics.nrOfNoises;
                nrNoises = sys.components[i].location[0].contDynamics.nrOfNoises
                # MATLAB: val{i}{j} = interval(zeros(nrNoises,1));
                val_i.append(Interval(np.zeros((nrNoises, 1))))
            val.append(val_i)
    
    return val


def aux_def_inputCompMap(sys: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Any:
    """
    Auxiliary function to get default inputCompMap
    """
    
    val = None
    # MATLAB: if isa(sys,'parallelHybridAutomaton')
    if hasattr(sys, '__class__') and 'parallelHybridAutomaton' in sys.__class__.__name__.lower():
        # MATLAB: val = ones(sys.nrOfInputs,1);
        val = np.ones((sys.nrOfInputs, 1))
    # no assignment for contDynamics
    
    return val

