"""
synchronize - synchronizes multiple transitions: the guard sets are
   intersected, the reset functions are combined into a single function,
   and the individual targets are combined to a joint updated target

TRANSLATED FROM: cora_matlab/hybridDynamics/@transition/synchronize.m

Syntax:
    trans = synchronize(transList,idStates,locID,compIdx,stateBinds,inputBinds,flowList)

Inputs:
    transList - list/array of transition objects
    idStates - states whose reset is mapped by identity
    locID - IDs of the currently active locations
    compIdx - components corresponding to elements in transList
    stateBinds - states of the high-dimensional space that correspond to
                   the states of the low-dimensional reset object
    inputBinds - connections of input to global input/outputs of other
                 components
    flowList - list of flow equations for each component

Outputs:
    trans - resulting transition object

Example:
    -

Authors:       Mark Wetzlinger (MATLAB)
Written:       10-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, List, Union, Optional
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from .transition import Transition


def synchronize(transList: List['Transition'], idStates: Union[np.ndarray, List[int]], 
                locID: np.ndarray, compIdx: np.ndarray, 
                stateBinds: List[Union[np.ndarray, List[int]]],
                inputBinds: List[Union[np.ndarray, List]],
                flowList: List) -> 'Transition':
    """
    Synchronizes multiple transitions: the guard sets are intersected,
    the reset functions are combined into a single function, and the
    individual targets are combined to a joint updated target.
    
    Args:
        transList: list of transition objects
        idStates: states whose reset is mapped by identity (0-based indices)
        locID: IDs of the currently active locations
        compIdx: components corresponding to elements in transList (0-based indices)
        stateBinds: list of state indices for each component (0-based)
        inputBinds: list of input bind arrays for each component
        flowList: list of flow equations for each component
    
    Returns:
        Transition object with synchronized guard, reset, and target
    """
    from .transition import Transition
    from cora_python.contSet.contSet.dim import dim
    from cora_python.contSet.fullspace import Fullspace
    from cora_python.contSet.contSet.and_ import and_
    
    if len(transList) == 0:
        raise CORAerror('CORA:wrongValue', 'first', 'At least one transition is required.')
    
    # Ensure that...
    # 1. guard sets are of same dimension
    # 2. reset functions are of same dimension
    # 3. target vectors are of equal size
    
    # Synchronize guards by intersection
    # MATLAB: guard = fullspace(dim(transList(1).guard));
    guard_dim = dim(transList[0].guard)
    guard = Fullspace(guard_dim)
    
    # MATLAB: for i=1:length(transList)
    for i in range(len(transList)):
        # MATLAB: guard = and_(guard,transList(i).guard,'exact');
        guard = and_(guard, transList[i].guard, 'exact')
    
    # Synchronize resets (without resolution of inputs), but they must all be
    # of the same class, so we may need to convert before synchronization
    resets = _aux_convertResets(transList)
    
    # Import reset synchronize methods
    from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
    from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
    
    # Check if all resets are of the same type
    if all(isinstance(r, LinearReset) for r in resets):
        # All linear resets - use LinearReset.synchronize (static method)
        reset_sync = LinearReset.synchronize(resets, idStates)
        # Note: number of total cells in outputEq must equal reset.inputDim
        # MATLAB: reset = resolve(reset_sync,flowList,stateBinds,inputBinds);
        # resolve is an instance method
        reset = reset_sync.resolve(flowList, stateBinds, inputBinds)
    elif all(isinstance(r, NonlinearReset) for r in resets):
        # All nonlinear resets - use NonlinearReset.synchronize (static method)
        # TODO: Implement NonlinearReset.synchronize and resolve
        try:
            # Try to use NonlinearReset.synchronize if it exists
            if hasattr(NonlinearReset, 'synchronize'):
                reset_sync = NonlinearReset.synchronize(resets, idStates)
            else:
                from cora_python.hybridDynamics.nonlinearReset.synchronize import synchronize as nonlinReset_synchronize
                reset_sync = nonlinReset_synchronize(resets, idStates)
            # resolve is an instance method
            reset = reset_sync.resolve(flowList, stateBinds, inputBinds)
        except (ImportError, AttributeError):
            raise CORAerror('CORA:notImplemented',
                           'NonlinearReset.synchronize and NonlinearReset.resolve are not yet implemented.')
    else:
        raise CORAerror('CORA:notSupported',
                       'Cannot synchronize mixed reset types. All resets must be linear or all nonlinear.')
    
    # Compose the target vector by overriding all entries in the currently
    # location ID vector for which there is a transition
    # MATLAB: target = locID;
    target = locID.copy() if isinstance(locID, np.ndarray) else np.array(locID)
    
    # MATLAB: for i=1:length(compIdx)
    for i in range(len(compIdx)):
        # MATLAB: target(compIdx) = transList(i).target;
        # compIdx is 0-based in Python, but MATLAB uses 1-based
        # If compIdx contains MATLAB 1-based indices, convert to 0-based
        comp_idx = int(compIdx[i])
        # Check if it looks 1-based (>= 1) or 0-based
        if comp_idx > 0:
            # Assume 1-based, convert to 0-based
            comp_idx_py = comp_idx - 1
        else:
            comp_idx_py = comp_idx
        
        # Ensure target is numpy array and has correct shape
        if not isinstance(target, np.ndarray):
            target = np.array(target)
        if target.ndim == 0:
            target = np.array([target])
        
        # Set target for this component
        if comp_idx_py < len(target):
            target[comp_idx_py] = transList[i].target
    
    # Instantiate resulting transition (no synchronization label anymore)
    trans = Transition(guard, reset, target)
    
    return trans


def _aux_convertResets(transList: List['Transition']) -> List:
    """
    Auxiliary function to convert resets to the same type if needed.
    
    Args:
        transList: list of transition objects
    
    Returns:
        List of reset objects, all of the same type
    """
    from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
    from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
    
    # Check which ones are linearReset and which are nonlinearReset objects
    isLinearReset = [isinstance(trans.reset, LinearReset) for trans in transList]
    isNonlinearReset = [isinstance(trans.reset, NonlinearReset) for trans in transList]
    
    if all(isLinearReset) or all(isNonlinearReset):
        # All of the same type
        return [trans.reset for trans in transList]
    
    # Convert all linear reset functions to nonlinear reset functions: since
    # the order of reset functions does not matter (they must have been
    # correctly lifted to the same dimension with correct binding before),
    # we first gather all nonlinear reset functions and then append the remaining
    # ones (note: isNonlinearReset has at least one true element)
    resets = [transList[i].reset for i in range(len(transList)) if isNonlinearReset[i]]
    resets_converted = [transList[i].reset.nonlinearReset() for i in range(len(transList)) if isLinearReset[i]]
    resets = resets + resets_converted
    
    return resets

