"""
reach - computes the reachable set for linear systems

Syntax:
    R = reach(linsys, params)
    R = reach(linsys, params, options)
    [R, res] = reach(linsys, params, options, spec)

Inputs:
    linsys - continuous system object
    params - model parameters
    options - options for the computation of reachable sets
    spec - object of class specification 

Outputs:
    R - object of class reachSet storing the reachable set
    res - true/false whether specifications are satisfied

Example:
    -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 26-June-2019 (MATLAB)
Last update: 08-October-2019 (MATLAB)
             23-April-2020 (added params) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from cora_python.g.functions.matlab.validate.preprocessing.set_default_values import set_default_values
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.classes.reachSet import ReachSet
from cora_python.g.classes.taylorLinSys import TaylorLinSys
from cora_python.contSet.zonotope import Zonotope

from .private.priv_reach_adaptive import priv_reach_adaptive
from .private.priv_reach_standard import priv_reach_standard
from .private.priv_reach_wrappingfree import priv_reach_wrappingfree


def reach(linsys, params: Dict[str, Any], *args) -> Union['ReachSet', Tuple['ReachSet', bool]]:
    """
    Computes the reachable set for linear systems
    
    Args:
        linsys: LinearSys object
        params: Model parameters dictionary
        *args: Variable arguments (options, spec)
        
    Returns:
        R: ReachSet object storing the reachable set
        res: (optional) True/false whether specifications are satisfied
    """
    # Parse input arguments
    options, spec = set_default_values([{'linAlg': 'adaptive'}, None], list(args))
    
    # Options preprocessing
    params, options = _validateOptions(linsys, params, options)
    
    specLogic = None
    if spec is not None:
        # TODO: Implement specification handling
        # spec, specLogic = splitLogic(spec)
        # if spec is not None:
        #     options['specification'] = spec
        pass
    
    # Hybrid systems: if invariant is empty set (used to model instant
    # transitions), exit immediately with only start set as reachable set
    # same goes for tStart = tFinal, which may occur in hybrid systems
    if (abs(params['tStart'] - params['tFinal']) < 1e-9 or 
        ('specification' in options and options['specification'] is not None and
         len(options['specification']) > 0 and 
         options['specification'][0].get('type') == 'invariant' and
         _representsa_emptySet(options['specification'][0].get('set')))):
        
        timePoint = {
            'set': [params['R0']], 
            'time': [params['tStart']]
        }
        timeInt = None
        res = False
    else:
        # Initialize taylorLinSys helper property for algorithms below
        if not hasattr(linsys, 'taylor') or linsys.taylor is None:
            linsys.taylor = TaylorLinSys(linsys.A)
        
        # Decide which reach function to execute by options.linAlg
        if options['linAlg'] == 'adaptive':
            timeInt, timePoint, res, savedata = priv_reach_adaptive(linsys, params, options)
            # ToDo: Handle savedata for subsequent runs, e.g. options['savedata'] = savedata
        else:
            # All below, const. time step sizes
            if options['linAlg'] == 'standard':
                timeInt, timePoint, res = priv_reach_standard(linsys, params, options)
            elif options['linAlg'] == 'wrapping-free':
                timeInt, timePoint, res = priv_reach_wrappingfree(linsys, params, options)
            elif options['linAlg'] == 'fromStart':
                # TODO: Implement fromStart algorithm
                raise CORAerror('CORA:notImplemented', 
                               'fromStart reachability algorithm not yet implemented')
            elif options['linAlg'] == 'decomp':
                # TODO: Implement decomp algorithm
                raise CORAerror('CORA:notImplemented', 
                               'decomp reachability algorithm not yet implemented')
            elif options['linAlg'] == 'krylov':
                # TODO: Implement krylov algorithm
                raise CORAerror('CORA:notImplemented', 
                               'krylov reachability algorithm not yet implemented')
            else:
                raise CORAerror('CORA:wrongFieldValue', 'options.linAlg',
                               ['standard', 'wrapping-free', 'adaptive', 'fromStart', 'decomp', 'krylov'])
            
            # Error vector (initial set: no error; error not computed -> NaN)
            timePoint['error'] = [0] + [np.nan] * (len(timePoint['set']) - 1)
            if timeInt is not None and 'set' in timeInt:
                timeInt['error'] = [np.nan] * len(timeInt['set'])
    
    # Delete all helper variables
    linsys.taylor = None
    
    # Create object of class reachSet
    R = ReachSet.initReachSet(timePoint, timeInt)
    
    # Check temporal logic specifications
    if res and specLogic is not None:
        # TODO: Implement specification checking
        # res = check(specLogic, R)
        # options['specification'] = spec
        pass
    
    # Return based on number of expected outputs
    if len(args) >= 2:  # spec was provided, return both R and res
        return R, res
    else:
        return R


def _validateOptions(linsys, params: Dict[str, Any], options: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Validate and set default values for parameters and options
    
    Args:
        linsys: LinearSys object
        params: Model parameters
        options: Computation options
        
    Returns:
        Validated params and options
    """
    # Set default values for params
    if 'tStart' not in params:
        params['tStart'] = 0.0
    
    if 'tFinal' not in params:
        raise CORAerror('CORA:specialError', 'Final time tFinal must be specified')
    
    if 'R0' not in params:
        raise CORAerror('CORA:specialError', 'Initial set R0 must be specified')
    
    # Set default values for options based on algorithm
    if options['linAlg'] in ['standard', 'wrapping-free', 'fromStart', 'decomp', 'krylov']:
        # Constant time step algorithms
        if 'timeStep' not in options:
            options['timeStep'] = 0.01
        
        if 'taylorTerms' not in options:
            options['taylorTerms'] = 4
        
        if 'zonotopeOrder' not in options:
            options['zonotopeOrder'] = 50
        
        if 'reductionTechnique' not in options:
            options['reductionTechnique'] = 'girard'
        
        if 'verbose' not in options:
            options['verbose'] = 0
    
    # Set default input set if not provided
    if 'U' not in params:
        params['U'] = Zonotope.origin(linsys.nr_of_inputs)
    
    # Set default input trajectory if not provided
    if 'uTrans' not in params:
        params['uTrans'] = np.zeros((linsys.nr_of_inputs, 1))
    
    # Set default disturbance set if not provided
    if 'W' not in params:
        params['W'] = Zonotope.origin(linsys.nr_of_disturbances)
    
    # Set default noise set if not provided
    if 'V' not in params:
        params['V'] = Zonotope.origin(linsys.nr_of_outputs)
    
    return params, options


def _representsa_emptySet(set_obj) -> bool:
    """Check if a set represents an empty set"""
    if set_obj is None:
        return True
    if hasattr(set_obj, 'representsa_'):
        return set_obj.representsa_('emptySet')
    return False 