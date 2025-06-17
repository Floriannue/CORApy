"""
simulateRandom - performs several random simulation of the system

This function performs several random simulations of the system. It can be set 
how many simulations should be performed, what percentage of initial states 
should start at vertices of the initial set, what percentage of inputs should 
be chosen from vertices of the input set, and of how many piecewise constant 
parts the input is constructed.

Authors: Florian Nüssel (Python implementation)
Date: 2025-01-08
"""

from typing import Dict, Any, Optional, List, Union
import numpy as np

# Use absolute import to avoid relative import issues
try:
    from cora_python.g.classes import SimResult
except ImportError:
    try:
        # Fallback for when running as script
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
        from g.classes import SimResult
    except ImportError:
        # Final fallback with relative import
        from ...g.classes import SimResult


def simulateRandom(sys, params: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> List[SimResult]:
    """
    Performs several random simulation of the system
    
    Syntax:
        simRes = simulateRandom(sys, params)
        simRes = simulateRandom(sys, params, options)
    
    Args:
        sys: contDynamics object
        params: System parameters
            - tStart: initial time (default: 0)
            - tFinal: final time
            - R0: initial set specified by one of the set representations
            - U: input set (optional)
            - W: disturbance set (optional)
            - V: sensor noise set (optional)
        options: Settings for random simulation
            - type: sampling method ('standard' (default), 'gaussian', 'rrt', 'constrained')
            - points: number of simulation runs (default: 1)
            
            For type = 'standard':
            - fracVert: fraction of initial states starting from vertices (default: 0.5)
            - fracInpVert: fraction of input values taken from vertices (default: 0.5)
            - nrConstInp: number of piecewise-constant input segments (default: 1)
            
            For type = 'gaussian':
            - p_conf: probability that a value is within the set (default: 0.99)
            - nrConstInp: number of piecewise-constant input segments (default: 1)
    
    Returns:
        simRes: List of SimResult objects storing time and states of simulated trajectories
    
    Example:
        # System dynamics
        sys = LinearSys([[-0.7, -2], [2, -0.7]], [[1], [1]], [[-2], [-1]])
        
        # Parameters
        params = {
            'tFinal': 5,
            'R0': Zonotope([2, 2], 0.1 * np.eye(2)),
            'U': Zonotope(0, 0.1)
        }
        
        # Simulation settings
        options = {
            'points': 7,
            'fracVert': 0.5,
            'fracInpVert': 1,
            'nrConstInp': 10
        }
        
        # Random simulation
        simRes = simulateRandom(sys, params, options)
    """
    
    # Set default options
    if options is None:
        options = {}
    
    # Input argument validation
    params, options = _validateOptions(sys, params, options)
    
    # Call private simulation function based on type
    sim_type = options.get('type', 'standard')
    
    if sim_type == 'standard':
        simRes = _priv_simulateStandard(sys, params, options)
    elif sim_type == 'gaussian':
        simRes = _priv_simulateGaussian(sys, params, options)
    elif sim_type == 'rrt':
        simRes = _priv_simulateRRT(sys, params, options)
    elif sim_type == 'constrained':
        simRes = _priv_simulateConstrainedRandom(sys, params, options)
    else:
        raise ValueError(f"Unknown simulation type: {sim_type}")
    
    return simRes


def _validateOptions(sys, params: Dict[str, Any], options: Dict[str, Any]) -> tuple:
    """Validate and set default values for parameters and options"""
    
    # Set default values for params
    if 'tStart' not in params:
        params['tStart'] = 0
    
    # Set default values for options
    if 'type' not in options:
        options['type'] = 'standard'
    
    if 'points' not in options:
        options['points'] = 1
    
    # Set defaults for fracVert and fracInpVert (used by all simulation types that call _priv_simulateStandard)
    if 'fracVert' not in options:
        options['fracVert'] = 0.5
    if 'fracInpVert' not in options:
        options['fracInpVert'] = 0.5
    
    # Type-specific defaults
    if options['type'] == 'standard':
        if 'nrConstInp' not in options:
            options['nrConstInp'] = 1
    
    elif options['type'] == 'gaussian':
        if 'p_conf' not in options:
            options['p_conf'] = 0.99
        if 'nrConstInp' not in options:
            options['nrConstInp'] = 1
    
    # Set default input set if not provided
    if 'U' not in params:
        # Create zero input set
        try:
            from cora_python.contSet.zonotope import Zonotope
        except ImportError:
            from contSet.zonotope import Zonotope
        params['U'] = Zonotope(np.zeros((sys.nr_of_inputs, 1)), np.zeros((sys.nr_of_inputs, 1)))
    
    # Set default disturbance and noise sets for output computation
    if hasattr(sys, 'C') and sys.C is not None and sys.C.size > 0:
        if 'W' not in params:
            try:
                from cora_python.contSet.zonotope import Zonotope
            except ImportError:
                from contSet.zonotope import Zonotope
            # Create disturbance set with correct dimension
            # Check if E matrix exists and has non-zero entries
            if (hasattr(sys, 'E') and sys.E is not None and sys.E.size > 0 and 
                not np.allclose(sys.E, 0)):
                w_dim = sys.E.shape[1]
            else:
                # No actual disturbance, create minimal set
                w_dim = 1
            params['W'] = Zonotope(np.zeros((w_dim, 1)), np.zeros((w_dim, 1)))
        # Determine correct V dimension for simulateRandom
        try:
            from cora_python.contSet.zonotope import Zonotope
        except ImportError:
            from contSet.zonotope import Zonotope
        
        # Check if F matrix exists and has non-zero entries
        if (hasattr(sys, 'F') and sys.F is not None and sys.F.size > 0 and 
            not np.allclose(sys.F, 0)):
            v_dim = sys.F.shape[1]  # Use F matrix column dimension
        else:
            # No actual noise, create minimal set
            v_dim = 1
        
        # Check if V exists and has correct dimension for simulateRandom
        if 'V' not in params or params['V'].dim() != v_dim:
            params['V'] = Zonotope(np.zeros((v_dim, 1)), np.zeros((v_dim, 1)))
    else:
        # For systems without output, still need disturbance set for simulation
        if 'W' not in params:
            try:
                from cora_python.contSet.zonotope import Zonotope
            except ImportError:
                from contSet.zonotope import Zonotope
            # Create disturbance set with correct dimension
            # Check if E matrix exists and has non-zero entries
            if (hasattr(sys, 'E') and sys.E is not None and sys.E.size > 0 and 
                not np.allclose(sys.E, 0)):
                w_dim = sys.E.shape[1]
            else:
                # No actual disturbance, create minimal set
                w_dim = 1
            params['W'] = Zonotope(np.zeros((w_dim, 1)), np.zeros((w_dim, 1)))
    
    # Create time vector for input switching
    if 'nrConstInp' in options:
        if isinstance(options['nrConstInp'], int):
            options['nrConstInp'] = [options['nrConstInp']]
        
        # Create time vector
        n_blocks = len(options['nrConstInp'])
        params['tu'] = np.linspace(params['tStart'], params['tFinal'], n_blocks + 1)
    
    return params, options


def _priv_simulateStandard(sys, params: Dict[str, Any], options: Dict[str, Any]) -> List[SimResult]:
    """Private function for standard random simulation"""
    
    # Trajectory tracking
    tracking = 'uTransVec' in params
    
    # Location for contDynamics always 0
    loc = 0
    
    # Output equation check
    comp_y = (hasattr(sys, 'C') and sys.C is not None and sys.C.size > 0)
    
    # Generate random initial points
    nr_extreme = int(np.ceil(options['points'] * options['fracVert']))
    nr_standard = options['points'] - nr_extreme
    
    X0_list = []
    
    if nr_extreme > 0:
        X0_extreme = _randPoint(params['R0'], nr_extreme, 'extreme')
        X0_list.append(X0_extreme)
    
    if nr_standard > 0:
        X0_standard = _randPoint(params['R0'], nr_standard, 'standard')
        X0_list.append(X0_standard)
    
    # Concatenate all initial points
    if X0_list:
        X0 = np.hstack(X0_list)
    else:
        # Fallback: generate one standard point
        X0 = _randPoint(params['R0'], 1, 'standard')
    
    # Initialize array of simResult objects
    res = []
    
    # Loop over all starting points in X0
    for r in range(options['points']):
        
        # Initialize cells for current simulation run r
        t_total = np.array([])
        x_total = np.empty((0, sys.nr_of_dims))
        if comp_y:
            y_total = np.empty((0, sys.nr_of_outputs))
        
        # Start of trajectory
        params_sim = params.copy()
        params_sim['x0'] = X0[:, r]
        
        # Loop over number of constant inputs per partial simulation run r
        for block in range(len(options['nrConstInp'])):
            
            # Update initial state
            if block > 0:
                params_sim['x0'] = x_temp[-1, :]
            
            # Update input
            if tracking:
                params_sim['uTrans'] = params['uTransVec'][:, block]
            else:
                params_sim['uTrans'] = np.zeros((sys.nr_of_inputs,))
            
            params_sim['tStart'] = params['tu'][block]
            params_sim['tFinal'] = params['tu'][block + 1]
            
            # Set input (random input from set of uncertainty)
            if r < options['points'] * options['fracInpVert']:
                u_rand = _randPoint(params['U'], options['nrConstInp'][block], 'extreme')
            else:
                u_rand = _randPoint(params['U'], options['nrConstInp'][block], 'standard')
            
            # Combine inputs (random input + tracking)
            params_sim['u'] = u_rand + params_sim['uTrans'].reshape(-1, 1)
            
            if comp_y:
                # Sample from disturbance set and sensor noise set
                if options['nrConstInp'][block] == 1:
                    params_sim['w'] = _randPoint(params['W'], 1)
                    params_sim['v'] = _randPoint(params['V'], 1)
                else:
                    params_sim['w'] = _randPoint(params['W'], options['nrConstInp'][block])
                    params_sim['v'] = _randPoint(params['V'], options['nrConstInp'][block] + 1)
            else:
                # Set default disturbance
                params_sim['w'] = _randPoint(params['W'], options['nrConstInp'][block])
                if 'v' in params_sim:
                    del params_sim['v']
            
            # Create simulation options without simulateRandom-specific keys
            sim_options = {}
            
            # Simulate dynamical system
            if comp_y:
                t_temp, x_temp, _, y_temp = sys.simulate(params_sim, sim_options)
            else:
                t_temp, x_temp = sys.simulate(params_sim, sim_options)
            
            # Append to previous values, overwrite first one
            if block == 0:
                t_total = t_temp
                x_total = x_temp
                if comp_y:
                    y_total = y_temp
            else:
                t_total = np.concatenate([t_total[:-1], t_temp])
                x_total = np.vstack([x_total[:-1, :], x_temp])
                if comp_y:
                    y_total = np.vstack([y_total[:-1, :], y_temp])
        
        # Append simResult object
        if comp_y:
            res.append(SimResult([x_total], [t_total], loc, [y_total]))
        else:
            res.append(SimResult([x_total], [t_total], loc))
    
    return res


def _priv_simulateGaussian(sys, params: Dict[str, Any], options: Dict[str, Any]) -> List[SimResult]:
    """Private function for Gaussian random simulation"""
    # For now, implement as standard simulation
    # TODO: Implement proper Gaussian sampling
    return _priv_simulateStandard(sys, params, options)


def _priv_simulateRRT(sys, params: Dict[str, Any], options: Dict[str, Any]) -> List[SimResult]:
    """Private function for RRT-based random simulation"""
    # For now, implement as standard simulation
    # TODO: Implement proper RRT sampling
    return _priv_simulateStandard(sys, params, options)


def _priv_simulateConstrainedRandom(sys, params: Dict[str, Any], options: Dict[str, Any]) -> List[SimResult]:
    """Private function for constrained random simulation"""
    # For now, implement as standard simulation
    # TODO: Implement proper constrained sampling
    return _priv_simulateStandard(sys, params, options)


def _randPoint(set_obj, N: int = 1, type_: str = 'standard') -> np.ndarray:
    """Generate random points from a set object"""
    
    # Check if the set has a randPoint method
    if hasattr(set_obj, 'randPoint'):
        return set_obj.randPoint(N, type_)
    
    # Try to import and use specific randPoint functions
    try:
        from cora_python.contSet.zonotope import randPoint as zonotope_randPoint
        from cora_python.contSet.interval import randPoint as interval_randPoint
        
        if hasattr(set_obj, 'c') and hasattr(set_obj, 'G'):
            # Likely a zonotope
            return zonotope_randPoint(set_obj, N, type_)
        elif hasattr(set_obj, 'inf') and hasattr(set_obj, 'sup'):
            # Likely an interval
            return interval_randPoint(set_obj, N, type_)
        else:
            raise ValueError(f"Unknown set type: {type(set_obj)}")
    
    except ImportError:
        raise ValueError(f"randPoint not implemented for set type: {type(set_obj)}") 