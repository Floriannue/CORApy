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

from typing import Dict, Any, Optional, List
import numpy as np

from cora_python.g.classes import SimResult
from cora_python.contSet.zonotope import Zonotope


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
    
    # Import private functions here to avoid circular imports
    from .private.priv_simulateStandard import priv_simulateStandard
    from .private.priv_simulateGaussian import priv_simulateGaussian
    from .private.priv_simulateRRT import priv_simulateRRT
    from .private.priv_simulateConstrainedRandom import priv_simulateConstrainedRandom
    
    if sim_type == 'standard':
        simRes = priv_simulateStandard(sys, params, options)
    elif sim_type == 'gaussian':
        simRes = priv_simulateGaussian(sys, params, options)
    elif sim_type == 'rrt':
        simRes = priv_simulateRRT(sys, params, options)
    elif sim_type == 'constrained':
        simRes = priv_simulateConstrainedRandom(sys, params, options)
    else:
        raise ValueError(f"Unknown simulation type: {sim_type}")
    
    class SimResultList(list):
        """A list that behaves like a SimResult for plotting"""
        
        def plot(self, *args, **kwargs):
            """Plot method for list of SimResult objects"""
            from cora_python.g.classes.simResult.plot import plot
            return plot(self, *args, **kwargs)
    
    return SimResultList(simRes)


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
        params['U'] = Zonotope(np.zeros((sys.nr_of_inputs, 1)), np.zeros((sys.nr_of_inputs, 1)))
    
    # Set default disturbance and noise sets for output computation
    if hasattr(sys, 'C') and sys.C is not None and sys.C.size > 0:
        if 'W' not in params:

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

raise NotImplementedError("check simresultlist here and in linearsys - does it make sense and works well together or is there a better way to do this?")