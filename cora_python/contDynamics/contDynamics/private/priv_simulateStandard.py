"""
priv_simulateStandard - performs several random simulation of the system

This function performs several random simulations of the system. It can be set
how many simulations should be performed, what percentage of initial states
should start at vertices of the initial set, what percentage of inputs should
be chosen from vertices of the input set, and of how many different constant
inputs the input trajectory consists.

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 17-August-2016 (MATLAB)
Last update: 28-June-2021 (MATLAB)
Python translation: 2025
"""

from typing import Dict, Any, List
import numpy as np

from cora_python.g.classes import SimResult


def priv_simulateStandard(sys, params: Dict[str, Any], options: Dict[str, Any]) -> List[SimResult]:
    """
    Private function for standard random simulation
    
    Args:
        sys: contDynamics object
        params: Model parameters
        options: Settings for random simulation
        
    Returns:
        List[SimResult]: Simulation results
    """
    
    # Trajectory tracking
    tracking = 'uTransVec' in params
    
    # Location for contDynamics always 0
    loc = 0
    
    # Output equation only for linearSys and linearSysDT currently
    comp_y = ((sys.__class__.__name__ in ['LinearSys', 'LinearSysDT']) and 
              hasattr(sys, 'C') and sys.C is not None and sys.C.size > 0)
    
    # Generate random initial points
    nrExtreme = int(np.ceil(options['points'] * options['fracVert']))
    nrStandard = options['points'] - nrExtreme
    X0 = []
    
    if nrExtreme > 0:
        X0_extreme = _randPoint(params['R0'], nrExtreme, 'extreme')
        X0.append(X0_extreme)
    
    if nrStandard > 0:
        X0_standard = _randPoint(params['R0'], nrStandard, 'standard')
        X0.append(X0_standard)
    
    # Combine all initial points
    if len(X0) > 0:
        X0 = np.hstack(X0) if len(X0) > 1 else X0[0]
    else:
        X0 = np.zeros((sys.nr_of_dims, 0))
    
    # Initialize array of simResult objects
    res = []
    
    # Loop over all starting points in X0
    for r in range(options['points']):
        
        # Initialize cells for current simulation run r
        t_sim = []
        x_sim = []
        if comp_y:
            y_sim = []
        
        # Start of trajectory
        params_sim = params.copy()
        params_sim['x0'] = X0[:, r].flatten()  # 1D array for scipy.integrate.solve_ivp
        
        # Loop over number of constant inputs per partial simulation run r
        for block in range(len(options['nrConstInp'])):
            
            # Update initial state
            if block > 0:
                params_sim['x0'] = xTemp[-1, :].flatten()  # 1D array for scipy.integrate.solve_ivp
            
            # Update input
            if tracking:
                params_sim['uTrans'] = params['uTransVec'][:, block].reshape(-1, 1)
            
            params_sim['tStart'] = params['tu'][block]
            params_sim['tFinal'] = params['tu'][block + 1]
            
            # Set input (random input from set of uncertainty)
            if r < options['points'] * options['fracInpVert']:
                uRand = _randPoint(params['U'], options['nrConstInp'][block], 'extreme')
            else:
                uRand = _randPoint(params['U'], options['nrConstInp'][block], 'standard')
            
            # Combine inputs (random input + tracking)
            if 'uTrans' not in params_sim:
                params_sim['uTrans'] = np.zeros((sys.nr_of_inputs, 1))
            
            params_sim['u'] = uRand + params_sim['uTrans']
            
            if comp_y:
                # Sample from disturbance set and sensor noise set
                if options['nrConstInp'][block] == 1:
                    params_sim['w'] = _randPoint(params['W'], 1)
                    params_sim['v'] = _randPoint(params['V'], 1)
                else:
                    params_sim['w'] = _randPoint(params['W'], options['nrConstInp'][block])
                    params_sim['v'] = _randPoint(params['V'], options['nrConstInp'][block] + 1)
            
            # Note: for correct vector lengths in simulate, we require an
            # additional dummy entry in u and v (this is due to the evaluation
            # of the output equation at the end of the current [tStart,tFinal])
            # ONLY for linear systems, and only if there is a feedthrough matrix
            if (comp_y and hasattr(sys, 'D') and sys.D is not None and 
                np.any(sys.D != 0)):
                
                if params_sim['u'].shape[1] > 1:
                    dummy_u = np.ones((sys.nr_of_inputs, 1)) * (np.pi / 2)
                    params_sim['u'] = np.hstack([params_sim['u'], dummy_u])
                
                if 'v' in params_sim and params_sim['v'].shape[1] > 1:
                    dummy_v = np.ones((sys.nr_of_outputs, 1)) * (np.pi / 2)
                    params_sim['v'] = np.hstack([params_sim['v'], dummy_v])
            
            # Uncertain parameters
            if 'paramInt' in params:
                pInt = params['paramInt']
                if hasattr(pInt, 'inf') and hasattr(pInt, 'rad'):  # interval
                    params_sim['p'] = pInt.inf + 2 * pInt.rad * np.random.rand()
                else:
                    params_sim['p'] = pInt
            
            # Simulate dynamical system
            if sys.__class__.__name__ == 'nonlinearARX':
                params_sim['y_init'] = params_sim['x0'].reshape(sys.dim_y, -1)
                tTemp, xTemp, _, yTemp = sys.simulate(params_sim)
                xTemp = yTemp  # For nonlinearARX, x is the same as y
            elif comp_y:
                tTemp, xTemp, _, yTemp = sys.simulate(params_sim, options)
            else:
                tTemp, xTemp, _ = sys.simulate(params_sim, options)[:3]
            
            # Append to previous values, overwrite first one
            if block == 0:
                t_sim = tTemp.copy()
                x_sim = xTemp.copy()
                if comp_y:
                    y_sim = yTemp.copy()
            else:
                # Append, overwriting first point to avoid duplication
                t_sim = np.concatenate([t_sim[:-1], tTemp])
                x_sim = np.concatenate([x_sim[:-1], xTemp])
                if comp_y:
                    y_sim = np.concatenate([y_sim[:-1], yTemp])
        
        if comp_y and sys.__class__.__name__ in ['LinearSys', 'LinearSysDT']:
            # Final point of output trajectory uses different input and sensor noise
            ylast = _aux_outputTrajectoryEnd(sys, params_sim, x_sim)
            y_sim[-1, :] = ylast.flatten()
        
        # Append simResult object
        if comp_y:
            res.append(SimResult([x_sim], [t_sim], loc, [y_sim]))
        elif sys.__class__.__name__ == 'nonlinDASys':
            # Dimensions of algebraic variables in extended state vector
            dims_a = slice(sys.nr_of_dims, sys.nr_of_dims + sys.nr_of_constraints)
            a_sim = x_sim[:, dims_a]
            x_sim = x_sim[:, :sys.nr_of_dims]
            res.append(SimResult([x_sim], [t_sim], loc, [], [a_sim]))
        else:
            res.append(SimResult([x_sim], [t_sim], loc))
    
    return res


def _aux_outputTrajectoryEnd(sys, params: Dict[str, Any], xtraj: np.ndarray) -> np.ndarray:
    """
    Auxiliary function for computing the output at trajectory end
    
    Args:
        sys: System object
        params: Parameters dict
        xtraj: State trajectory
        
    Returns:
        np.ndarray: Output at trajectory end
    """
    
    if 'uTransVec' in params:
        params['uTrans'] = params['uTransVec'][:, -1].reshape(-1, 1)
    
    ulast = _randPoint(params['U'], 1) + params['uTrans']
    vlast = _randPoint(params['V'], 1)
    
    ylast = (sys.C @ xtraj[-1, :].reshape(-1, 1) + 
             sys.D @ ulast + 
             (sys.k if hasattr(sys, 'k') and sys.k is not None else 0) + 
             vlast)
    
    return ylast


def _randPoint(set_obj, N: int = 1, type_: str = 'standard') -> np.ndarray:
    """
    Helper function to generate random points from a set
    
    Args:
        set_obj: Set object (zonotope, interval, etc.)
        N: Number of points
        type_: Type of random point generation
        
    Returns:
        np.ndarray: Random points
    """
    
    if hasattr(set_obj, 'randPoint_'):
        return set_obj.randPoint_(N, type_)
    elif hasattr(set_obj, 'randPoint'):
        return set_obj.randPoint(N, type_)
    else:
        # Fallback for basic sets
        if hasattr(set_obj, 'center') and hasattr(set_obj, 'generators'):
            # Zonotope-like set
            center = set_obj.center()
            generators = set_obj.generators()
            factors = -1 + 2 * np.random.rand(generators.shape[1], N)
            return center + generators @ factors
        else:
            raise ValueError(f"Cannot generate random points from {type(set_obj)}") 