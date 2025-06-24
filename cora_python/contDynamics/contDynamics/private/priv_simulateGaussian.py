"""
priv_simulateGaussian - performs several random simulation of the system

This function performs several random simulations of the system assuming
Gaussian distributions of the initial states, the disturbance, and the
sensor noise. In order to respect hard limits of the aforementioned variables,
values are cut off to respect these bounds.

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 19-November-2020 (MATLAB)
Last update: 10-November-2021 (MATLAB)
Python translation: 2025
"""

from typing import Dict, Any, List
import numpy as np

from cora_python.g.classes import SimResult


def priv_simulateGaussian(sys, params: Dict[str, Any], options: Dict[str, Any]) -> List[SimResult]:
    """
    Private function for Gaussian random simulation
    
    Args:
        sys: contDynamics object
        params: Model parameters
        options: Settings for random simulation
        
    Returns:
        List[SimResult]: Simulation results
    """
    
    # Check if trajectory tracking is required
    tracking = 'uTransVec' in params
    
    # Output equation only handled for linear systems
    comp_y = ((sys.__class__.__name__ in ['LinearSys', 'LinearSysDT']) and 
              hasattr(sys, 'C') and sys.C is not None and sys.C.size > 0)
    
    # Location for contDynamics always zero
    loc = 0
    
    # Initialize array of simResult objects
    res = []
    
    # Loop over all starting points
    for r in range(options['points']):
        
        # Start of trajectory
        t_sim = []
        params_sim = params.copy()
        params_sim['x0'] = _randPoint(params['R0'], 1, 'gaussian', options['p_conf'])
        x_sim = params_sim['x0'].T.copy()
        
        if comp_y:
            y_sim = np.zeros((1, sys.nr_of_outputs))
        
        # Loop over number of constant inputs per partial simulation
        for block in range(len(options['nrConstInp'])):
            
            # Update initial state
            if block > 0:
                params_sim['x0'] = xTemp[-1, :].flatten()  # 1D array for scipy.integrate.solve_ivp
            
            # Update input
            if tracking:
                params_sim['uTrans'] = params['uTransVec'][:, block].reshape(-1, 1)
            
            params_sim['tStart'] = params['tu'][block]
            params_sim['tFinal'] = params['tu'][block + 1]
            
            # Obtain random input
            if not _representsa_emptySet(params['U']):
                # Set input
                uRand = _randPoint(params['U'], 1, 'gaussian', options['p_conf'])
                
                # Combine inputs (random input + tracking)
                if 'uTrans' not in params_sim:
                    params_sim['uTrans'] = np.zeros((sys.nr_of_inputs, 1))
                params_sim['u'] = uRand + params_sim['uTrans']
            else:
                if 'uTrans' not in params_sim:
                    params_sim['uTrans'] = np.zeros((sys.nr_of_inputs, 1))
                params_sim['u'] = params_sim['uTrans']
            
            if comp_y:
                # Obtain disturbance
                params_sim['w'] = _randPoint(params['W'], 1, 'gaussian', options['p_conf'])
                # Obtain sensor noise
                params_sim['v'] = _randPoint(params['V'], 1, 'gaussian', options['p_conf'])
            
            # Uncertain parameters
            if 'paramInt' in params:
                pInt = params['paramInt']
                if hasattr(pInt, 'inf') and hasattr(pInt, 'rad'):  # interval
                    params_sim['p'] = pInt.inf + 2 * pInt.rad * np.random.rand()
                else:
                    params_sim['p'] = pInt
            
            # Simulate dynamic system
            if comp_y:
                tTemp, xTemp, _, yTemp = sys.simulate(params_sim)
            else:
                tTemp, xTemp, _ = sys.simulate(params_sim)[:3]
            
            # Append to previous values
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
        
        if comp_y:
            # Final point of output trajectory uses different input and sensor noise
            ylast = _aux_outputTrajectoryEnd(sys, params_sim, x_sim)
            y_sim[-1, :] = ylast.flatten()
            
            # Append simResult object for r-th trajectory
            res.append(SimResult([x_sim], [t_sim], loc, [y_sim]))
        else:
            # Append simResult object for r-th trajectory
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


def _randPoint(set_obj, N: int = 1, type_: str = 'standard', p_conf: float = 0.99) -> np.ndarray:
    """
    Helper function to generate random points from a set
    
    Args:
        set_obj: Set object (zonotope, interval, etc.)
        N: Number of points
        type_: Type of random point generation
        p_conf: Confidence probability for Gaussian sampling
        
    Returns:
        np.ndarray: Random points
    """
    
    if hasattr(set_obj, 'randPoint_'):
        if type_ == 'gaussian':
            return set_obj.randPoint_(N, type_, p_conf)
        else:
            return set_obj.randPoint_(N, type_)
    elif hasattr(set_obj, 'randPoint'):
        if type_ == 'gaussian':
            return set_obj.randPoint(N, type_, p_conf)
        else:
            return set_obj.randPoint(N, type_)
    else:
        # Fallback for basic sets
        if hasattr(set_obj, 'center') and hasattr(set_obj, 'generators'):
            # Zonotope-like set
            center = set_obj.center()
            generators = set_obj.generators()
            if type_ == 'gaussian':
                # For Gaussian sampling, use normal distribution truncated to bounds
                factors = np.random.randn(generators.shape[1], N)
                # Simple truncation to respect bounds (could be improved)
                factors = np.clip(factors, -1, 1)
            else:
                factors = -1 + 2 * np.random.rand(generators.shape[1], N)
            return center + generators @ factors
        else:
            raise ValueError(f"Cannot generate random points from {type(set_obj)}")


def _representsa_emptySet(set_obj) -> bool:
    """
    Check if a set represents an empty set
    
    Args:
        set_obj: Set object
        
    Returns:
        bool: True if the set is empty
    """
    
    if hasattr(set_obj, 'representsa_'):
        return set_obj.representsa_('emptySet')
    elif hasattr(set_obj, 'isempty'):
        return set_obj.isempty()
    else:
        return False 