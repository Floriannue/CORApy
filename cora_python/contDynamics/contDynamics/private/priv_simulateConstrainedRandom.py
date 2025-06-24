"""
priv_simulateConstrainedRandom - performs several random simulation of the system

This function performs several random simulations of the system so that the
simulations stay within a given reachable set; this reachable set is typically
a backwards minmax reachable set. These reachable sets assume that a control
input exists such that the solutions stay within the reachable set.

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 05-January-2023 (MATLAB)
Last update: 29-June-2024 (MATLAB)
Python translation: 2025
"""

from typing import Dict, Any, List
import numpy as np

from cora_python.g.classes import SimResult


def priv_simulateConstrainedRandom(sys, params: Dict[str, Any], options: Dict[str, Any]) -> List[SimResult]:
    """
    Private function for constrained random simulation
    
    Args:
        sys: contDynamics object
        params: Model parameters
        options: Settings for random simulation including:
                - points: number of random initial points
                - fracVert: fraction of initial states starting from vertices
                - R: reachable set that constrains the simulation
        
    Returns:
        List[SimResult]: Simulation results
    """
    
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
    
    # Check number of generated points (might be different for e.g. empty sets)
    nrPoints = X0.shape[1]
    
    # Initialize time and state
    t = [None] * nrPoints
    x = [None] * nrPoints
    
    # Output equation only for linearSys and linearSysDT currently
    comp_y = ((sys.__class__.__name__ in ['LinearSys', 'LinearSysDT']) and 
              hasattr(sys, 'C') and sys.C is not None and sys.C.size > 0)
    
    if comp_y:
        y = [None] * nrPoints
    
    # Loop over all starting points in X0
    for r in range(nrPoints):
        
        # Is output desired?
        if comp_y:
            y[r] = np.zeros((1, sys.nr_of_outputs))
        
        # Start of trajectory
        params_sim = params.copy()
        params_sim['x0'] = X0[:, r].flatten()  # 1D array for scipy.integrate.solve_ivp
        
        # Simulate dynamical system using constrained simulation
        if comp_y:
            if hasattr(sys, 'simulateConstrained'):
                tTemp, xTemp, _, yTemp = sys.simulateConstrained(params_sim, options)
                t[r] = tTemp
                x[r] = xTemp
                y[r] = yTemp
            else:
                # Fallback to regular simulation if simulateConstrained not implemented
                tTemp, xTemp, _, yTemp = sys.simulate(params_sim, options)
                t[r] = tTemp
                x[r] = xTemp
                y[r] = yTemp
        else:
            if hasattr(sys, 'simulateConstrained'):
                tTemp, xTemp, _ = sys.simulateConstrained(params_sim, options)[:3]
                t[r] = tTemp
                x[r] = xTemp
            else:
                # Fallback to regular simulation if simulateConstrained not implemented
                tTemp, xTemp, _ = sys.simulate(params_sim, options)[:3]
                t[r] = tTemp
                x[r] = xTemp
    
    # Construct object storing the simulation results
    res = []
    for r in range(nrPoints):
        if comp_y:
            res.append(SimResult([x[r]], [t[r]], 0, [y[r]]))
        else:
            res.append(SimResult([x[r]], [t[r]], 0))
    
    return res


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