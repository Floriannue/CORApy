"""
priv_simulateRRT - simulates a system using rapidly exploring random trees

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant  
Written: 02-September-2011 (MATLAB)
Last update: 16-June-2023 (MATLAB)
Python translation: 2025
"""

from typing import Dict, Any, List
import numpy as np

from cora_python.g.classes import SimResult


def priv_simulateRRT(sys, params: Dict[str, Any], options: Dict[str, Any]) -> List[SimResult]:
    """
    Private function for RRT-based random simulation
    
    Args:
        sys: contDynamics object
        params: Model parameters
        options: Settings for random simulation including:
                - points: number of random initial points (positive integer)
                - vertSamp: flag for vertex sampling (0 or 1)
                - stretchFac: stretching factor for enlarging reachable sets (scalar > 1)
                - R: reachable set object
        
    Returns:
        List[SimResult]: Simulation results
    """
    
    # Read out reachable set
    R = options['R']
    
    # Obtain set of uncertain inputs
    if 'uTransVec' in params:
        U = params['uTransVec'][:, 0].reshape(-1, 1) + params['U']
    else:
        U = params.get('uTrans', np.zeros((sys.nr_of_inputs, 1))) + params['U']
    
    # Possible extreme inputs
    V_input = _vertices(U)
    nrOfExtrInputs = V_input.shape[1]
    
    # Initialize simulation results
    x = [None] * options['points']
    t = [None] * options['points']
    
    # Initialize obtained states from the RRT
    if options.get('vertSamp', False):
        X = _randPoint(params['R0'], options['points'], 'extreme')
    else:
        X = _randPoint(params['R0'], options['points'], 'standard')
    
    # Set flag for conformance checking
    conformanceChecking = 'convertFromAbstractState' in options
    if conformanceChecking:
        # Create full state: lift each sample to the full state space
        Xfull = np.zeros((sys.nr_of_dims, options['points']))
        for iSample in range(X.shape[1]):
            Xfull[:, iSample] = options['convertFromAbstractState'](X[:, iSample])
    
    # Number of time steps; time point solutions have one step more compared to
    # time interval solutions
    nrSteps = len(R.timePoint.set) - 1
    
    # Loop over all time steps
    for iStep in range(nrSteps):
        
        # Display current step (execution rather slow...)
        print(f"Step {iStep + 1} of {nrSteps}")
        
        # Update time
        params_sim = params.copy()
        params_sim['tStart'] = R.timePoint.time[iStep]
        params_sim['tFinal'] = R.timePoint.time[iStep + 1]
        
        # Enlarge reachable set at starting point in time
        R_enl = _enlarge(R.timePoint.set[iStep], options['stretchFac'])
        
        # Compute normalization factors
        # eps added to avoid division by 0
        X_sample_size = _rad(_interval(R_enl)) + 1e-15
        normMatrix = np.diag(1.0 / X_sample_size)
        
        # Loop over all trajectories
        X_new = np.zeros_like(X)
        if conformanceChecking:
            X_newFull = np.zeros_like(Xfull)
        
        for iSample in range(options['points']):
            
            # Sample
            if options.get('vertSamp', False):
                x_sample = _randPoint(R_enl, 1, 'extreme')
            else:
                x_sample = _randPoint(R_enl, 1, 'standard')
            
            # Nearest neighbor and selected state
            ind = _aux_nearestNeighbor(x_sample, X, normMatrix)
            if not conformanceChecking:
                params_sim['x0'] = X[:, ind].flatten()
            else:
                params_sim['x0'] = Xfull[:, ind].flatten()
            
            # Update set of uncertain inputs when tracking
            if 'uTransVec' in params:
                U = params['uTransVec'][:, iStep].reshape(-1, 1) + params['U']
                V_input = _vertices(U)
            
            # Simulate model to find out best input
            x_next = np.zeros((X.shape[0], nrOfExtrInputs))
            if conformanceChecking:
                x_nextFull = np.zeros((Xfull.shape[0], nrOfExtrInputs))
            
            t_traj = [None] * nrOfExtrInputs
            x_traj = [None] * nrOfExtrInputs
            
            for iInput in range(nrOfExtrInputs):
                # Set input
                params_sim['u'] = V_input[:, iInput].reshape(-1, 1)
                
                # Simulate
                tTemp, xTemp, _ = sys.simulate(params_sim)[:3]
                t_traj[iInput] = tTemp
                x_traj[iInput] = xTemp
                
                x_final = x_traj[iInput][-1, :]
                
                # Reduce to abstract state for conformance checking
                if conformanceChecking:
                    # Save result
                    x_nextFull[:, iInput] = x_final
                    # Project to reduced model
                    x_final = options['convertToAbstractState'](x_final)
                
                # Save result
                x_next[:, iInput] = x_final
            
            # Nearest neighbor is added to new set of sampled states
            ind = _aux_nearestNeighbor(x_sample, x_next, normMatrix)
            X_new[:, iSample] = x_next[:, ind]
            if conformanceChecking:
                X_newFull[:, iSample] = x_nextFull[:, ind]
            
            # Store initial values
            if iStep == 0:
                x[iSample] = x_traj[ind][0, :].reshape(1, -1)
                t[iSample] = np.array([t_traj[ind][0]])
            
            # Store trajectories
            x[iSample] = np.vstack([x[iSample], x_traj[ind][-1, :].reshape(1, -1)])
            t[iSample] = np.append(t[iSample], t_traj[ind][-1])
        
        # Update X
        X = X_new
        if conformanceChecking:
            Xfull = X_newFull
    
    # Store computed simulations in the same format as for other simulation types
    simRes = []
    for iSample in range(options['points']):
        simRes.append(SimResult([x[iSample]], [t[iSample]]))
    
    return simRes


def _aux_nearestNeighbor(x_sample: np.ndarray, X: np.ndarray, normMatrix: np.ndarray) -> int:
    """
    Find the nearest neighbor index
    
    Args:
        x_sample: Sample point
        X: Matrix of points
        normMatrix: Normalization matrix
        
    Returns:
        int: Index of nearest neighbor
    """
    
    # Norm of distance
    X_rel = normMatrix @ (X - x_sample.reshape(-1, 1))
    norm_val = np.linalg.norm(X_rel, axis=0)  # Compute 2-norm
    
    # Find index with smallest norm
    ind = np.argmin(norm_val)
    
    return ind


def _randPoint(set_obj, N: int = 1, type_: str = 'standard') -> np.ndarray:
    """
    Helper function to generate random points from a set
    
    Args:
        set_obj: Set object
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
        # Fallback
        if hasattr(set_obj, 'center') and hasattr(set_obj, 'generators'):
            center = set_obj.center()
            generators = set_obj.generators()
            factors = -1 + 2 * np.random.rand(generators.shape[1], N)
            return center + generators @ factors
        else:
            raise ValueError(f"Cannot generate random points from {type(set_obj)}")


def _vertices(set_obj) -> np.ndarray:
    """
    Get vertices of a set
    
    Args:
        set_obj: Set object
        
    Returns:
        np.ndarray: Vertices
    """
    
    if hasattr(set_obj, 'vertices_'):
        return set_obj.vertices_()
    elif hasattr(set_obj, 'vertices'):
        return set_obj.vertices()
    else:
        # Fallback for simple sets
        raise ValueError(f"Cannot compute vertices of {type(set_obj)}")


def _enlarge(set_obj, factor: float):
    """
    Enlarge a set by a factor
    
    Args:
        set_obj: Set object
        factor: Enlargement factor
        
    Returns:
        Enlarged set
    """
    
    if hasattr(set_obj, 'enlarge'):
        return set_obj.enlarge(factor)
    else:
        # Fallback - multiply generators by factor
        if hasattr(set_obj, 'center') and hasattr(set_obj, 'generators'):
            center = set_obj.center()
            generators = set_obj.generators() * factor
            return set_obj.__class__(center, generators)
        else:
            raise ValueError(f"Cannot enlarge {type(set_obj)}")


def _interval(set_obj):
    """
    Convert set to interval
    
    Args:
        set_obj: Set object
        
    Returns:
        Interval representation
    """
    
    if hasattr(set_obj, 'interval'):
        return set_obj.interval()
    else:
        # Simple conversion
        from cora_python.contSet.interval import Interval
        return Interval(set_obj)


def _rad(interval_obj) -> np.ndarray:
    """
    Get radius of interval
    
    Args:
        interval_obj: Interval object
        
    Returns:
        np.ndarray: Radius vector
    """
    
    if hasattr(interval_obj, 'rad'):
        return interval_obj.rad()
    else:
        return (interval_obj.sup - interval_obj.inf) / 2 