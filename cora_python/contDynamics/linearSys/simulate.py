"""
simulate - simulates a linear system 

This function simulates a linear system and returns a trajectory starting from
the initial state x0 = x(t0) ∈ R^n for an input signal u(t) ∈ R^m and 
disturbance w(t) ∈ R^r.

Authors: Florian Nüssel (Python implementation)
Date: 8.6.2025
"""

from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
from scipy.integrate import solve_ivp
import warnings


def simulate(linsys, params: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray, Optional[int], Optional[np.ndarray]]:
    """
    Simulates a linear system
    
    Syntax:
        t, x = simulate(linsys, params)
        t, x, ind = simulate(linsys, params, options)
        t, x, ind, y = simulate(linsys, params, options)
    
    Args:
        linsys: LinearSys object
        params: Dict containing the parameters for the simulation
            - tStart: initial time (default: 0)
            - tFinal: final time
            - timeStep: time step to get from initial time to final time (optional)
            - x0: initial point
            - u: piecewise-constant input signal u(t) of dimension
                 m x z, where
                 m = number of system inputs, and
                 z = 1 (fixed input) or
                     s (if no feedthrough matrix given)
                     s+1 (if feedthrough matrix given)
                     where s is the number of steps
            - w: disturbance of dimension n x s (n = system dimension)
            - v: sensor noise of dimension n_out x s+1 (n_out = output dimension)
        options: ODE solver options (optional)
    
    Returns:
        t: time vector
        x: state trajectory
        ind: returns the event which has been detected (None if no event)
        y: output trajectory (None if no output matrix C)
    
    Example:
        A = np.array([[1, 0], [0, 2]])
        B = np.array([[1], [2]])
        linsys = LinearSys(A, B)
        
        params = {'x0': np.array([1, 2]), 'tFinal': 2}
        
        t, x = simulate(linsys, params)
    """
    
    # Make a copy of params to avoid modifying the original
    params = params.copy()
    
    # Parse input arguments
    is_opt = options is not None and len(options) > 0
    
    # Set tStart
    if 'tStart' not in params:
        params['tStart'] = 0
    
    # Shift time horizon to start at 0
    t_final = params['tFinal'] - params['tStart']
    
    # Set time step
    if 'timeStep' in params:
        # Use proper time vector generation to avoid floating-point accumulation errors
        steps = int(np.round(t_final / params['timeStep']))
        t_span = np.linspace(0, t_final, steps + 1)
        time_step_given = True
    else:
        # Time vector and number of steps
        t_span = np.array([0, t_final])
        steps = 1
        time_step_given = False
    
    # Check values of u, w, and v
    params, steps, t_span = _aux_uwv(linsys, params, steps, t_span)
    t_span = t_span[:2]
    
    # Initializations
    params_ = params.copy()
    t = np.array([])
    x = np.array([]).reshape(0, len(params['x0']))
    ind = None
    y = np.array([])
    x0 = params['x0'].copy()
    
    # Computation of output set desired / possible
    comp_y = linsys.C is not None and linsys.C.size > 0
    
    # Loop over all time steps
    for i in range(steps):
        # Input and disturbance are processed in getfcn
        params_['u'] = params['u'][:, i] if params['u'].ndim > 1 else params['u']
        params_['w'] = params['w'][:, i] if params['w'].ndim > 1 else params['w']
        
        # Get function handle
        def dynamics(t_val, x_val):
            return _getfcn(linsys, params_)(t_val, x_val)
        
        # Simulate using scipy's solve_ivp function
        try:
            if is_opt:
                sol = solve_ivp(dynamics, t_span, x0, **options)
            else:
                sol = solve_ivp(dynamics, t_span, x0)
            
            if not sol.success:
                raise RuntimeError(f"Integration failed: {sol.message}")
                
            t_ = sol.t
            x_ = sol.y.T  # Transpose to match MATLAB convention
            
        except Exception as e:
            # Fallback without options if there's an error
            if is_opt:
                sol = solve_ivp(dynamics, t_span, x0)
                if not sol.success:
                    raise RuntimeError(f"Integration failed: {sol.message}")
                t_ = sol.t
                x_ = sol.y.T
            else:
                raise e
        
        # Store the results
        if i == 0:
            if time_step_given:
                # Only save first and last entry
                t = np.array([t_[0], t_[-1]]) + params['tStart']
                x = np.array([x_[0, :], x_[-1, :]])
            else:
                t = t_ + params['tStart']
                x = x_
        else:
            if time_step_given:
                # Only save last entry
                t = np.append(t, t_[-1] + t[-1])
                x = np.vstack([x, x_[-1, :]])
            else:
                # Save all simulated points
                t = np.append(t, t_[1:] + t[-1])
                x = np.vstack([x, x_[1:, :]])
        
        x0 = x[-1, :]
        
        # Compute output: skip last entry since the input is not valid there,
        # instead, this will be covered by the next input
        if comp_y:
            # Get current input for output computation
            u_current = params['u'][:, i] if params['u'].ndim > 1 else params['u']
            u_current = np.asarray(u_current).reshape(-1, 1)
            
            # Handle dimension mismatch for D matrix
            if linsys.D is not None and linsys.D.size > 0:
                if u_current.shape[0] != linsys.D.shape[1]:
                    # If u is scalar and D expects more inputs, broadcast u
                    if u_current.shape[0] == 1 and linsys.D.shape[1] > 1:
                        u_current = np.tile(u_current, (linsys.D.shape[1], 1))
                    else:
                        # Use zero input if dimensions don't match
                        u_current = np.zeros((linsys.D.shape[1], 1))
            
            # Get current noise for output computation
            v_current = params['v'][:, i] if params['v'].ndim > 1 else params['v']
            v_current = np.asarray(v_current).reshape(-1, 1)
            
            if time_step_given:
                y_ = linsys.C @ x_[0, :].reshape(-1, 1)
                if linsys.D is not None and linsys.D.size > 0:
                    y_ += linsys.D @ u_current
                if linsys.k is not None and linsys.k.size > 0:
                    y_ += linsys.k.reshape(-1, 1)
                if linsys.F is not None and linsys.F.size > 0:
                    y_ += linsys.F @ v_current
                y_ = y_.flatten()
            else:
                y_ = linsys.C @ x_[:-1, :].T
                if linsys.D is not None and linsys.D.size > 0:
                    y_ += linsys.D @ u_current
                if linsys.k is not None and linsys.k.size > 0:
                    y_ += linsys.k.reshape(-1, 1)
                if linsys.F is not None and linsys.F.size > 0:
                    y_ += linsys.F @ v_current
                y_ = y_.T
            
            if y.size == 0:
                y = y_.reshape(-1, y_.shape[-1]) if y_.ndim > 1 else y_.reshape(1, -1)
            else:
                if y_.ndim == 1:
                    y = np.vstack([y, y_.reshape(1, -1)])
                else:
                    y = np.vstack([y, y_])
        
        if ind is not None:
            break
    
    # Compute last output
    if comp_y:
        # Get last input for output computation
        u_last = params['u'][:, -1] if params['u'].ndim > 1 else params['u']
        u_last = np.asarray(u_last).reshape(-1, 1)
        
        # Handle dimension mismatch for D matrix
        if linsys.D is not None and linsys.D.size > 0:
            if u_last.shape[0] != linsys.D.shape[1]:
                # If u is scalar and D expects more inputs, broadcast u
                if u_last.shape[0] == 1 and linsys.D.shape[1] > 1:
                    u_last = np.tile(u_last, (linsys.D.shape[1], 1))
                else:
                    # Use zero input if dimensions don't match
                    u_last = np.zeros((linsys.D.shape[1], 1))
        
        # Get last noise for output computation
        v_last = params['v'][:, -1] if params['v'].ndim > 1 else params['v']
        v_last = np.asarray(v_last).reshape(-1, 1)
        
        y_last = linsys.C @ x_[-1, :].reshape(-1, 1)
        if linsys.D is not None and linsys.D.size > 0:
            y_last += linsys.D @ u_last
        if linsys.k is not None and linsys.k.size > 0:
            y_last += linsys.k.reshape(-1, 1)
        if linsys.F is not None and linsys.F.size > 0:
            y_last += linsys.F @ v_last
        y_last = y_last.flatten()
        
        y = np.vstack([y, y_last.reshape(1, -1)])
    
    return t, x, ind, y if comp_y else None


def _getfcn(linsys, params):
    """
    Returns the function handle of the continuous function specified
    by the linear system object
    
    Args:
        linsys: LinearSys object
        params: model parameters
    
    Returns:
        Function handle for the dynamics
    """
    def dynamics(t, x):
        x = np.asarray(x).reshape(-1, 1)
        u = np.asarray(params['u']).reshape(-1, 1)
        w = np.asarray(params['w']).reshape(-1, 1)
        
        # Ensure proper dimensions for matrix multiplication
        result = linsys.A @ x
        
        if linsys.B is not None and linsys.B.size > 0:
            # Check if u has the right dimensions for matrix multiplication
            if u.shape[0] != linsys.B.shape[1]:
                # If u is scalar and B expects more inputs, broadcast u
                if u.shape[0] == 1 and linsys.B.shape[1] > 1:
                    u = np.tile(u, (linsys.B.shape[1], 1))
                else:
                    raise ValueError(f"Input u has dimension {u.shape[0]} but system expects {linsys.B.shape[1]} inputs")
            result += linsys.B @ u
        
        if linsys.c is not None and linsys.c.size > 0:
            result += linsys.c.reshape(-1, 1)
        
        if linsys.E is not None and linsys.E.size > 0:
            # Check if w has the right dimensions for matrix multiplication
            if w.shape[0] != linsys.E.shape[1]:
                # If w is scalar and E expects more disturbances, broadcast w
                if w.shape[0] == 1 and linsys.E.shape[1] > 1:
                    w = np.tile(w, (linsys.E.shape[1], 1))
                else:
                    raise ValueError(f"Disturbance w has dimension {w.shape[0]} but system expects {linsys.E.shape[1]} disturbances")
            result += linsys.E @ w
        
        return result.flatten()
    
    return dynamics


def _aux_uwv(obj, params, steps, t_span):
    """
    Set input vector u, disturbance w, and sensor noise v correctly
    
    Args:
        obj: LinearSys object
        params: parameters dictionary
        steps: number of steps
        t_span: time span
    
    Returns:
        Updated params, steps, and t_span
    """
    # Set default values
    if 'u' not in params:
        params['u'] = np.zeros((obj.nr_of_inputs, 1))
    if 'w' not in params:
        params['w'] = np.zeros((obj.nr_of_disturbances, 1))
    if 'v' not in params:
        params['v'] = np.zeros((obj.nr_of_noises, 1))
    
    # Convert to numpy arrays and ensure proper shape
    params['u'] = np.atleast_2d(params['u'])
    params['w'] = np.atleast_2d(params['w'])
    params['v'] = np.atleast_2d(params['v'])
    
    # Check sizes
    size_u = params['u'].shape[1]
    size_w = params['w'].shape[1]
    size_v = params['v'].shape[1]
    
    # Check if feedthrough matrix D has non-zero elements
    is_d = np.any(obj.D != 0) if obj.D is not None else False
    
    if steps > 1:  # equal to 'timeStep' in params
        
        if is_d:
            columns_u = steps + 1
        else:
            columns_u = steps
        
        # params.u|w|v have to match timeStep
        if size_u != 1 and size_u != columns_u:
            raise ValueError(f'params.u has to be of size 1 or of the size of steps ({columns_u})')
        elif size_w != 1 and size_w != steps:
            raise ValueError(f'params.w has to be of size 1 or of the size of steps ({steps})')
        elif size_v != 1 and size_v != steps + 1:
            raise ValueError(f'params.v has to be of size 1 or of the size of steps ({steps + 1})')
        
        # Repeat fixed values
        if size_u == 1:
            params['u'] = np.tile(params['u'], (1, steps + 1))
        elif size_u == steps:
            # Extend by dummy value
            params['u'] = np.hstack([params['u'], np.zeros((params['u'].shape[0], 1))])
        
        if size_w == 1:
            params['w'] = np.tile(params['w'], (1, steps))
        
        if size_v == 1:
            params['v'] = np.tile(params['v'], (1, steps + 1))
    
    else:  # steps == 1, i.e., no params.timeStep given
        
        # Check if sizes are either 1 or match one another
        if size_u == 1 and size_w == 1 and size_v == 1:
            # All sizes are 1, equals number of steps -> return
            return params, steps, t_span
        else:
            # Check whether sizes fit one another
            sizes_match = True
            # w always one less than v
            if (size_w != 1 and size_v != 1) and size_w != size_v - 1:
                sizes_match = False
            elif is_d:
                if (size_u != 1 and size_v != 1) and size_u != size_v:
                    sizes_match = False
                elif (size_u != 1 and size_w != 1) and size_u != size_w + 1:
                    sizes_match = False
            else:
                if (size_u != 1 and size_v != 1) and size_u != size_v - 1:
                    sizes_match = False
                elif (size_u != 1 and size_w != 1) and size_u != size_w:
                    sizes_match = False
            
            if not sizes_match:  # throw detailed error
                raise ValueError(
                    'If params.timeStep is not provided, the number of columns of the\n'
                    '    input signal (params.u)\n'
                    '    disturbance (params.w)\n'
                    '    sensor noise (params.v)\n'
                    'have to be either 1 or match one another, that is\n'
                    '    Case #1: feedthrough matrix D provided:\n'
                    '        params.u: x+1 columns\n'
                    '        params.w: x   columns\n'
                    '        params.v: x+1 columns\n'
                    '    Case #2: feedthrough matrix D not provided:\n'
                    '        params.u: x   columns\n'
                    '        params.w: x   columns\n'
                    '        params.v: x+1 columns'
                )
        
        # Compute steps
        if size_v != 1:
            steps = size_v - 1
        elif size_w != 1:
            steps = size_w
        elif size_u != 1 and is_d:
            steps = size_u - 1
        elif size_u != 1 and not is_d:
            steps = size_u
        
        # Lengthen vectors
        if size_u == 1:
            params['u'] = np.tile(params['u'], (1, steps + 1))
        elif size_u == steps:
            # Extend by dummy value
            params['u'] = np.hstack([params['u'], np.zeros((params['u'].shape[0], 1))])
        
        if size_w == 1:
            params['w'] = np.tile(params['w'], (1, steps))
        
        if size_v == 1:
            params['v'] = np.tile(params['v'], (1, steps + 1))
        
        # Recompute t_span
        t_span = np.linspace(0, t_span[-1], steps + 1)
        t_span = t_span[:2]
    
    return params, steps, t_span 