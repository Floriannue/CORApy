"""
simulate - simulates a nonlinear system

Syntax:
    [t,x] = simulate(nlnsys,params)
    [t,x,ind] = simulate(nlnsys,params,options)
    [t,x,ind,y] = simulate(nlnsys,params,options)

Inputs:
    nlnsys - nonlinearSys object
    params - struct containing the parameters for the simulation
       .tStart: initial time
       .tFinal: final time
       .timeStep: time step size
       .x0: initial point
       .u: input (piecewise-constant)
    options - ODE solver options (optional)

Outputs:
    t - time vector
    x - state vector
    ind - event indicator (unused, None)
    y - output vector (not supported)

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       03-May-2007 (MATLAB)
Last update:   28-August-2025 (MATLAB)
Python translation: 2025
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy.integrate import solve_ivp

from .getfcn import getfcn


def simulate(nlnsys: Any, params: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray, Optional[int], Optional[np.ndarray]]:
    params = params.copy()
    options = options.copy() if options else {}

    if 'u' not in params:
        params['u'] = np.zeros((nlnsys.nr_of_inputs, 1))
    if 'tStart' not in params:
        params['tStart'] = 0.0

    t_final = params['tFinal'] - params['tStart']
    u = np.asarray(params['u'])
    if u.ndim == 1:
        u = u.reshape(-1, 1)

    # split time horizon based on input segments
    segs = u.shape[1]
    seg_final = t_final / segs

    if 'timeStep' in params:
        t_span = np.arange(0, seg_final + params['timeStep'], params['timeStep'])
        if abs(t_span[-1] - seg_final) > 1e-10:
            t_span = np.append(t_span, seg_final)
        t_eval = t_span
        t_span = (t_span[0], t_span[-1])
    else:
        # MATLAB: ode45 returns adaptive intermediate points if no timeStep is set
        t_span = (0.0, seg_final)
        t_eval = None

    t = []
    x = []
    ind = None
    x0 = np.asarray(params['x0']).flatten()

    # Extract solver options (filter out simulateRandom options)
    refine = options.pop('Refine', options.pop('refine', 4))
    valid_solve_keys = {
        'rtol', 'atol', 'method', 'max_step', 'first_step',
        'vectorized', 'jac', 'jac_sparsity', 'events', 'dense_output'
    }

    for i in range(segs):
        params_seg = params.copy()
        params_seg['u'] = u[:, i:i+1]

        f = getfcn(nlnsys, params_seg)

        solve_opts = {k: v for k, v in options.items() if k in valid_solve_keys}
        solve_opts.setdefault('rtol', 1e-6)
        solve_opts.setdefault('atol', 1e-9)
        if t_eval is None and refine is not None and refine > 1:
            solve_opts.setdefault('dense_output', True)

        sol = solve_ivp(f, t_span, x0, t_eval=t_eval, **solve_opts)
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        if t_eval is None and refine is not None and refine > 1 and sol.sol is not None:
            # MATLAB ode45 uses a refine factor to output additional points per step
            t_dense = [sol.t[0]]
            for j in range(len(sol.t) - 1):
                t_local = np.linspace(sol.t[j], sol.t[j + 1], refine + 1)[1:]
                t_dense.extend(t_local.tolist())
            t_seg = np.array(t_dense)
            x_seg = sol.sol(t_seg).T
        else:
            t_seg = sol.t
            x_seg = sol.y.T

        if i == 0:
            t = t_seg + params['tStart']
            x = x_seg
        else:
            t = np.concatenate([t, t_seg[1:] + t[-1]])
            x = np.vstack([x, x_seg[1:, :]])

        x0 = x_seg[-1, :]

        if ind is not None:
            break

    # Output trajectories not supported for nonlinearSys (MATLAB warning)
    y = None
    return t, x, ind, y
