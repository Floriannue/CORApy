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
    else:
        t_span = np.array([0.0, seg_final])

    t = []
    x = []
    ind = None
    x0 = np.asarray(params['x0']).flatten()

    for i in range(segs):
        params_seg = params.copy()
        params_seg['u'] = u[:, i:i+1]

        f = getfcn(nlnsys, params_seg)

        solve_opts = options.copy()
        solve_opts.setdefault('rtol', 1e-6)
        solve_opts.setdefault('atol', 1e-9)

        sol = solve_ivp(f, (t_span[0], t_span[-1]), x0, t_eval=t_span, **solve_opts)
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

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
