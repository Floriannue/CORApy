"""
priv_initReach_Krylov - computes necessary preliminary results for
reachability analysis in the Krylov subspace

Syntax:
    [linsys,params,options] = priv_initReach_Krylov(linsys,params,options)

Inputs:
    linsys - linearSys object
    params - model parameters
    options - options for the computation of the reachable set

Outputs:
    linsys - linearSys object
    params - model parameters
    options - options for the computation of the reachable set

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff, Maximilian Perschl
Written:       15-November-2016
Last update:   24-July-2020 (box inputs removed)
               25-April-2025 (MP, major refactor including new error bound for subspaces)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Dict, Tuple
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.g.classes.taylorLinSys import TaylorLinSys
from .priv_krylov_R_uTrans import priv_krylov_R_uTrans
from .priv_inputSolution_Krylov import priv_inputSolution_Krylov
from .priv_subspace_Krylov_jaweckiBound import priv_subspace_Krylov_jaweckiBound


def priv_initReach_Krylov(linsys: Any, params: Dict[str, Any], 
                         options: Dict[str, Any]) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Computes necessary preliminary results for reachability analysis in the Krylov subspace
    
    Args:
        linsys: linearSys object
        params: model parameters
        options: options for the computation of the reachable set
        
    Returns:
        linsys: linearSys object (with krylov data initialized)
        params: model parameters (possibly modified)
        options: options (possibly modified)
    """
    
    # augment options
    options['tFinal'] = params['tFinal']
    
    # initialize field for tp solutions
    if not hasattr(linsys, 'krylov') or linsys.krylov is None:
        linsys.krylov = {}
    linsys.krylov['Rhom_tp_prev'] = linsys.C @ params['R0']
    
    # INPUT SOLUTION-----------------------------------------------------------
    
    # project input set and vector
    params['U'] = linsys.B @ params['U']
    uTrans = linsys.B @ params.get('uTrans', np.zeros((linsys.nr_of_inputs, 1)))
    # compute solution for constant inputs
    linsys, _ = priv_krylov_R_uTrans(linsys, options['timeStep'], uTrans, options)
    # Computation of first input solution of input set
    # (is necessary for subspaces)
    
    linsys = priv_inputSolution_Krylov(linsys, params, options)
    
    # STATE SOLUTION-----------------------------------------------------------
    
    # compute state subspaces and save them in obj.krylov.state
    linsys = _aux_create_state_subspaces(linsys, params['R0'], options)
    
    # compute and save subspaces for input solution
    linsys = _aux_create_input_subspaces(linsys, options)
    
    return linsys, params, options


# Auxiliary functions -----------------------------------------------------

def _aux_create_state_subspaces(linsys: Any, R0: Any, options: Dict[str, Any]) -> Any:
    """
    Create Krylov subspaces for initial set Rinit and save them in obj.krylov.state
    """
    
    # compute subspaces
    V_c, H_c, V_g, H_g, _ = priv_subspace_Krylov_jaweckiBound(linsys.A, R0, options)
    
    # save subspace matrices within linearSys object to create taylor object and project them
    # center
    C = linsys.C
    
    if not hasattr(linsys, 'krylov'):
        linsys.krylov = {}
    if 'state' not in linsys.krylov:
        linsys.krylov['state'] = {}
    
    if V_c is not None and H_c is not None:
        linsys.krylov['state']['c_sys'] = LinearSys(H_c, V_c.T)
        linsys.krylov['state']['c_sys'].taylor = TaylorLinSys(H_c)
        linsys.krylov['state']['c_sys_proj'] = LinearSys(H_c, (C @ V_c).T)
        linsys.krylov['state']['c_sys_proj'].taylor = TaylorLinSys(H_c)
    else:
        linsys.krylov['state']['c_sys'] = None
        linsys.krylov['state']['c_sys_proj'] = None
    
    # generators
    if V_g and len(V_g) > 0:
        linsys.krylov['state']['g_sys'] = []
        linsys.krylov['state']['g_sys_proj'] = []
        for i in range(len(V_g)):
            if V_g[i] is not None and H_g[i] is not None:
                linsys.krylov['state']['g_sys'].append(LinearSys(H_g[i], V_g[i].T))
                linsys.krylov['state']['g_sys'][-1].taylor = TaylorLinSys(H_g[i])
                linsys.krylov['state']['g_sys_proj'].append(LinearSys(H_g[i], (C @ V_g[i]).T))
                linsys.krylov['state']['g_sys_proj'][-1].taylor = TaylorLinSys(H_g[i])
            else:
                linsys.krylov['state']['g_sys'].append(None)
                linsys.krylov['state']['g_sys_proj'].append(None)
    else:
        linsys.krylov['state']['g_sys'] = []
        linsys.krylov['state']['g_sys_proj'] = []
    
    return linsys


def _aux_create_input_subspaces(linsys: Any, options: Dict[str, Any]) -> Any:
    """
    Create Krylov subspaces for input set U_0
    and save them in linsys.krylov.input
    """
    
    # U_0
    # compute subspaces
    V_c, H_c, V_g, H_g, _ = priv_subspace_Krylov_jaweckiBound(linsys.A, linsys.krylov['RV_0'], options)
    
    # save subspace matrices within linearSys object to create taylor object and project them
    # center
    C = linsys.C
    
    if 'input' not in linsys.krylov:
        linsys.krylov['input'] = {}
    
    if V_c is not None and H_c is not None:
        linsys.krylov['input']['c_sys_proj'] = LinearSys(H_c, (C @ V_c).T)
        linsys.krylov['input']['c_sys_proj'].taylor = TaylorLinSys(H_c)
    else:
        linsys.krylov['input']['c_sys_proj'] = None
    
    # generators
    if V_g and len(V_g) > 0:
        linsys.krylov['input']['g_sys_proj'] = []
        for i in range(len(V_g)):
            if V_g[i] is not None and H_g[i] is not None:
                linsys.krylov['input']['g_sys_proj'].append(LinearSys(H_g[i], (C @ V_g[i]).T))
                linsys.krylov['input']['g_sys_proj'][-1].taylor = TaylorLinSys(H_g[i])
            else:
                linsys.krylov['input']['g_sys_proj'].append(None)
    else:
        linsys.krylov['input']['g_sys_proj'] = []
    
    return linsys

