"""
priv_exponential_Krylov_projected_linSysInput - computes the overapproximation of the 
   exponential of a system matrix up to a certain accuracy using a Krylov 
   subspace; the subspace is already precomputed

Syntax:
    [R,R_Krylov] = priv_exponential_Krylov_projected_linSysInput(c_sys,g_sys,R0,dim_proj,options)

Inputs:
    c_sys - Linear system with sys.A = H_c, sys.B = V_c'
    g_sys - Cell array of linear systems with sys{i}.A = H_g{i}, sys{i}.B = V_g{i}'
    R0 - initial set
    dim_proj - dimension of projection (output dimension)
    options - reachability options

Outputs:
    R - reachable set
    R_Krylov - reachable set due to Krylov error

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Matthias Althoff, Maximilian Perschl
Written:       02-November-2018
Last update:   21-August-2024 (MP, replace stateflag with V/H as inputs)
               25-April-2025 (MP, take linearSys as input to avoid recomputations)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import scipy.linalg
from scipy.sparse import issparse, csc_matrix
from typing import Any, Dict, List, Optional, Tuple
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.interval.interval import Interval


def priv_exponential_Krylov_projected_linSysInput(c_sys: Any, g_sys: List[Any], R0: Any,
                                                  dim_proj: int, options: Dict[str, Any]) -> Tuple[Zonotope, Zonotope]:
    """
    Computes the overapproximation of the exponential of a system matrix using Krylov subspace
    
    Args:
        c_sys: Linear system with sys.A = H_c, sys.B = V_c' (or None if center is zero)
        g_sys: List of linear systems with sys[i].A = H_g[i], sys[i].B = V_g[i]'
        R0: initial set (zonotope)
        dim_proj: dimension of projection (output dimension)
        options: reachability options (needs: t, krylovError)
        
    Returns:
        R: reachable set
        R_Krylov: reachable set due to Krylov error
    """
    
    # Multiply previous reachable set with exponential matrix
    # obtain center and generators of previous reachable set
    c = R0.center() if hasattr(R0, 'center') else R0.c
    G = R0.generators() if hasattr(R0, 'generators') else R0.G
    
    # Convert to numpy arrays
    if issparse(c):
        c = c.toarray().flatten()
    else:
        c = np.asarray(c).flatten()
    
    if issparse(G):
        G = G.toarray()
    else:
        G = np.asarray(G)
    
    c_norm = np.linalg.norm(c)
    
    # check if center is zero
    if c_sys is None or (hasattr(c_sys, '__len__') and len(c_sys) == 0):
        c_new = np.zeros((dim_proj, 1))
    else:
        # Compute new center
        t = options.get('t', 0.0)
        expMatrix = scipy.linalg.expm(c_sys.A * t)
        # MATLAB: c_new = c_norm*c_sys.B'*expMatrix(:,1);
        c_new = c_norm * (c_sys.B.T @ expMatrix[:, 0:1])
    
    # preallocation
    nrOfGens = G.shape[1] if G.ndim > 1 and G.shape[1] > 0 else 0
    g_norm = np.zeros(nrOfGens)
    
    # obtain generators using the Arnoldi iteration
    G_new = np.zeros((dim_proj, nrOfGens)) if nrOfGens > 0 else np.array([]).reshape(dim_proj, 0)
    
    if nrOfGens > 0:
        t = options.get('t', 0.0)
        for iGen in range(nrOfGens):
            g_col = G[:, iGen]
            g_norm[iGen] = np.linalg.norm(g_col)
            if g_norm[iGen] == 0:
                G_new[:, iGen] = g_col[:dim_proj] if len(g_col) >= dim_proj else np.pad(g_col, (0, dim_proj - len(g_col)))[:dim_proj]
            else:
                # Compute new generator
                expMatrix = scipy.linalg.expm(g_sys[iGen].A * t)
                # MATLAB: G_new(:,iGen) = g_norm(iGen)*g_sys{iGen}.B'*expMatrix(:,1);
                G_new[:, iGen] = (g_norm[iGen] * (g_sys[iGen].B.T @ expMatrix[:, 0:1])).flatten()
    else:
        G_new = np.array([]).reshape(dim_proj, 0)  # no generators
    
    krylovError = options.get('krylovError', 0.0)
    
    if krylovError > 2 * np.finfo(float).eps:
        # Krylov error computation
        # +1 due to center
        if hasattr(R0, 'generators'):
            R0_G = R0.generators()
        elif hasattr(R0, 'G'):
            R0_G = R0.G
        else:
            dim = R0.dim() if hasattr(R0, 'dim') else (R0.nr_of_dims if hasattr(R0, 'nr_of_dims') else 1)
            R0_G = np.array([]).reshape(dim, 0)
        R0_G_cols = R0_G.shape[1] if R0_G.ndim > 1 and R0_G.shape[1] > 0 else 0
        error = krylovError * (R0_G_cols + 1)
        Krylov_interval = Interval(-np.ones((len(c_new), 1)), np.ones((len(c_new), 1))) * error
        R_Krylov = Zonotope(Krylov_interval)
        
        # initial-state-solution zonotope
        error_matrix = error * np.eye(len(c_new))
        R = Zonotope(c_new, np.hstack([G_new, error_matrix]) if G_new.size > 0 else error_matrix)
    else:
        R_Krylov = Zonotope(np.zeros((dim_proj, 1)), np.array([]).reshape(dim_proj, 0))
        R = Zonotope(c_new, G_new) if G_new.size > 0 else Zonotope(c_new, np.array([]).reshape(dim_proj, 0))
    
    return R, R_Krylov

