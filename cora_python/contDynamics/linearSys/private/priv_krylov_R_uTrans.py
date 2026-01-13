"""
priv_krylov_R_uTrans - computes the reachable continuous set for time t 
                  due to constant input vector uTrans

Syntax:
    [linsys,R_uTrans_apx] = priv_krylov_R_uTrans(linsys,t,uTrans,options)

Inputs:
    linsys - linearSys object modeling the integrator system for inputs
             (see [1] Lemma 4.1) 
    t      - time point (scalar double)
    uTrans - constant input vector
    options - options for the computation of the reachable set

Outputs:
    linsys   - original linearSys object (with results of this function saved)
    R_uTrans_apx - reachable set due to uTrans at the time t

References:
    [1] Matthias Althoff, Reachability Analysis of Large Linear Systems
        With Uncertain Inputs in the Krylov Subspace

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: - 

Authors:       Maximilian Perschl
Written:       25-April-2025
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import scipy.linalg
from scipy.sparse import issparse
from typing import Any, Dict, Tuple
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.g.classes.taylorLinSys import TaylorLinSys
from .priv_subspace_Krylov_individual_Jawecki import priv_subspace_Krylov_individual_Jawecki
from .priv_correctionMatrixInput import priv_correctionMatrixInput


def priv_krylov_R_uTrans(linsys: Any, t: float, uTrans: np.ndarray, 
                        options: Dict[str, Any]) -> Tuple[Any, Zonotope]:
    """
    Computes the reachable continuous set for time t due to constant input vector uTrans
    
    Args:
        linsys: linearSys object
        t: time point (scalar double)
        uTrans: constant input vector
        options: options for the computation of the reachable set
        
    Returns:
        linsys: original linearSys object (with results of this function saved)
        R_uTrans_apx: reachable set due to uTrans at the time t
    """
    
    # Convert to numpy arrays if needed
    if issparse(uTrans):
        uTrans = uTrans.toarray().flatten()
    else:
        uTrans = np.asarray(uTrans).flatten()
    
    state_dim = linsys.A.shape[0]
    
    # create integrator system
    # MATLAB: A_int = [linsys.A, uTrans; zeros(1,state_dim), 0];
    A_int = np.block([[linsys.A, uTrans.reshape(-1, 1)],
                      [np.zeros((1, state_dim)), np.array([[0]])]])
    
    # equivalent initial state
    # MATLAB: eqivState = [zeros(state_dim,1); 1];
    eqivState = np.zeros((state_dim + 1, 1))
    eqivState[-1, 0] = 1.0
    
    # Arnoldi
    V_uT, H_uT, _, _ = priv_subspace_Krylov_individual_Jawecki(A_int, eqivState, 1, options)
    
    # MATLAB: V_uT_proj = linsys.C*V_uT(1:state_dim,:);
    V_uT_proj = linsys.C @ V_uT[:state_dim, :]
    
    # initialize linear reduced dynamics
    # MATLAB: linRedSys = linearSys('reduced_sys_uTrans',H_uT,V_uT_proj');
    linRedSys = LinearSys('reduced_sys_uTrans', H_uT, V_uT_proj.T)
    # need taylor object for correction matrix
    linRedSys.taylor = TaylorLinSys(H_uT)
    
    # compute P*(...)*e_1 via indexing for fast computation
    # norm(equiv_state) = 1, so we leave it out;
    expMat = scipy.linalg.expm(H_uT * t)
    R_uTrans_apx = V_uT_proj @ expMat[:, 0:1]
    
    # add krylov error
    krylovError = options.get('krylovError', 0.0)
    if krylovError > 2 * np.finfo(float).eps:
        R_uTrans_proj = Zonotope(R_uTrans_apx, 
                                 np.eye(R_uTrans_apx.shape[0]) * krylovError)
    else:
        R_uTrans_proj = Zonotope(R_uTrans_apx, np.array([]).reshape(R_uTrans_apx.shape[0], 0))
    
    G = priv_correctionMatrixInput(linRedSys, t, options.get('taylorTerms', 10))
    # compute P*(...)*e_1 via indexing for fast computation
    # norm(equiv_state) = 1, so we leave it out;
    inputCorr_unprojected_interval = V_uT[:state_dim, :] @ G[:, 0:1]
    # inputCorrection = zonotope(inputCorr_unprojected(1:state_dim)); 
    # MATLAB: inputCorr_unprojected = zonotope(inputCorr_unprojected);
    # Convert Interval to Zonotope
    inputCorr_unprojected = Zonotope(inputCorr_unprojected_interval)
    
    # save results for future re-computations of correction matrix
    if not hasattr(linsys, 'krylov'):
        linsys.krylov = {}
    linsys.krylov['uTrans_sys'] = linRedSys
    linsys.krylov['R_uTrans'] = V_uT[:state_dim, :] @ expMat[:, 0:1]
    linsys.krylov['R_uTrans_proj'] = R_uTrans_proj
    linsys.krylov['inputCorr'] = linsys.C @ inputCorr_unprojected
    from cora_python.contSet.zonotope.radius import radius
    linsys.krylov['inputCorr_radius'] = radius(inputCorr_unprojected)
    
    return linsys, R_uTrans_proj

