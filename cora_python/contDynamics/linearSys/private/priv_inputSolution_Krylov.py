"""
priv_inputSolution_Krylov - computes the set of input solutions in the Krylov
   subspace

Syntax:
    linsys = priv_inputSolution_Krylov(linsys,options)

Inputs:
    linsys - linearSys object
    params - model parameters
    options - options for the computation of the reachable set

Outputs:
    linsys - linearSys object

Example:
    -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff, Maximilian Perschl
Written:       19-December-2016
Last update:   25-April-2025 (MP, major refactor)
Last revision: 21-March-2025 (TL)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from scipy.sparse import issparse, csc_matrix
from scipy.special import factorial
from typing import Any, Dict
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.interval.interval import Interval
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.g.classes.taylorLinSys import TaylorLinSys
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from .priv_subspace_Krylov_individual_Jawecki import priv_subspace_Krylov_individual_Jawecki
from .priv_expmRemainder import priv_expmRemainder


def priv_inputSolution_Krylov(linsys: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Any:
    """
    Computes the set of input solutions in the Krylov subspace
    
    Args:
        linsys: linearSys object
        params: model parameters (needs: U)
        options: options for the computation of the reachable set
        
    Returns:
        linsys: linearSys object (with results saved in linsys.krylov)
    """
    
    U = params['U']
    
    # initialize variables
    U_G = U.generators() if hasattr(U, 'generators') else U.G
    if issparse(U_G):
        U_G = U_G.toarray()
    else:
        U_G = np.asarray(U_G)
    nrOfGens = U_G.shape[1] if U_G.ndim > 1 and U_G.shape[1] > 0 else 0
    
    # equivalent initial state
    # MATLAB: eqivState = [sparse([],[],[],linsys.nrOfDims,1);1];
    eqivState = np.zeros((linsys.nr_of_dims + 1, 1))
    eqivState[-1, 0] = 1.0
    
    # init Krylov order
    KrylovOrder = 1
    
    # init
    Ugen = None
    errorTaylor_sum_rad = np.zeros((linsys.nr_of_dims, 1))
    
    # consider generators; first check if generator is zero
    for iGen in range(nrOfGens):
        U_g = U_G[:, iGen]
        if not np.all(withinTol(U_g, 0)):
            
            # create integrator system; minus sign to fit paper
            # MATLAB: A_int_g = [linsys.A, U_g; zeros(1,linsys.nrOfDims), 0];
            A_int_g = np.block([[linsys.A, U_g.reshape(-1, 1)],
                               [np.zeros((1, linsys.nr_of_dims)), np.array([[0]])]])
            
            # Arnoldi
            V_Ug, H_Ug, KrylovOrder, _ = priv_subspace_Krylov_individual_Jawecki(
                A_int_g, eqivState, KrylovOrder, options)
            
            # generate reduced order system for the generator
            A_red_g = H_Ug
            B_red_g = V_Ug
            # initialize linear reduced dynamics
            linRedSys_g = LinearSys('linearReducedDynamics', A_red_g, B_red_g.T)
            linRedSys_g.taylor = TaylorLinSys(A_red_g)
            
            # Loop through Taylor terms
            taylorTerms = options.get('taylorTerms', 10)
            timeStep = options.get('timeStep', 0.01)
            Ugen_list = []
            for i in range(1, taylorTerms + 2):  # i from 1 to taylorTerms+1
                # unprojected, partial result
                # MATLAB: Apower_i = getTaylor(linRedSys_g,'Apower',struct('ithpower',i));
                # Use getTaylor method
                Apower_i = linRedSys_g.getTaylor('Apower', {'ithpower': i})
                U_unprojected = V_Ug @ Apower_i[:, 0:1] * (timeStep ** i) / factorial(i)
                # compute sums
                Ugen_list.append(U_unprojected[:linsys.nr_of_dims, 0])
            
            if Ugen is None:
                Ugen = np.column_stack(Ugen_list) if len(Ugen_list) > 0 else np.array([]).reshape(linsys.nr_of_dims, 0)
            else:
                Ugen = np.column_stack([Ugen, np.column_stack(Ugen_list)])
            
            # compute exponential matrix
            E = priv_expmRemainder(linRedSys_g, timeStep, taylorTerms)
            
            # error due to finite Taylor series
            # MATLAB: errorTaylor_g = supremum(V_Ug*E(:,1));
            E_col = E[:, 0:1] if E.ndim > 1 else E.reshape(-1, 1)
            V_Ug_E = V_Ug @ E_col
            errorTaylor_g = np.max(np.abs(V_Ug_E), axis=0)  # supremum approximation
            errorTaylor_sum_rad = errorTaylor_sum_rad + errorTaylor_g[:linsys.nr_of_dims].reshape(-1, 1)
    
    # round to 0 for numerical stability
    errorTaylor_sum_rad[np.abs(errorTaylor_sum_rad) < np.finfo(float).eps] = 0
    
    errorTaylor_sum = Interval(-errorTaylor_sum_rad, errorTaylor_sum_rad)
    
    # input zonotope without error terms
    if Ugen is None or Ugen.size == 0:
        inputSolV = Zonotope(np.zeros((linsys.nr_of_dims, 1)), np.array([]).reshape(linsys.nr_of_dims, 0))
    else:
        inputSolV = Zonotope(np.zeros((linsys.nr_of_dims, 1)), Ugen)
    
    # Check if error due to finite Taylor series needs to be considered
    krylovError = options.get('krylovError', 0.0)
    if krylovError > 2 * np.finfo(float).eps:
        # error due to order reduction
        U_G_size = U_G.shape[1] if U_G.ndim > 1 and U_G.shape[1] > 0 else 0
        errorRed = Interval(-np.ones((linsys.nr_of_dims, 1)), np.ones((linsys.nr_of_dims, 1))) * \
                   (krylovError * U_G_size)
        # total error
        error_set = errorTaylor_sum + errorRed
    else:
        error_set = errorTaylor_sum
    
    # initialize error for adaptive algorithm
    if not hasattr(linsys, 'krylov'):
        linsys.krylov = {}
    from cora_python.contSet.zonotope.radius import radius
    error_set_zonotope = Zonotope(error_set)
    linsys.krylov['total_U_0_error'] = radius(error_set_zonotope)
    
    # final input set
    inputSolV = inputSolV + Zonotope(error_set)
    
    # write to object structure
    linsys.krylov['V'] = U
    linsys.krylov['RV'] = inputSolV
    
    C = linsys.C
    
    # if clause in case this method is called again in adaptive case
    if 'Rpar_proj' not in linsys.krylov:
        linsys.krylov['Rpar_proj'] = np.zeros((linsys.nr_of_outputs, 1))
    linsys.krylov['Rpar_proj_0'] = C @ inputSolV
    
    linsys.krylov['RV_0'] = inputSolV
    
    return linsys

