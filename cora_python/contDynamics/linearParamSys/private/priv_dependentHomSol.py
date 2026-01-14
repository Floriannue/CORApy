"""
priv_dependentHomSol - computes the homogeneous solution when the parameters
   of the system matrix and the input are dependent

Syntax:
    Rhom_tp = priv_dependentHomSol(sys, Rinit, Uconst)

Inputs:
    sys - LinearParamSys object
    Rinit - initial reachable set
    Uconst - set of constant inputs

Outputs:
    Rhom_tp - homogeneous reachable set

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

import numpy as np
from typing import Any
from cora_python.contSet.zonotope import Zonotope
# center is attached to Zonotope class, use object.center()


def priv_dependentHomSol(sys: Any, Rinit: Any, Uconst: Any) -> Zonotope:
    """
    Computes the dependent homogeneous solution for linear parametric systems
    
    Args:
        sys: LinearParamSys object (must have A as matZonotope, stepSize, nr_of_dims, mappingMatrixSet)
        Rinit: Initial reachable set (Zonotope)
        Uconst: Set of constant inputs (Zonotope)
        
    Returns:
        Rhom_tp: Homogeneous reachable set (Zonotope)
    """
    # Obtain required variables
    # MATLAB: Ac = obj.A.C;
    # MATLAB: Ag = obj.A.G;
    Ac = sys.A.C
    Ag = sys.A.G
    # MATLAB: c = center(Rinit);
    c = Rinit.center()
    r = sys.stepSize
    n = sys.nr_of_dims
    # MATLAB: Uc = Uconst.c;
    # MATLAB: UG = Uconst.G;
    Uc = Uconst.center()
    UG = Uconst.generators()
    
    # Transform from input dimension to state space dimension using sys.B
    # Uc should be a column vector for matrix multiplication
    if Uc.ndim == 1:
        Uc = Uc.reshape(-1, 1)
    # Transform to state space dimension
    if Uc.shape[0] != n:
        # Uc is in input dimension, transform to state space dimension
        Uc = sys.B @ Uc
    # Ensure Uc is a column vector
    if Uc.ndim == 1:
        Uc = Uc.reshape(-1, 1)
    
    # Transform UG generators similarly
    if UG.size > 0:
        if UG.ndim == 1:
            UG = UG.reshape(-1, 1)
        # Check if UG is in input dimension
        if UG.shape[0] != n:
            # UG is in input dimension, transform to state space dimension
            UG = sys.B @ UG
        # Ensure UG is 2D
        if UG.ndim == 1:
            UG = UG.reshape(-1, 1)
    # MATLAB: params = obj.A.numgens;
    from cora_python.matrixSet.matZonotope.numgens import numgens
    params = numgens(sys.A)
    
    # SECOND ORDER DEPENDENT SOLUTION
    # Zero parametric order
    # MATLAB: R_c = c + Uc*r;
    # Ensure c and Uc are column vectors
    if c.ndim == 1:
        c = c.reshape(-1, 1)
    if Uc.ndim == 1:
        Uc = Uc.reshape(-1, 1)
    R_c = c + Uc * r
    # Ensure R_c is a 1D array (Zonotope expects 1D, not 2D column vector)
    if R_c.ndim == 2:
        R_c = R_c.flatten()
    elif R_c.ndim > 2:
        raise ValueError(f"R_c has unexpected shape: {R_c.shape}")
    # MATLAB: for i=1:2
    for i in range(1, 3):
        # MATLAB: R_c = R_c + Ac^i*r^i/factorial(i)*(c + Uc*r/(i+1));
        # Compute (c + Uc*r/(i+1)) first to get a column vector
        c_plus_Uc = c + Uc * r / (i + 1)
        # Ensure it's a column vector
        if c_plus_Uc.ndim == 1:
            c_plus_Uc = c_plus_Uc.reshape(-1, 1)
        # Compute Ac^i * (c + Uc*r/(i+1)) to get a column vector
        R_c_add = np.linalg.matrix_power(Ac, i) @ c_plus_Uc * (r ** i / np.math.factorial(i))
        # Ensure R_c_add is a 1D array
        if R_c_add.ndim == 2:
            R_c_add = R_c_add.flatten()
        R_c = R_c + R_c_add
    
    # First parametric order
    # Auxiliary value
    # MATLAB: M = eye(n)*r + Ac*r^2/2 + Ac^2*r^3/6;
    M = np.eye(n) * r + Ac * r ** 2 / 2 + np.linalg.matrix_power(Ac, 2) * r ** 3 / 6
    
    # Loop
    # MATLAB: for i=1:params
    R_g_list = []
    for i in range(params):
        # MATLAB: R_g(:,i) = (Ag(:,:,i)*r + Ac*Ag(:,:,i)*r^2/2 + Ag(:,:,i)*Ac*r^2/2) * c + ...
        #                    (Ag(:,:,i)*r^2/2 + Ac*Ag(:,:,i)*r^3/6 + Ag(:,:,i)*Ac*r^3/6) * Uc + ...
        #                    M * UG(:,i);
        Ag_i = Ag[:, :, i]
        # Use first generator of UG if i is out of bounds (UG might have fewer columns than params)
        ug_col_idx = min(i, UG.shape[1] - 1) if UG.size > 0 and UG.shape[1] > 0 else 0
        ug_col = UG[:, ug_col_idx] if UG.size > 0 else np.zeros((n, 1))
        # Ensure c, Uc, ug_col are column vectors
        if c.ndim == 1:
            c = c.reshape(-1, 1)
        if Uc.ndim == 1:
            Uc = Uc.reshape(-1, 1)
        if ug_col.ndim == 1:
            ug_col = ug_col.reshape(-1, 1)
        # Compute R_g_col with correct order of operations
        term1 = (Ag_i * r + Ac @ Ag_i * r ** 2 / 2 + Ag_i @ Ac * r ** 2 / 2) @ c
        term2 = (Ag_i * r ** 2 / 2 + Ac @ Ag_i * r ** 3 / 6 + Ag_i @ Ac * r ** 3 / 6) @ Uc
        term3 = M @ ug_col
        R_g_col = term1 + term2 + term3
        # Ensure R_g_col is a 1D array
        if R_g_col.ndim == 2:
            R_g_col = R_g_col.flatten()
        R_g_list.append(R_g_col)
    
    # Second parametric order
    # Same index (i,i)
    # MATLAB: for i=1:params
    for i in range(params):
        # MATLAB: Rtmp = Ag(:,:,i)^2*r^2/2*c + ...
        #                Ag(:,:,i)*r^3/6*Uc + ...
        #                (Ac*Ag(:,:,i) + Ag(:,:,i)*Ac)*r^3/6*UG(:,i);
        Ag_i = Ag[:, :, i]
        # Use first generator of UG if i is out of bounds
        ug_col_idx = min(i, UG.shape[1] - 1) if UG.size > 0 and UG.shape[1] > 0 else 0
        ug_col = UG[:, ug_col_idx] if UG.size > 0 else np.zeros((n, 1))
        # Ensure c, Uc, ug_col are column vectors
        if c.ndim == 1:
            c = c.reshape(-1, 1)
        if Uc.ndim == 1:
            Uc = Uc.reshape(-1, 1)
        if ug_col.ndim == 1:
            ug_col = ug_col.reshape(-1, 1)
        # Compute Rtmp with correct order of operations
        Rtmp = ((np.linalg.matrix_power(Ag_i, 2) @ c) * (r ** 2 / 2) +
                (Ag_i @ Uc) * (r ** 3 / 6) +
                ((Ac @ Ag_i + Ag_i @ Ac) @ ug_col) * (r ** 3 / 6))
        # Ensure Rtmp is a 1D array
        if Rtmp.ndim == 2:
            Rtmp = Rtmp.flatten()
        # MATLAB: R_g(:,end+1) = 0.5*Rtmp;
        R_g_list.append(0.5 * Rtmp)
        # MATLAB: R_c = R_c + 0.5*Rtmp;
        R_c = R_c + 0.5 * Rtmp
    
    # Different index
    # MATLAB: if (params>=2)
    if params >= 2:
        # MATLAB: ind = combinator(params,2,'c');
        # Note: combinator generates combinations. We'll use itertools.combinations
        from itertools import combinations
        ind = list(combinations(range(params), 2))
        # MATLAB: for i=1:length(ind(:,1))
        for comb in ind:
            # MATLAB: Atmp = Ag(:,:,ind(i,1))*Ag(:,:,ind(i,2)) + Ag(:,:,ind(i,2))*Ag(:,:,ind(i,1));
            Atmp = (Ag[:, :, comb[0]] @ Ag[:, :, comb[1]] + 
                    Ag[:, :, comb[1]] @ Ag[:, :, comb[0]])
            # MATLAB: R_g(:,end+1) = Atmp*r^2/2*c + ...
            #                        Atmp*r^3/6*Uc + ...
            #                        (Ac*Ag(:,:,ind(i,1)) + Ag(:,:,ind(i,1))*Ac)*r^3/6*UG(:,ind(i,2));
            # Use first generator of UG if comb[1] is out of bounds
            ug_col_idx = min(comb[1], UG.shape[1] - 1) if UG.size > 0 and UG.shape[1] > 0 else 0
            ug_col = UG[:, ug_col_idx] if UG.size > 0 else np.zeros((n, 1))
            # Ensure c, Uc, ug_col are column vectors
            if c.ndim == 1:
                c = c.reshape(-1, 1)
            if Uc.ndim == 1:
                Uc = Uc.reshape(-1, 1)
            if ug_col.ndim == 1:
                ug_col = ug_col.reshape(-1, 1)
            # Compute R_g_col with correct order of operations
            R_g_col = ((Atmp @ c) * (r ** 2 / 2) +
                       (Atmp @ Uc) * (r ** 3 / 6) +
                       ((Ac @ Ag[:, :, comb[0]] + Ag[:, :, comb[0]] @ Ac) @ ug_col) * (r ** 3 / 6))
            # Ensure R_g_col is a 1D array
            if R_g_col.ndim == 2:
                R_g_col = R_g_col.flatten()
            R_g_list.append(R_g_col)
    
    # Obtain zonotope
    # MATLAB: R_lowOrder = zonotope(R_c,R_g);
    if len(R_g_list) > 0:
        R_g = np.column_stack(R_g_list)
    else:
        R_g = np.zeros((n, 0))
    R_lowOrder = Zonotope(R_c, R_g)
    
    # HIGHER ORDER INDEPENDENT SOLUTION
    # Get state transition matrices
    # MATLAB: eZhigh = obj.mappingMatrixSet.highOrderZono;
    # MATLAB: eIhigh = obj.mappingMatrixSet.highOrderInt;
    eZhigh = sys.mappingMatrixSet['highOrderZono']
    eIhigh = sys.mappingMatrixSet['highOrderInt']
    # Get input transition matrices
    # MATLAB: eZhigh_input = obj.mappingMatrixSet.highOrderZonoInput;
    # MATLAB: eIhigh_input = obj.mappingMatrixSet.highOrderIntInput;
    eZhigh_input = sys.mappingMatrixSet['highOrderZonoInput']
    eIhigh_input = sys.mappingMatrixSet['highOrderIntInput']
    
    # Remaining reachable set
    # MATLAB: R_rem_state = eZhigh*zonotope(c) + eIhigh*zonotope(c);
    R_rem_state = eZhigh * Zonotope(c, np.zeros((n, 0))) + eIhigh * Zonotope(c, np.zeros((n, 0)))
    # MATLAB: R_rem_input = eZhigh_input*Uconst + eIhigh_input*Uconst;
    # Transform Uconst from input dimension to state space dimension using sys.B
    Uconst_state = sys.B * Uconst
    R_rem_input = eZhigh_input * Uconst_state + eIhigh_input * Uconst_state
    # MATLAB: R_rem = R_rem_state + R_rem_input;
    R_rem = R_rem_state + R_rem_input
    
    # STATE SOLUTION WITHOUT CENTER
    # MATLAB: Rinit_noCenter = Rinit + (-c);
    Rinit_noCenter = Rinit + (-c)
    # MATLAB: R_hom_state = obj.mappingMatrixSet.zono*Rinit_noCenter + obj.mappingMatrixSet.int*Rinit_noCenter;
    R_hom_state = (sys.mappingMatrixSet['zono'] * Rinit_noCenter + 
                   sys.mappingMatrixSet['int'] * Rinit_noCenter)
    
    # FINAL SOLUTION
    # MATLAB: Rhom_tp = R_hom_state + R_lowOrder + R_rem;
    Rhom_tp = R_hom_state + R_lowOrder + R_rem
    
    return Rhom_tp
