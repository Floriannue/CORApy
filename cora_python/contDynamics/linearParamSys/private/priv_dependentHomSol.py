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
    # MATLAB: params = obj.A.numgens;
    from cora_python.matrixSet.matZonotope.numgens import numgens
    params = numgens(sys.A)
    
    # SECOND ORDER DEPENDENT SOLUTION
    # Zero parametric order
    # MATLAB: R_c = c + Uc*r;
    R_c = c + Uc * r
    # MATLAB: for i=1:2
    for i in range(1, 3):
        # MATLAB: R_c = R_c + Ac^i*r^i/factorial(i)*(c + Uc*r/(i+1));
        R_c = R_c + (np.linalg.matrix_power(Ac, i) * r ** i / np.math.factorial(i) * 
                     (c + Uc * r / (i + 1)))
    
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
        R_g_col = ((Ag_i * r + Ac @ Ag_i * r ** 2 / 2 + Ag_i @ Ac * r ** 2 / 2) @ c +
                   (Ag_i * r ** 2 / 2 + Ac @ Ag_i * r ** 3 / 6 + Ag_i @ Ac * r ** 3 / 6) @ Uc +
                   M @ UG[:, i])
        R_g_list.append(R_g_col)
    
    # Second parametric order
    # Same index (i,i)
    # MATLAB: for i=1:params
    for i in range(params):
        # MATLAB: Rtmp = Ag(:,:,i)^2*r^2/2*c + ...
        #                Ag(:,:,i)*r^3/6*Uc + ...
        #                (Ac*Ag(:,:,i) + Ag(:,:,i)*Ac)*r^3/6*UG(:,i);
        Ag_i = Ag[:, :, i]
        Rtmp = (np.linalg.matrix_power(Ag_i, 2) * r ** 2 / 2 @ c +
                Ag_i * r ** 3 / 6 @ Uc +
                (Ac @ Ag_i + Ag_i @ Ac) * r ** 3 / 6 @ UG[:, i])
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
            R_g_col = (Atmp * r ** 2 / 2 @ c +
                      Atmp * r ** 3 / 6 @ Uc +
                      (Ac @ Ag[:, :, comb[0]] + Ag[:, :, comb[0]] @ Ac) * r ** 3 / 6 @ UG[:, comb[1]])
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
    R_rem_input = eZhigh_input * Uconst + eIhigh_input * Uconst
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
