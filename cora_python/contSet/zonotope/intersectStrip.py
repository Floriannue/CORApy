"""
intersectStrip - computes the intersection between one zonotope and
   list of strips according to [1] and [2]
   the strip is defined as | Cx-y | <= phi

Syntax:
    Zres = intersectStrip(Z,C,phi,y,varargin)

Inputs:
    Z - zonotope object
    C - matrix of normal vectors of strips
    phi - vector of widths of strips
    y - center of intersected strips
    varargin - methods to calculate the weights
               - 'normGen' (default, analytical solution)
               - 'svd'
               - 'radius'
               - 'alamo-volume' according to [3]
               - 'alamo-FRad' according to [3]
               - 'wang-FRad' according to [4]  auxiliary values as a struct
               - 'bravo' accroding to [5]
               - or lambda value

Outputs:
    res - boolean whether obj is contained in Z, or not

Example: (three strips and one zonotope)
    C = [1 0; 0 1; 1 1];
    phi = [5; 3; 3];
    y = [-2; 2; 2];

    Z = zonotope([1 2 2 2 6 2 8;1 2 2 0 5 0 6 ]);
    res_zono = intersectStrip(Z,C,phi,y);

    % just for comparison
    poly = polytope([1 0;-1 0; 0 1;0 -1; 1 1;-1 -1],[3;7;5;1;5;1]);
    Zpoly = Z & poly;

    figure; hold on 
    plot(Z,[1 2],'b','DisplayName','zonotope');
    plot(poly,[1 2],'k*','DisplayName','Strips');
    plot(Zpoly,[1 2],'r','DisplayName','zono&strip');
    plot(res_zono,[1 2],'g','DisplayName','zonoStrips');
    legend();

References:
    [1] V. T. H. Le, C. Stoica, T. Alamo, E. F. Camacho, and
        D. Dumur. Zonotope-based set-membership estimation for
        multi-output uncertain systems. In Proc. of the IEEE
        International Symposium on Intelligent Control (ISIC),
        pages 212–217, 2013.
    [2] Amr Alanwar, Jagat Jyoti Rath, Hazem Said, Matthias Althoff
        Distributed Set-Based Observers Using Diffusion Strategy
    [3] T. Alamo, J. M. Bravo, and E. F. Camacho. Guaranteed
        state estimation by zonotopes. Automatica, 41(6):1035–1043,
        2005.
    [4] Ye Wang, Vicenç Puig, and Gabriela Cembrano. Set-
        membership approach and Kalman observer based on
        zonotopes for discrete-time descriptor systems. Automatica,
        93:435-443, 2018.
    [5] J. M. Bravo, T. Alamo, and E. F. Camacho. Bounded error 
        identification of systems with time-varying parameters. IEEE 
        Transactions on Automatic Control, 51(7):1144–1150, 2006.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       09-March-2020 (MATLAB)
Last update:   29-March-2023 (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import Union, Optional, Any, Dict
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.converter.CORAlinprog import CORAlinprog
from scipy.optimize import minimize
from .zonotope import Zonotope


def intersectStrip(Z: Zonotope, C: np.ndarray, phi: np.ndarray, y: np.ndarray, 
                  method: Union[str, float, Dict] = 'normGen') -> Zonotope:
    """
    Computes the intersection between one zonotope and list of strips.
    
    Args:
        Z: Zonotope object
        C: Matrix of normal vectors of strips
        phi: Vector of widths of strips
        y: Center of intersected strips
        method: Method to calculate the weights (default: 'normGen')
        
    Returns:
        Zonotope: Result of intersection
        
    Raises:
        CORAerror: If inputs are invalid or computation fails
    """
    # Check input arguments
    if not isinstance(Z, Zonotope):
        raise CORAerror('CORA:wrongValue', 'first', 'zonotope')
    
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        raise CORAerror('CORA:wrongValue', 'second', 'numeric matrix')
    
    if not isinstance(phi, np.ndarray) or phi.ndim != 2:
        raise CORAerror('CORA:wrongValue', 'third', 'numeric column')
    
    if not isinstance(y, np.ndarray) or y.ndim != 2:
        raise CORAerror('CORA:wrongValue', 'fourth', 'numeric column')
    
    # Check dimensions
    if Z.dim() != C.shape[1]:
        raise CORAerror('CORA:dimensionMismatch', Z, C)
    
    if C.shape[0] != phi.shape[0]:
        raise CORAerror('CORA:dimensionMismatch', C, phi)
    
    if C.shape[0] != y.shape[0]:
        raise CORAerror('CORA:dimensionMismatch', C, y)
    
    # Parse method
    if isinstance(method, (int, float)) or (isinstance(method, np.ndarray) and method.size > 0):
        lambda_val = method
        return _aux_zonotopeFromLambda(Z, phi, C, y, lambda_val)
    
    elif isinstance(method, dict):
        aux = method
        _aux_checkAuxStruct(aux)
        method = aux['method']
    
    if not isinstance(method, str) or method not in ['normGen', 'svd', 'radius', 'alamo-FRad', 
                      'alamo-volume', 'wang-FRad', 'bravo']:
        raise CORAerror('CORA:wrongValue', 'fifth', f"Unknown method '{method}'")
    
    # Methods for finding good lambda values
    if method in ['svd', 'radius']:
        Zres = _aux_methodSvd(Z, C, phi, y, method)
    
    elif method == 'alamo-volume':
        Zres = _aux_methodAlamoVolume(Z, C, phi, y)
    
    elif method == 'alamo-FRad':
        Zres = _aux_methodAlamoFRad(Z, C, phi, y)
    
    elif method == 'wang-FRad':
        Zres = _aux_methodWangFRad(Z, C, phi, y, aux)
    
    elif method == 'bravo':
        Zres = _aux_methodBravo(Z, C, phi, y)
    
    elif method == 'normGen':
        Zres = _aux_methodNormGen(Z, C, phi, y)
    
    else:
        raise CORAerror('CORA:wrongValue', 'fifth', f"Unknown method '{method}'")
    
    return Zres


def _aux_methodSvd(Z: Zonotope, C: np.ndarray, phi: np.ndarray, y: np.ndarray, method: str) -> Zonotope:
    """SVD and radius method"""
    
    G = Z.G
    dim, _ = G.shape
    
    # Initialize lambda
    lambda0 = np.zeros((dim, len(phi)))
    
    # Find optimal lambda
    def fun(lambda_vec_param):
        """Embedded function to be minimized for optimal lambda"""
        lambda_mat = lambda_vec_param.reshape(dim, -1)
        part1 = np.eye(len(Z.center()))
        part2 = np.zeros((dim, len(phi)))
        
        for i in range(len(phi)):
            part1 = part1 - lambda_mat[:, i:i+1] @ C[i:i+1, :]
            part2[:, i:i+1] = phi[i] * lambda_mat[:, i:i+1]
        
        part1 = part1 @ G
        G_new = np.hstack([part1, part2])
        
        if method == 'svd':
            return np.sum(np.linalg.svd(G_new, compute_uv=False))
        elif method == 'radius':
            return Zonotope(np.zeros((len(Z.center()), 1)), G_new).radius()
    
    # Minimize using scipy
    result = minimize(fun, lambda0.flatten(), method='BFGS', options={'disp': False})
    lambda_opt = result.x.reshape(dim, -1)
    
    # Resulting zonotope
    return _aux_zonotopeFromLambda(Z, phi, C, y, lambda_opt)


def _aux_methodAlamoVolume(Z: Zonotope, C: np.ndarray, phi: np.ndarray, y: np.ndarray) -> Zonotope:
    """Volume minimization according to Alamo, [3]"""
    
    # Warning if more than one strip is used
    if len(phi) > 1:
        raise CORAerror('CORA:specialError',
                       'Alamo method should only be used for single strips to ensure convergence')
    
    G = Z.G
    dim, nrGens = G.shape
    
    # Implement volume expression from [3], Vol(\hat{X}(lambda)) (last eq.
    # before Sec. 7); from now on referred to as (V)
    # Obtain possible combinations of generators
    from itertools import combinations
    
    comb_A = list(combinations(range(nrGens), dim))
    comb_B = list(combinations(range(nrGens), dim-1))
    
    a_obj = 0
    # |1-c'*lambda| can be pulled out from first summation in (V)
    for iComb in comb_A:
        a_obj = a_obj + 2**dim * abs(np.linalg.det(G[:, list(iComb)]))
    
    b_obj = np.zeros(len(comb_B))
    Bp_cstr = np.zeros((len(comb_B), dim))
    
    # As we set z_i = |v_i'*lambda| (also see below), each summand in
    # second summation of (V) is exactly the coefficient for z_i
    for iComb, comb in enumerate(comb_B):
        Bi = G[:, list(comb)]
        if np.linalg.matrix_rank(Bi) < dim - 1:
            b_obj[iComb] = 0
            continue
        
        # rank(Bi) = n-1 => there exists exactly 1 vector orthogonal to
        # image(Bi) (vi'*Bi = 0)
        # Compute null space manually since numpy.linalg.null_space may not exist
        U, S, Vt = np.linalg.svd(Bi.T)
        # Find the singular value closest to zero
        min_idx = np.argmin(S)
        vi = U[:, min_idx:min_idx+1]  # Ensure vi is a column vector
        # Ensure vi has the same number of rows as Bi for concatenation
        if vi.shape[0] != Bi.shape[0]:
            # If dimensions don't match, we need to handle this case
            # This should not happen in normal cases, but let's add a safety check
            continue
        b_obj[iComb] = 2**dim * phi[0] * abs(np.linalg.det(np.hstack([Bi, vi])))
        Bp_cstr[iComb, :] = vi.flatten()
    
    # Clean-up zeros
    ind_0 = b_obj == 0
    b_obj = b_obj[~ind_0]
    Bp_cstr = Bp_cstr[~ind_0, :]
    nz = len(b_obj)
    
    # Formulate linear program
    # opt variables:lambda, r, z (r=|1-phi'*lambda|,z_i = |v_i'*lambda|)
    # x = [lambda;r;z];
    f_obj = np.concatenate([np.zeros(dim), [a_obj], b_obj])
    
    # Constraints for |1-phi'*lambda|<=r (same as =r since we want to
    # minimize the expression)
    # Note: alamo-volume method is only for single strips, so use C[0:1, :]
    Cr_cstr = np.vstack([
        np.hstack([-C[0:1, :], -np.ones((1, 1)), np.zeros((1, nz))]),
        np.hstack([C[0:1, :], -np.ones((1, 1)), np.zeros((1, nz))])
    ])
    dr_cstr = np.array([-1, 1])
    
    # Constraints handling z_i = |v_i'*lambda|
    Cz_cstr = np.vstack([
        np.hstack([Bp_cstr, np.zeros((nz, 1)), -np.eye(nz)]),
        np.hstack([-Bp_cstr, np.zeros((nz, 1)), -np.eye(nz)])
    ])
    dz_cstr = np.zeros(2 * nz)
    
    # Collect all constraints
    C_cstr = np.vstack([Cr_cstr, Cz_cstr])
    d_cstr = np.concatenate([dr_cstr, dz_cstr])
    
    # Solve linear program
    problem = {
        'f': f_obj,
        'Aineq': C_cstr,
        'bineq': d_cstr,
        'Aeq': None,
        'beq': None,
        'lb': None,
        'ub': None
    }
    
    x_opt, fval, exitflag, output, lambda_out = CORAlinprog(problem)
    
    if exitflag != 1:
        print(f"Warning: Linear programming failed with exitflag {exitflag}")
        # Fallback to a simple method
        lambda_val = np.zeros(dim)
    else:
        # Extract lambda
        lambda_val = x_opt[:dim]
    
    # Resulting zonotope
    return _aux_zonotopeFromLambda(Z, phi, C, y, lambda_val)


def _aux_methodAlamoFRad(Z: Zonotope, C: np.ndarray, phi: np.ndarray, y: np.ndarray) -> Zonotope:
    """F-radius minimization according to Alamo, [3]"""
    
    G = Z.G
    
    # Auxiliary variables
    aux1 = G @ G.T
    aux2 = aux1 @ C[0, :].T
    aux3 = C[0, :] @ aux1 @ C[0, :].T + phi[0]**2
    
    # Return lambda
    lambda_val = aux2 / aux3
    
    # Resulting zonotope
    Zres = _aux_zonotopeFromLambda(Z, phi, C, y, lambda_val)
    
    # Warning
    if len(phi) > 1:
        print('Alamo method should only be used for single strips to ensure convergence')
    
    return Zres


def _aux_methodWangFRad(Z: Zonotope, C: np.ndarray, phi: np.ndarray, y: np.ndarray, aux: Dict) -> Zonotope:
    """F-radius minimization according to Wang, Theorem 2 in [4]"""
    
    G = Z.G
    
    # Auxiliary variables
    P = G @ G.T
    Q_w = aux['E'] @ aux['E'].T
    Q_v = aux['F'] @ aux['F'].T
    
    # eq. (15)
    Rbar = aux['A'] @ P @ aux['A'].T + Q_w
    
    # eq. (14)
    S = aux['C'] @ Rbar @ aux['C'].T + Q_v
    
    # eq. (13)
    L = Rbar @ aux['C'].T
    
    # eq. (12)
    lambda_val = L @ np.linalg.inv(S)
    
    # Resulting zonotope
    return _aux_zonotopeFromLambda(Z, phi, C, y, lambda_val)


def _aux_methodBravo(Z: Zonotope, C: np.ndarray, phi: np.ndarray, y: np.ndarray) -> Zonotope:
    """Method according to Bravo, [5]"""
    
    # Property 1 in [5]
    # Obtain center of zonotope
    p = Z.center()
    G = Z.G
    dim, nrGens = G.shape
    
    c_cell = [None] * (nrGens + 1)
    G_cell = [None] * (nrGens + 1)
    volApproxZ = np.zeros(nrGens + 1)
    
    # Loop through generators
    for j in range(nrGens + 1):
        # Normal vector of strip and generator are not perpendicular
        if j > 0 and abs(C[0, :] @ G[:, j-1]) > 1e10 * np.finfo(float).eps:
            # Init T
            T = np.zeros((dim, nrGens))
            for iGen in range(nrGens):
                if iGen != j - 1:
                    T[:, iGen] = (G[:, iGen] - 
                                 (C[0, :] @ G[:, iGen]) / (C[0, :] @ G[:, j-1]) * G[:, j-1])
                else:
                    T[:, iGen] = phi[0] / (C[0, :] @ G[:, j-1]) * G[:, j-1]
            v = p + ((y[0] - C[0, :] @ p) / (C[0, :] @ G[:, j-1])) * G[:, j-1]
        # First generator or normal vector of strip and generator are perpendicular
        else:
            v = p  # new center
            T = G  # new generators
        
        # Save center and generator
        c_cell[j] = v
        G_cell[j] = T
        
        # Approximate volume of obtained zonotope
        volApproxZ[j] = np.linalg.det(G_cell[j] @ G_cell[j].T)
    
    # Find zonotope with minimum approximated volume
    ind = np.argmin(volApproxZ)
    
    # Return best zonotope
    Zres = Zonotope(c_cell[ind], G_cell[ind])
    
    # Warning
    if len(phi) > 1:
        print('Bravo method is only applied to the first strip')
    
    return Zres


def _aux_methodNormGen(Z: Zonotope, C: np.ndarray, phi: np.ndarray, y: np.ndarray) -> Zonotope:
    """Norm Gen method"""
    G = Z.G
    
    # Find the analytical solution
    gamma = np.eye(C.shape[0])
    num = G @ G.T @ C.T
    den = C @ G @ G.T @ C.T
    for iStrip in range(C.shape[0]):
        den = den + gamma[:, iStrip:iStrip+1] @ phi[iStrip]**2 @ gamma[:, iStrip:iStrip+1].T
    
    lambda_val = num @ np.linalg.inv(den)
    
    # Resulting zonotope
    return _aux_zonotopeFromLambda(Z, phi, C, y, lambda_val)


def _aux_zonotopeFromLambda(Z: Zonotope, phi: np.ndarray, C: np.ndarray, y: np.ndarray, 
                           Lambda: np.ndarray) -> Zonotope:
    """Return zonotope from a given lambda vector, see Prop. 1 of [1]"""
    # Strips: |Cx − y| <= phi
    # Zonotope: Z = c+G[-1,1]^o
    
    # New center
    c_new = Z.center() + Lambda @ (y - C @ Z.center())
    
    # New generators
    I = np.eye(len(c_new))
    G_new = np.hstack([(I - Lambda @ C) @ Z.generators(), Lambda @ np.diag(phi.flatten())])
    
    # Resulting zonotope
    return Zonotope(c_new, G_new)


def _aux_checkAuxStruct(aux: Dict):
    """Error handling for auxiliary structure"""
    # Correct method specified?
    if 'method' not in aux:
        raise CORAerror('CORA:wrongFieldValue', 'aux.method', ['wang-FRad'])
    
    # Is E numeric?
    if 'E' not in aux:
        raise CORAerror('CORA:wrongFieldValue', 'aux.E', 'numeric')
    
    # Is F numeric?
    if 'F' not in aux:
        raise CORAerror('CORA:wrongFieldValue', 'aux.F', 'numeric')
    
    # Is A numeric?
    if 'A' not in aux:
        raise CORAerror('CORA:wrongFieldValue', 'aux.A', 'numeric')
    
    # Is C numeric?
    if 'C' not in aux:
        raise CORAerror('CORA:wrongFieldValue', 'aux.C', 'numeric') 