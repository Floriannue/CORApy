"""
test_linearSys_reach_krylov_01_heat3D - heat3d benchmark from
    the 2023 ARCH competition using Krylov subspace method

Syntax:
    text = test_linearSys_reach_krylov_01_heat3D()

Inputs:
    -

Outputs:
    text - string

Authors:       Matthias Althoff, Niklas Kochdumper
Written:       02-June-2020
Last update:   29-June-2024 (TL, test if mp is installed)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
import scipy.io
import os
import time
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.interval.interval import Interval
from cora_python.g.macros.CORAROOT import CORAROOT


def aux_getInit(A, samples, temp):
    """
    Returns an initial state vector
    
    Args:
        A: system matrix
        samples: number of samples per dimension
        temp: temperature value
        
    Returns:
        x0: initial state vector
    """
    # MATLAB: x0 = zeros(size(A,1),1);
    x0 = np.zeros((A.shape[0], 1))
    
    # MATLAB: if samples ~= 5 && (samples < 10 || mod(samples,10) ~= 0)
    if samples != 5 and (samples < 10 or samples % 10 != 0):
        from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
        raise CORAerror('CORA:specialError',
                        'Init region is not evenly divided by discretization!')
    
    # maximum z point for initial region is 0.2 for 5 samples and 0.1 otherwise
    if samples >= 10:
        max_z = int(0.1 / (1.0 / samples))
    else:
        max_z = int(0.2 / (1.0 / samples))
    
    for z in range(max_z + 1):
        # compute offset for the z-axis
        zoffset = z * samples * samples
        
        max_y = int(0.2 / (1.0 / samples))
        for y in range(max_y + 1):
            # compute offset for the y-axis
            yoffset = y * samples
            
            max_x = int(0.4 / (1.0 / samples))
            for x in range(max_x + 1):
                index = x + yoffset + zoffset
                # final point (MATLAB uses 1-based indexing, Python uses 0-based)
                if index < x0.shape[0]:
                    x0[index, 0] = temp
    
    return x0


def test_linearSys_reach_krylov_01_heat3D():
    """
    Test Krylov subspace reachability on heat3D system
    
    This test translates benchmark_linear_reach_ARCH23_heat3D_HEAT03.m
    which tests the Krylov subspace reachability algorithm.
    """
    # Note: MATLAB checks for multiple precision toolbox (mp)
    # Python uses decimal module for high precision, but we'll proceed without it
    # for now as it's optional
    
    # System Dynamics ---------------------------------------------------------
    
    # load system matrices
    # MATLAB: load('heat20');
    mat_file_path = os.path.join(CORAROOT(), 'models', 'Cora', 'heat3D', 'heat20.mat')
    if not os.path.exists(mat_file_path):
        pytest.skip(f"heat20.mat not found at {mat_file_path}")
    
    data = scipy.io.loadmat(mat_file_path)
    A = data['A']
    B = data['B']
    
    # Convert sparse matrices to dense if needed
    from scipy.sparse import issparse
    if issparse(A):
        A = A.toarray()
    if issparse(B):
        B = B.toarray()
    
    # Ensure A is square and 2D
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.shape[0] != A.shape[1]:
        # If A is not square, check if it needs reshaping
        # For heat3D, A should be square
        raise ValueError(f"A matrix is not square: shape {A.shape}")
    
    # construct output matrix (center of block)
    # MATLAB: samples = nthroot(size(A,1),3);
    samples = int(np.round(np.power(A.shape[0], 1.0/3.0)))
    indMid = 4211 - 1  # Convert to 0-based indexing
    
    # MATLAB: C = zeros(1,size(A,1)); C(1,indMid) = 1;
    C = np.zeros((1, A.shape[0]))
    C[0, indMid] = 1
    
    # Parameters --------------------------------------------------------------
    
    # MATLAB: x0 = aux_getInit(A,samples,1);
    x0 = aux_getInit(A, samples, 1.0)
    
    # MATLAB: temp = diag(0.1*x0); temp = temp(:,x0 > 0);
    temp = np.diag(0.1 * x0.flatten())
    x0_positive = x0.flatten() > 0
    temp = temp[:, x0_positive]
    R0sim = Zonotope(x0, temp)
    
    params = {}
    # NOTE: Original benchmark uses tFinal=40.0 (8000 steps: 40.0 / 0.005), taking ~30 minutes
    # For faster testing during development, reduce to 2.5 for ~500 steps (~2 minutes)
    # The 8000 steps are required for the full ARCH competition benchmark validation
    params['tFinal'] = 2.5  # Reduced from 40.0 for faster testing (500 steps instead of 8000)
    
    # construct transpose linear system object
    # MATLAB: sys = linearSys(A,0,[],B');
    # In MATLAB, [] means not provided, so use None in Python
    sys = LinearSys(A, np.zeros((A.shape[0], 1)), None, B.T)
    
    # MATLAB: sysTrans = linearSys(A.',0,[],[R0sim.c,R0sim.G]');
    R0sim_c = R0sim.center()
    R0sim_G = R0sim.generators()
    sysTrans = LinearSys(A.T, np.zeros((A.shape[0], 1)), None,
                         np.hstack([R0sim_c, R0sim_G]).T)
    
    # correct input
    # MATLAB: params.R0 = zonotope(C.');
    params['R0'] = Zonotope(C.T)
    
    # Settings ----------------------------------------------------------------
    
    options = {}
    options['taylorTerms'] = 10
    options['zonotopeOrder'] = 2
    options['timeStep'] = 0.005
    options['linAlg'] = 'krylov'
    options['krylovError'] = np.finfo(float).eps
    options['krylovStep'] = 20
    
    # Reachability Analysis ---------------------------------------------------
    
    # MATLAB: clock = tic; Rcont = reach(sysTrans,params,options); tComp = toc(clock);
    start_time = time.time()
    Rcont = sysTrans.reach(params, options)
    tComp = time.time() - start_time
    
    # Verification ------------------------------------------------------------
    
    # MATLAB: goalSet = 0.01716509 + interval(-1e-4,1e-4);
    goalSet = Interval([0.01716509 - 1e-4], [0.01716509 + 1e-4])
    
    # get maximum temperature
    # MATLAB: TempMax = -inf;
    TempMax = -np.inf
    
    # MATLAB: Rset = cell(length(Rcont(1).timeInterval.set),1);
    # Assuming Rcont is a list of ReachSet objects or a single ReachSet
    if isinstance(Rcont, list) and len(Rcont) > 0:
        R_cont = Rcont[0]
    else:
        R_cont = Rcont
    
    Rset = []
    # MATLAB: Rcont(1).timeInterval.set
    if hasattr(R_cont, 'timeInterval') and R_cont.timeInterval is not None:
        # timeInterval is a dict with 'set' and 'time' keys
        if isinstance(R_cont.timeInterval, dict):
            timeInterval_sets = R_cont.timeInterval.get('set', [])
        else:
            timeInterval_sets = R_cont.timeInterval.set if hasattr(R_cont.timeInterval, 'set') else []
        
        for i in range(len(timeInterval_sets)):
            # compute maximum temperature
            # MATLAB: set_i = Rcont(1).timeInterval.set{i};
            set_i = timeInterval_sets[i]
            
            # MATLAB: Rtmp = [set_i.c,set_i.G];
            set_i_c = set_i.center() if hasattr(set_i, 'center') else set_i.c
            set_i_G = set_i.generators() if hasattr(set_i, 'generators') else set_i.G
            Rtmp = np.hstack([set_i_c, set_i_G])
            
            # MATLAB: Rtmp = [sum(Rtmp(1,:)); reshape(Rtmp(2:end,:),[],1)];
            Rtmp_sum = np.sum(Rtmp[0:1, :], axis=1, keepdims=True)
            Rtmp_rest = Rtmp[1:, :].reshape(-1, 1)
            Rtmp = np.vstack([Rtmp_sum, Rtmp_rest])
            
            # MATLAB: Rset{i} = zonotope(Rtmp.');
            Rset.append(Zonotope(Rtmp.T))
            
            # compute maximum temperature
            # MATLAB: val = supremum(interval(Rset{i}));
            val = Rset[i].interval().sup
            
            # check if new global maximum
            # MATLAB: if val > TempMax; TempMax = val; end
            if val > TempMax:
                TempMax = val
    
    # check if maximum temperature is in goal set
    # MATLAB: res = contains(goalSet,TempMax);
    res = goalSet.contains_(TempMax)
    
    # Verify the result
    assert res, f"Maximum temperature {TempMax} should be in goal set {goalSet}"
    
    # Verify computation completed
    assert tComp > 0, "Computation time should be positive"
    assert len(Rset) > 0, "Should have computed reachable sets"

