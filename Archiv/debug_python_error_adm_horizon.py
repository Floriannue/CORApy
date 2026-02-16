"""
Debug script to trace error_adm_horizon growth in Python
This traces the computation path step by step to compare with MATLAB
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.contDynamics.nonlinearSys.nonlinearSys import NonlinearSys

print('=== Python Debug: error_adm_horizon Growth ===')

# Setup similar to test
params = {
    'tStart': 0,
    'tFinal': 500,
    'R0': Zonotope(np.zeros((5, 1)), 0.1 * np.eye(5)),
    'U': Zonotope(np.zeros((1, 1)), np.zeros((1, 1)))
}

options = {
    'alg': 'lin',
    'tensorOrder': 3,
    'timeStep': 0.1,
    'taylorTerms': 10,
    'zonotopeOrder': 50,
    'reductionTechnique': 'adaptive',
    'redFactor': 0.9,
    'decrFactor': 0.5,
    'minorder': 1,
    'maxError': 0.1 * np.ones((5, 1)),
    'zetaphi': [0.5, 0.5],
    'zetaK': 0.1,
    'orders': np.ones((5, 1)),
    'i': 1
}

# Initialize error_adm_horizon to a huge value
options['error_adm_horizon'] = 1e+75 * np.ones((5, 1))

print(f"Starting with error_adm_horizon = [{options['error_adm_horizon'][0, 0]:.6e}; "
      f"{options['error_adm_horizon'][1, 0]:.6e}; {options['error_adm_horizon'][2, 0]:.6e}; "
      f"{options['error_adm_horizon'][3, 0]:.6e}; {options['error_adm_horizon'][4, 0]:.6e}]")

try:
    error_adm = options['error_adm_horizon']
    
    print('\n--- Step 1: Create Verror from error_adm ---')
    errG = np.diag(error_adm.flatten())
    Verror = Zonotope(np.zeros_like(error_adm), errG[:, np.any(errG, axis=0)])
    print(f"Verror center: [{Verror.center()[0, 0]:.6e}; {Verror.center()[1, 0]:.6e}; "
          f"{Verror.center()[2, 0]:.6e}; {Verror.center()[3, 0]:.6e}; {Verror.center()[4, 0]:.6e}]")
    print(f"Verror generators shape: {Verror.generators().shape}")
    print(f"Verror generators max abs: {np.max(np.abs(Verror.generators())):.6e}")
    
    print('\n--- Step 2: Compute RallError via errorSolution_adaptive ---')
    print(f"Verror radius estimate: {np.max(np.sum(np.abs(Verror.generators()), axis=1)):.6e}")
    
    print('\n--- Step 3: Check what happens in priv_abstractionError_adaptive ---')
    # Create a dummy Rmax that would result from huge RallError
    Rmax_center = np.ones((5, 1))
    Rmax_gen = 1e+39 * np.eye(5)  # Simulate huge generators
    Rmax = Zonotope(Rmax_center, Rmax_gen)
    print(f"Rmax center: [{Rmax_center[0, 0]:.6e}; {Rmax_center[1, 0]:.6e}; "
          f"{Rmax_center[2, 0]:.6e}; {Rmax_center[3, 0]:.6e}; {Rmax_center[4, 0]:.6e}]")
    print(f"Rmax generators max abs: {np.max(np.abs(Rmax_gen)):.6e}")
    print(f"Rmax radius estimate: {np.max(np.sum(np.abs(Rmax_gen), axis=1)):.6e}")
    
    # Check what happens when we reduce Rmax
    print('\n--- Step 4: Reduce Rmax ---')
    Rred_res = Rmax.reduce('adaptive', np.sqrt(options['redFactor']))
    Rred = Rred_res[0] if isinstance(Rred_res, tuple) else Rred_res
    print(f"Rred generators max abs: {np.max(np.abs(Rred.generators())):.6e}")
    print(f"Rred radius estimate: {np.max(np.sum(np.abs(Rred.generators()), axis=1)):.6e}")
    
    # Check what happens with Z = cartProd(Rred, U)
    print('\n--- Step 5: Create Z = cartProd(Rred, U) ---')
    Z = Rred.cartProd_(params['U'])
    print(f"Z generators max abs: {np.max(np.abs(Z.generators())):.6e}")
    print(f"Z radius estimate: {np.max(np.sum(np.abs(Z.generators()), axis=1)):.6e}")
    
    # Check what happens with quadMap
    print('\n--- Step 6: Compute errorSec = 0.5 * quadMap(Z, H) ---')
    # We need H - for debugging, use identity
    H = [np.eye(5) for _ in range(5)]
    errorSec = 0.5 * Z.quadMap(H)
    print(f"errorSec center: [{errorSec.center()[0, 0]:.6e}; {errorSec.center()[1, 0]:.6e}; "
          f"{errorSec.center()[2, 0]:.6e}; {errorSec.center()[3, 0]:.6e}; {errorSec.center()[4, 0]:.6e}]")
    print(f"errorSec generators max abs: {np.max(np.abs(errorSec.generators())):.6e}")
    print(f"errorSec radius estimate: {np.max(np.sum(np.abs(errorSec.generators()), axis=1)):.6e}")
    
    # Check what happens with VerrorDyn
    print('\n--- Step 7: Compute VerrorDyn and trueError ---')
    VerrorDyn = errorSec  # Simplified: no errorLagr
    VerrorDyn_res = VerrorDyn.reduce('adaptive', 10 * options['redFactor'])
    VerrorDyn = VerrorDyn_res[0] if isinstance(VerrorDyn_res, tuple) else VerrorDyn_res
    print(f"VerrorDyn center: [{VerrorDyn.center()[0, 0]:.6e}; {VerrorDyn.center()[1, 0]:.6e}; "
          f"{VerrorDyn.center()[2, 0]:.6e}; {VerrorDyn.center()[3, 0]:.6e}; {VerrorDyn.center()[4, 0]:.6e}]")
    print(f"VerrorDyn generators max abs: {np.max(np.abs(VerrorDyn.generators())):.6e}")
    print(f"VerrorDyn radius estimate: {np.max(np.sum(np.abs(VerrorDyn.generators()), axis=1)):.6e}")
    
    trueError = np.abs(VerrorDyn.center()) + np.sum(np.abs(VerrorDyn.generators()), axis=1).reshape(-1, 1)
    print(f"trueError: [{trueError[0, 0]:.6e}; {trueError[1, 0]:.6e}; "
          f"{trueError[2, 0]:.6e}; {trueError[3, 0]:.6e}; {trueError[4, 0]:.6e}]")
    print(f"trueError max: {np.max(trueError):.6e}")
    
    # Check perfIndCurr
    print('\n--- Step 8: Compute perfIndCurr = max(trueError ./ error_adm) ---')
    with np.errstate(divide='ignore', invalid='ignore'):
        perfIndCurr_ratio = trueError / error_adm
        perfIndCurr = np.max(perfIndCurr_ratio)
        if np.isnan(perfIndCurr):
            perfIndCurr = 0
    print(f"perfIndCurr: {perfIndCurr:.6e}")
    print(f"perfIndCurr <= 1: {perfIndCurr <= 1}")
    print(f"np.isinf(perfIndCurr): {np.isinf(perfIndCurr)}")
    print(f"np.isnan(perfIndCurr): {np.isnan(perfIndCurr)}")
    
    print('\n=== Python handles huge values naturally ===')
    print('If perfIndCurr > 1, the inner loop continues and error_adm = 1.1 * trueError')
    print('This would make error_adm even larger, causing the cycle to continue.')
    
except Exception as e:
    print(f'\n=== ERROR CAUGHT ===')
    print(f'Type: {type(e).__name__}')
    print(f'Message: {str(e)}')
    import traceback
    traceback.print_exc()
