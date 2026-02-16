"""
Debug script to trace intermediate values in linReach_adaptive (Python)
This tracks all intermediate values step by step to compare with MATLAB
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.contDynamics.nonlinearSys.nonlinearSys import NonlinearSys

print('=== Tracing Intermediate Values in linReach_adaptive (Python) ===\n')

# Open file for detailed logging
with open('python_intermediate_values.txt', 'w') as f:
    f.write('=== Python Intermediate Values Trace ===\n\n')
    
    # Setup test case - use a simpler case that we can trace
    params = {
        'tStart': 0,
        'tFinal': 10,
        'R0': Zonotope(np.zeros((2, 1)), 0.1 * np.eye(2)),
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
        'maxError': 0.1 * np.ones((2, 1)),
        'zetaphi': [0.5, 0.5],
        'zetaK': 0.1,
        'orders': np.ones((2, 1)),
        'i': 1,
        'error_adm_horizon': 1e-10 * np.ones((2, 1))  # Start with small value
    }
    
    try:
        # Simulate one iteration of the inner loop
        error_adm = options['error_adm_horizon']
        print('Step 0: Initial error_adm_horizon')
        f.write('Step 0: Initial error_adm_horizon\n')
        print(f"  error_adm = [{error_adm[0, 0]:.15e}; {error_adm[1, 0]:.15e}]")
        f.write(f"  error_adm = [{error_adm[0, 0]:.15e}; {error_adm[1, 0]:.15e}]\n")
        
        # Create Verror
        print('\nStep 1: Create Verror from error_adm')
        f.write('\nStep 1: Create Verror from error_adm\n')
        errG = np.diag(error_adm.flatten())
        Verror = Zonotope(np.zeros_like(error_adm), errG[:, np.any(errG, axis=0)])
        Verror_center = Verror.center()
        Verror_gens = Verror.generators()
        print(f"  Verror center: [{Verror_center[0, 0]:.15e}; {Verror_center[1, 0]:.15e}]")
        f.write(f"  Verror center: [{Verror_center[0, 0]:.15e}; {Verror_center[1, 0]:.15e}]\n")
        print(f"  Verror generators shape: {Verror_gens.shape}")
        f.write(f"  Verror generators shape: {Verror_gens.shape}\n")
        print(f"  Verror generators max abs: {np.max(np.abs(Verror_gens)):.15e}")
        f.write(f"  Verror generators max abs: {np.max(np.abs(Verror_gens)):.15e}\n")
        
        print('\n=== Expected Computation Flow ===')
        f.write('\n=== Expected Computation Flow ===\n')
        print('1. error_adm -> Verror (zonotope with diag(error_adm))')
        print('2. Verror -> RallError (via errorSolution_adaptive)')
        print('3. Rlinti + RallError -> Rmax')
        print('4. Rmax -> Rred (reduced)')
        print('5. Rred, U -> Z (cartProd)')
        print('6. Z, H -> errorSec (0.5 * quadMap)')
        print('7. errorSec -> VerrorDyn (after reduction)')
        print('8. VerrorDyn -> trueError (abs(center) + sum(abs(generators)))')
        print('9. trueError, error_adm -> perfIndCurr (max(trueError ./ error_adm))')
        print('10. perfIndCurr -> convergence check')
        
        f.write('1. error_adm -> Verror (zonotope with diag(error_adm))\n')
        f.write('2. Verror -> RallError (via errorSolution_adaptive)\n')
        f.write('3. Rlinti + RallError -> Rmax\n')
        f.write('4. Rmax -> Rred (reduced)\n')
        f.write('5. Rred, U -> Z (cartProd)\n')
        f.write('6. Z, H -> errorSec (0.5 * quadMap)\n')
        f.write('7. errorSec -> VerrorDyn (after reduction)\n')
        f.write('8. VerrorDyn -> trueError (abs(center) + sum(abs(generators)))\n')
        f.write('9. trueError, error_adm -> perfIndCurr (max(trueError ./ error_adm))\n')
        f.write('10. perfIndCurr -> convergence check\n')
        
        print('\n=== Key Values to Track ===')
        f.write('\n=== Key Values to Track ===\n')
        print('At each inner loop iteration:')
        print('  - error_adm (input)')
        print('  - RallError radius (max sum of abs generators)')
        print('  - Rmax radius (max sum of abs generators)')
        print('  - Z radius (max sum of abs generators)')
        print('  - errorSec radius (max sum of abs generators)')
        print('  - VerrorDyn radius (max sum of abs generators)')
        print('  - trueError (vector)')
        print('  - perfIndCurr (scalar)')
        print('  - perfInds (array of perfIndCurr values)')
        
        f.write('At each inner loop iteration:\n')
        f.write('  - error_adm (input)\n')
        f.write('  - RallError radius (max sum of abs generators)\n')
        f.write('  - Rmax radius (max sum of abs generators)\n')
        f.write('  - Z radius (max sum of abs generators)\n')
        f.write('  - errorSec radius (max sum of abs generators)\n')
        f.write('  - VerrorDyn radius (max sum of abs generators)\n')
        f.write('  - trueError (vector)\n')
        f.write('  - perfIndCurr (scalar)\n')
        f.write('  - perfInds (array of perfIndCurr values)\n')
        
    except Exception as e:
        print(f'\n=== ERROR ===')
        print(f'Type: {type(e).__name__}')
        print(f'Message: {str(e)}')
        f.write(f'\n=== ERROR ===\n')
        f.write(f'Type: {type(e).__name__}\n')
        f.write(f'Message: {str(e)}\n')
        import traceback
        traceback.print_exc()
        f.write(traceback.format_exc())

print('\n=== Trace complete. Values saved to python_intermediate_values.txt ===')
