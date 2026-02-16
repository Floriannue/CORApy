"""
Test intermediate value tracking with jetEngine case
This enables tracking and runs a short reachability analysis
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import time
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.models.Cora.contDynamics.nonlinearSys.models.jetEngine import jetEngine

print('=== Testing Intermediate Value Tracking with jetEngine ===\n')

# Setup matching test_nonlinearSys_reach_adaptive_01_jetEngine
dim_x = 2

params = {}
params['tFinal'] = 1.0  # Short time for testing
params['R0'] = Zonotope(np.array([[1], [1]]), 0.1 * np.eye(dim_x))
params['U'] = Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0))

options = {}
options['alg'] = 'lin-adaptive'
# Note: tensorOrder is determined by _aux_initStepTensorOrder for first step
# It will be set to 2 or 3 based on comparison of L0_2 vs L0_3
# For testing, we can set it after initialization if needed
options['traceIntermediateValues'] = True  # Enable tracking
options['progress'] = True
options['progressInterval'] = 1  # More frequent updates for short test

# Create system
sys = NonlinearSys(name='jetEngine', fun=jetEngine, states=dim_x, inputs=1)

print('Running reach_adaptive with tracking enabled...')
print('This will create intermediate_values_step{N}_inner_loop.txt files\n')

try:
    start_time = time.time()
    result = sys.reach(params, options)
    tComp = time.time() - start_time
    
    # Handle different return types
    if isinstance(result, tuple):
        R, _, opt = result
    else:
        R = result
        opt = options
    
    print(f'[OK] reach_adaptive completed in {tComp:.2f} seconds')
    
    # Check if trace files were created
    import glob
    trace_files = glob.glob('intermediate_values_step*_inner_loop.txt')
    if trace_files:
        print(f'\n[OK] Found {len(trace_files)} trace file(s):')
        for f in sorted(trace_files):
            size = os.path.getsize(f)
            print(f'  - {f} ({size} bytes)')
            
            # Show summary
            with open(f, 'r') as tf:
                lines = tf.readlines()
                iterations = sum(1 for line in lines if line.startswith('--- Inner Loop Iteration'))
                print(f'    Contains {iterations} inner loop iterations')
                
                # Show first iteration details
                in_iter = False
                iter_lines = []
                for line in lines[:30]:  # First 30 lines
                    if '--- Inner Loop Iteration' in line:
                        in_iter = True
                        iter_lines = [line]
                    elif in_iter and line.strip():
                        iter_lines.append(line)
                        if len(iter_lines) > 15:  # Show first 15 lines of first iteration
                            break
                
                if iter_lines:
                    print(f'    First iteration sample:')
                    for line in iter_lines[:10]:
                        print(f'      {line.rstrip()}')
    else:
        print('\n[WARNING] No trace files found. Tracking may not be working.')
        print('   Make sure traceIntermediateValues is enabled in options')
        
except Exception as e:
    print(f'\nError: {e}')
    import traceback
    traceback.print_exc()

print('\n=== Test Complete ===')
print('\nTo compare with MATLAB:')
print('1. Run MATLAB with equivalent tracking enabled')
print('2. Use: python compare_intermediate_values.py <matlab_file> <python_file> [tolerance]')
