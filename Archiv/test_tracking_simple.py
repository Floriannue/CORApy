"""
Simple test to verify intermediate value tracking works
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.contDynamics.nonlinearSys.nonlinearSys import NonlinearSys
from cora_python.contDynamics.nonlinearSys.reach_adaptive import reach_adaptive

print('=== Testing Intermediate Value Tracking ===\n')

# Create a simple 2D system
def f(x, u):
    return np.array([[-x[0, 0] + x[1, 0]], [-x[1, 0] + u[0, 0]]])

sys = NonlinearSys('simple', f, 2, 1)

# Setup parameters
params = {
    'tStart': 0,
    'tFinal': 1.0,  # Short time to test quickly
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
    'traceIntermediateValues': True,  # Enable tracking
    'progress': False  # Disable progress to reduce noise
}

print('Running reach_adaptive with tracking enabled...')
print('This will create intermediate_values_step{N}_inner_loop.txt files\n')

try:
    timeInt, timePoint, res, tVec, options = reach_adaptive(sys, params, options)
    print('✅ reach_adaptive completed successfully')
    
    # Check if trace files were created
    import glob
    trace_files = glob.glob('intermediate_values_step*_inner_loop.txt')
    if trace_files:
        print(f'\n✅ Found {len(trace_files)} trace file(s):')
        for f in sorted(trace_files):
            size = os.path.getsize(f)
            print(f'  - {f} ({size} bytes)')
            
            # Show first few lines
            with open(f, 'r') as tf:
                lines = tf.readlines()[:10]
                print(f'    First {len(lines)} lines:')
                for line in lines:
                    print(f'      {line.rstrip()}')
    else:
        print('\n⚠️  No trace files found. Tracking may not be working.')
        
except Exception as e:
    print(f'\nError: {e}')
    import traceback
    traceback.print_exc()

print('\n=== Test Complete ===')
