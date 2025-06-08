#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

from cora_python.contDynamics.linearSys import LinearSys
import numpy as np

def debug_failing_test():
    """Debug the failing test case"""
    print("Debugging failing test case...")
    
    # Recreate the test setup
    A = np.array([
        [-0.3780, 0.2839, 0.5403, -0.2962],
        [0.1362, 0.2742, 0.5195, 0.8266],
        [0.0502, -0.1051, -0.6572, 0.3874],
        [1.0227, -0.4877, 0.8342, -0.2372]
    ])
    B = 0.25 * np.array([[-2, 0, 3],
                         [2, 1, 0],
                         [0, 0, 1],
                         [0, -2, 1]])
    sys_A = LinearSys(A, B)
    params = {'x0': 10 * np.ones(4), 'tFinal': 1.0}
    
    print(f'System: inputs={sys_A.nr_of_inputs}, disturbances={sys_A.nr_of_disturbances}, noises={sys_A.nr_of_noises}')
    print(f'D matrix: {sys_A.D}')
    print(f'D is None: {sys_A.D is None}')
    
    # Test the failing case
    np.random.seed(42)
    v_sys = np.random.randn(1, 11)  # 1 noise, 11 time steps
    print(f'v_sys shape: {v_sys.shape}')
    
    params['v'] = v_sys
    try:
        t, x, ind, y = sys_A.simulate(params)
        print('Success!')
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

# Test system from the failing test
A = np.array([[-0.3780, 0.2839, 0.5403, -0.2962],
              [0.1362, 0.2742, 0.5195, 0.8266],
              [0.0502, -0.1051, -0.6572, 0.3874],
              [1.0227, -0.4877, 0.8342, -0.2372]])
B = 0.25 * np.array([[-2, 0, 3],
                     [2, 1, 0],
                     [0, 0, 1],
                     [0, -2, 1]])

sys = LinearSys(A, B)
print('A shape:', sys.A.shape)
print('B shape:', sys.B.shape)
print('C shape:', sys.C.shape)
print('D shape:', sys.D.shape)
print('E shape:', sys.E.shape)
print('F shape:', sys.F.shape)
print('nr_of_inputs:', sys.nr_of_inputs)
print('nr_of_outputs:', sys.nr_of_outputs)
print('nr_of_disturbances:', sys.nr_of_disturbances)

# Test uTrans creation
uTrans = np.zeros((sys.nr_of_inputs, 1))
print('uTrans shape:', uTrans.shape)

# Test matrix multiplication
try:
    result = sys.D @ uTrans
    print('D @ uTrans shape:', result.shape)
except Exception as e:
    print('Error in D @ uTrans:', e)

# Test the canonicalForm issue
from cora_python.contSet.zonotope import Zonotope
from cora_python.contDynamics.linearSys.canonicalForm import _center
V = Zonotope.origin(sys.nr_of_outputs)
vVec = np.zeros((sys.nr_of_outputs, 1))
centerV = _center(V)  # Use the _center function
print('V center shape:', centerV.shape)
print('vVec shape:', vVec.shape)
print('centerV + vVec shape:', (centerV + vVec).shape)
print('F shape:', sys.F.shape)
print('F.shape[1]:', sys.F.shape[1])
print('Condition check:', sys.F.shape[1] == 1 and (centerV + vVec).shape == sys.F.shape)

# Test the actual matrix multiplication that's failing
try:
    result = sys.F @ (centerV + vVec)
    print('F @ (centerV + vVec) shape:', result.shape)
except Exception as e:
    print('Error in F @ (centerV + vVec):', e)
    print('Trying element-wise multiplication:')
    result = sys.F * (centerV + vVec)
    print('F * (centerV + vVec) shape:', result.shape)

if __name__ == "__main__":
    debug_failing_test() 