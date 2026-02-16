"""compare_jetEngine_detailed - Detailed comparison of MATLAB and Python jetEngine results"""

import numpy as np
import scipy.io
import pickle
import os

print("=" * 80)
print("DETAILED JETENGINE COMPARISON")
print("=" * 80)

# Load MATLAB results
matlab_file = 'jetEngine_matlab_results.mat'
if os.path.exists(matlab_file):
    try:
        matlab_data = scipy.io.loadmat(matlab_file, squeeze_me=True)
        results_struct = matlab_data['results']
        
        # Extract MATLAB results
        if hasattr(results_struct, 'dtype') and results_struct.dtype.names:
            matlab_numSteps = int(results_struct['numSteps'][0, 0])
            matlab_finalTime = float(results_struct['finalTime'][0, 0])
            matlab_finalRadius = float(results_struct['finalRadius'][0, 0])
            matlab_alg = str(results_struct['options_alg'][0, 0])
        else:
            matlab_numSteps = int(results_struct['numSteps'])
            matlab_finalTime = float(results_struct['finalTime'])
            matlab_finalRadius = float(results_struct['finalRadius'])
            matlab_alg = str(results_struct['options_alg'])
        
        print("\nMATLAB Results:")
        print(f"  numSteps: {matlab_numSteps}")
        print(f"  finalTime: {matlab_finalTime:.10f}")
        print(f"  finalRadius: {matlab_finalRadius:.10e}")
        print(f"  options_alg: '{matlab_alg}'")
    except Exception as e:
        print(f"ERROR loading MATLAB results: {e}")
        matlab_numSteps = matlab_finalTime = matlab_finalRadius = matlab_alg = None
else:
    print(f"MATLAB results file not found: {matlab_file}")
    print("Please run test_jetEngine_matlab.m first")
    matlab_numSteps = matlab_finalTime = matlab_finalRadius = matlab_alg = None

# Load Python results
python_file = 'jetEngine_python_results.pkl'
if os.path.exists(python_file):
    try:
        with open(python_file, 'rb') as f:
            python_data = pickle.load(f)
            python_results = python_data['results']
        
        print("\nPython Results:")
        print(f"  numSteps: {python_results['numSteps']}")
        if python_results['finalTime'] is not None:
            print(f"  finalTime: {python_results['finalTime']:.10f}")
        if python_results['finalRadius'] is not None:
            print(f"  finalRadius: {python_results['finalRadius']:.10e}")
        print(f"  options_alg: '{python_results['options_alg']}'")
    except Exception as e:
        print(f"ERROR loading Python results: {e}")
        python_results = None
else:
    print(f"Python results file not found: {python_file}")
    print("Please run test_jetEngine_python.py first")
    python_results = None

# Detailed comparison
if matlab_numSteps is not None and python_results is not None:
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    
    # Final time comparison
    if python_results['finalTime'] is not None:
        time_diff = abs(python_results['finalTime'] - matlab_finalTime)
        time_rel_diff = time_diff / matlab_finalTime if matlab_finalTime != 0 else np.inf
        print(f"\nFinal Time:")
        print(f"  MATLAB: {matlab_finalTime:.10f}")
        print(f"  Python: {python_results['finalTime']:.10f}")
        print(f"  Absolute difference: {time_diff:.10f}")
        print(f"  Relative difference: {time_rel_diff*100:.6f}%")
        print(f"  Status: {'✓ MATCH' if time_diff < 1e-6 else '✗ MISMATCH'}")
        
        if time_diff > 0.1:
            print(f"  ⚠️  WARNING: Large time difference! Python stopped early.")
            print(f"     Expected: {matlab_finalTime:.6f}, Got: {python_results['finalTime']:.6f}")
            print(f"     Remaining time: {matlab_finalTime - python_results['finalTime']:.6f}")
    
    # Final radius comparison
    if python_results['finalRadius'] is not None:
        radius_diff = abs(python_results['finalRadius'] - matlab_finalRadius)
        radius_rel_diff = radius_diff / abs(matlab_finalRadius) if matlab_finalRadius != 0 else np.inf
        print(f"\nFinal Radius:")
        print(f"  MATLAB: {matlab_finalRadius:.10e}")
        print(f"  Python: {python_results['finalRadius']:.10e}")
        print(f"  Absolute difference: {radius_diff:.10e}")
        print(f"  Relative difference: {radius_rel_diff*100:.6f}%")
        
        # Check various tolerance levels
        tol_1e6 = radius_rel_diff < 1e-6
        tol_1e4 = radius_rel_diff < 1e-4
        tol_1e2 = radius_rel_diff < 1e-2
        tol_10pct = radius_rel_diff < 0.1
        
        print(f"  Tolerance checks:")
        print(f"    < 1e-6 (machine precision): {'✓' if tol_1e6 else '✗'}")
        print(f"    < 1e-4 (very tight): {'✓' if tol_1e4 else '✗'}")
        print(f"    < 1e-2 (1%): {'✓' if tol_1e2 else '✗'}")
        print(f"    < 10%: {'✓' if tol_10pct else '✗'}")
        
        if tol_1e4:
            print(f"  Status: ✓ EXCELLENT MATCH (< 1e-4 relative)")
        elif tol_1e2:
            print(f"  Status: ✓ GOOD MATCH (< 1% relative)")
        elif tol_10pct:
            print(f"  Status: ⚠️  ACCEPTABLE MATCH (< 10% relative)")
        else:
            print(f"  Status: ✗ POOR MATCH (> 10% relative)")
    
    # Number of steps comparison
    steps_diff = abs(python_results['numSteps'] - matlab_numSteps)
    steps_rel_diff = steps_diff / matlab_numSteps if matlab_numSteps != 0 else np.inf
    print(f"\nNumber of Steps:")
    print(f"  MATLAB: {matlab_numSteps}")
    print(f"  Python: {python_results['numSteps']}")
    print(f"  Difference: {steps_diff}")
    print(f"  Relative difference: {steps_rel_diff*100:.2f}%")
    if steps_diff > 0:
        print(f"  ⚠️  WARNING: Different number of steps!")
        print(f"     This suggests Python stopped early or took different path.")
    
    # Algorithm comparison
    alg_match = python_results['options_alg'] == matlab_alg
    print(f"\nAlgorithm:")
    print(f"  MATLAB: '{matlab_alg}'")
    print(f"  Python: '{python_results['options_alg']}'")
    print(f"  Status: {'✓ MATCH' if alg_match else '✗ MISMATCH'}")
