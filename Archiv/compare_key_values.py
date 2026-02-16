"""compare_key_values - Compare Python and MATLAB key values"""

import numpy as np
import pickle
import scipy.io
import os

print("=" * 80)
print("COMPARING KEY VALUES: PYTHON vs MATLAB")
print("=" * 80)

# Load Python values
python_file = 'jetEngine_python_key_values.pkl'
if os.path.exists(python_file):
    with open(python_file, 'rb') as f:
        python_data = pickle.load(f)
    print("\n[OK] Loaded Python key values")
else:
    print("\n[ERROR] Python key values file not found")
    python_data = None

# Load MATLAB values (if saved)
matlab_file = 'jetEngine_matlab_key_values.mat'
if os.path.exists(matlab_file):
    try:
        matlab_data = scipy.io.loadmat(matlab_file, squeeze_me=True)
        print("[OK] Loaded MATLAB key values")
    except:
        print("[ERROR] Could not load MATLAB key values")
        matlab_data = None
else:
    print("[WARNING] MATLAB key values file not found - run track_jetEngine_key_values_matlab.m first")
    matlab_data = None

# Compare first 20 steps
if python_data:
    print("\n" + "=" * 80)
    print("PYTHON VALUES (first 20 steps)")
    print("=" * 80)
    
    if 'finitehorizon' in python_data and len(python_data['finitehorizon']) > 0:
        print("\nPython finitehorizon:")
        for i in range(min(20, len(python_data['finitehorizon']))):
            print(f"  Step {i+1}: {python_data['finitehorizon'][i]:.6e}")
    
    if 'varphi' in python_data and len(python_data['varphi']) > 0:
        print("\nPython varphi:")
        for i in range(min(20, len(python_data['varphi']))):
            if python_data['varphi'][i] > 0:
                print(f"  Step {i+1}: {python_data['varphi'][i]:.6f}")
    
    if 'stepsize' in python_data and len(python_data['stepsize']) > 0:
        print("\nPython stepsize:")
        for i in range(min(20, len(python_data['stepsize']))):
            print(f"  Step {i+1}: {python_data['stepsize'][i]:.6e}")
    
    # Analyze finitehorizon growth
    if 'debug_finitehorizon' in python_data and len(python_data['debug_finitehorizon']) > 0:
        print("\n" + "=" * 80)
        print("FINITEHORIZON GROWTH ANALYSIS")
        print("=" * 80)
        print("\nFirst 20 finitehorizon computations:")
        for entry in python_data['debug_finitehorizon'][:20]:
            print(f"\nStep {entry['step']}:")
            print(f"  prev_finitehorizon: {entry['prev_finitehorizon']:.6e}")
            print(f"  prev_varphi: {entry['prev_varphi']:.6f}")
            print(f"  zetaphi: {entry['zetaphi']:.6f}")
            print(f"  computed_finitehorizon: {entry['computed_finitehorizon']:.6e}")
            print(f"  remTime: {entry['remTime']:.6f}")
            print(f"  capped_finitehorizon: {entry['capped_finitehorizon']:.6e}")
            
            # Check if unbounded
            if entry['computed_finitehorizon'] > entry['remTime']:
                print(f"  ⚠️  UNBOUNDED: computed ({entry['computed_finitehorizon']:.6e}) > remTime ({entry['remTime']:.6f})")
                print(f"     Ratio: {entry['computed_finitehorizon'] / entry['remTime']:.2e}")
            
            # Check growth factor
            if entry['step'] > 1:
                prev_entry = python_data['debug_finitehorizon'][entry['step'] - 2]
                growth = entry['computed_finitehorizon'] / prev_entry['computed_finitehorizon'] if prev_entry['computed_finitehorizon'] > 0 else 0
                print(f"  Growth factor: {growth:.6f}")
