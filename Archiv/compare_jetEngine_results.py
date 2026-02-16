"""compare_jetEngine_results - Compare MATLAB and Python jetEngine results"""

import numpy as np
import pickle
import scipy.io

# Load MATLAB results
matlab_data = scipy.io.loadmat('jetEngine_matlab_results.mat', squeeze_me=True, struct_as_record=False)
# MATLAB structs are loaded as numpy structured arrays
if 'results' in matlab_data:
    results_struct = matlab_data['results']
    # Convert to dict
    matlab_results = {
        'tComp': float(results_struct['tComp'][0, 0]) if hasattr(results_struct, 'tComp') else None,
        'numSteps': int(results_struct['numSteps'][0, 0]) if hasattr(results_struct, 'numSteps') else None,
        'finalTime': float(results_struct['finalTime'][0, 0]) if hasattr(results_struct, 'finalTime') else None,
        'finalRadius': float(results_struct['finalRadius'][0, 0]) if hasattr(results_struct, 'finalRadius') else None,
        'options_alg': str(results_struct['options_alg'][0, 0]) if hasattr(results_struct, 'options_alg') else None,
    }
else:
    # Fallback: try direct access
    matlab_results = {
        'tComp': float(matlab_data.get('tComp', [0])[0]) if isinstance(matlab_data.get('tComp'), np.ndarray) else matlab_data.get('tComp', 0),
        'numSteps': int(matlab_data.get('numSteps', [0])[0]) if isinstance(matlab_data.get('numSteps'), np.ndarray) else matlab_data.get('numSteps', 0),
        'finalTime': float(matlab_data.get('finalTime', [0])[0]) if isinstance(matlab_data.get('finalTime'), np.ndarray) else matlab_data.get('finalTime', 0),
        'finalRadius': float(matlab_data.get('finalRadius', [0])[0]) if isinstance(matlab_data.get('finalRadius'), np.ndarray) else matlab_data.get('finalRadius', 0),
        'options_alg': str(matlab_data.get('options_alg', [''])[0]) if isinstance(matlab_data.get('options_alg'), np.ndarray) else matlab_data.get('options_alg', ''),
    }

# Load Python results
with open('jetEngine_python_results.pkl', 'rb') as f:
    python_data = pickle.load(f)
python_results = python_data['results']

print("=" * 80)
print("JETENGINE RESULTS COMPARISON")
print("=" * 80)
print()

print("MATLAB Results:")
print(f"  Computation time: {matlab_results['tComp']:.2f} seconds")
print(f"  Number of steps: {matlab_results['numSteps']}")
print(f"  Final time: {matlab_results['finalTime']:.6f}")
print(f"  Final radius: {matlab_results['finalRadius']:.6e}")
print(f"  Final alg: {matlab_results['options_alg']}")
print()

print("Python Results:")
print(f"  Computation time: {python_results['tComp']:.2f} seconds")
print(f"  Number of steps: {python_results['numSteps']}")
if python_results['finalTime'] is not None:
    print(f"  Final time: {python_results['finalTime']:.6f}")
if python_results['finalRadius'] is not None:
    print(f"  Final radius: {python_results['finalRadius']:.6e}")
print(f"  Final alg: {python_results['options_alg']}")
print()

print("Differences:")
print(f"  Computation time: Python {python_results['tComp']/matlab_results['tComp']:.2f}x slower")
print(f"  Number of steps: Python {python_results['numSteps']/matlab_results['numSteps']:.2f}x more steps")
if python_results['finalTime'] is not None:
    print(f"  Final time: Python reached {python_results['finalTime']/matlab_results['finalTime']*100:.1f}% of MATLAB's final time")
    print(f"  Final time difference: {matlab_results['finalTime'] - python_results['finalTime']:.6f} seconds")
if python_results['finalRadius'] is not None:
    print(f"  Final radius: Python {python_results['finalRadius']/matlab_results['finalRadius']:.2e}x larger")
print()

# Check if Python completed successfully
if python_results['finalTime'] is not None and python_results['finalTime'] < matlab_results['finalTime']:
    print("WARNING: Python stopped early before reaching tFinal!")
    print(f"  Expected final time: {matlab_results['finalTime']:.6f}")
    print(f"  Actual final time: {python_results['finalTime']:.6f}")
    print("  This suggests Python may have aborted or encountered an error.")
print()

# Check algorithm match
if matlab_results['options_alg'] == python_results['options_alg']:
    print("✓ Algorithm match: Both use 'lin' (after 'adaptive' removal)")
else:
    print(f"✗ Algorithm mismatch: MATLAB={matlab_results['options_alg']}, Python={python_results['options_alg']}")
