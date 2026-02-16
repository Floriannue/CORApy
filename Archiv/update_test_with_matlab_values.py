"""update_test_with_matlab_values - Update Python test with MATLAB-generated expected values"""

import scipy.io
import numpy as np

# Load MATLAB expected values
try:
    matlab_data = scipy.io.loadmat('jetEngine_expected_values.mat', squeeze_me=True)
    expected_values = matlab_data['expected_values']
    
    # Extract values (handle MATLAB struct format)
    if hasattr(expected_values, 'dtype') and expected_values.dtype.names:
        # Structured array
        numSteps = int(expected_values['numSteps'][0, 0])
        finalTime = float(expected_values['finalTime'][0, 0])
        finalRadius = float(expected_values['finalRadius'][0, 0])
        options_alg = str(expected_values['options_alg'][0, 0])
    else:
        # Direct access
        numSteps = int(expected_values['numSteps'])
        finalTime = float(expected_values['finalTime'])
        finalRadius = float(expected_values['finalRadius'])
        options_alg = str(expected_values['options_alg'])
    
    print("Loaded MATLAB expected values:")
    print(f"  numSteps: {numSteps}")
    print(f"  finalTime: {finalTime:.10f}")
    print(f"  finalRadius: {finalRadius:.10e}")
    print(f"  options_alg: '{options_alg}'")
    
    # Generate Python test code snippet
    print("\n=== Python Test Code (to insert) ===")
    print(f"    expected_numSteps = {numSteps}")
    print(f"    expected_finalTime = {finalTime:.10f}")
    print(f"    expected_finalRadius = {finalRadius:.10e}")
    print(f"    expected_alg = '{options_alg}'")
    
except Exception as e:
    print(f"Error loading MATLAB values: {e}")
    print("Please run generate_jetEngine_expected_values.m first")
