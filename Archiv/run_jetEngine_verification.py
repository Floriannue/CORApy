"""run_jetEngine_verification - Run MATLAB and Python tests and verify they match

This script:
1. Runs MATLAB to generate expected values
2. Runs Python test
3. Compares results
"""

import subprocess
import sys
import os
import scipy.io
import pickle

def run_matlab_generate_expected():
    """Run MATLAB to generate expected values"""
    print("=" * 80)
    print("STEP 1: Running MATLAB to generate expected values")
    print("=" * 80)
    
    result = subprocess.run(
        ['matlab', '-batch', 'generate_jetEngine_expected_values'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("ERROR: MATLAB failed to generate expected values")
        print(result.stderr)
        return False
    
    print(result.stdout)
    
    # Check if expected values file was created
    if not os.path.exists('jetEngine_expected_values.mat'):
        print("ERROR: Expected values file not created")
        return False
    
    print("\n✓ MATLAB expected values generated successfully")
    return True

def run_python_test():
    """Run Python test"""
    print("\n" + "=" * 80)
    print("STEP 2: Running Python test")
    print("=" * 80)
    
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', 
         'cora_python/tests/contDynamics/nonlinearSys/test_nonlinearSys_reach_adaptive_01_jetEngine.py::test_nonlinearSys_reach_adaptive_01_jetEngine',
         '-v', '--tb=short'],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode == 0

def compare_results():
    """Compare MATLAB and Python results"""
    print("\n" + "=" * 80)
    print("STEP 3: Comparing MATLAB and Python results")
    print("=" * 80)
    
    # Load MATLAB expected values
    try:
        matlab_data = scipy.io.loadmat('jetEngine_expected_values.mat', squeeze_me=True)
        expected_values = matlab_data['expected_values']
        
        if hasattr(expected_values, 'dtype') and expected_values.dtype.names:
            matlab_numSteps = int(expected_values['numSteps'][0, 0])
            matlab_finalTime = float(expected_values['finalTime'][0, 0])
            matlab_finalRadius = float(expected_values['finalRadius'][0, 0])
            matlab_alg = str(expected_values['options_alg'][0, 0])
        else:
            matlab_numSteps = int(expected_values['numSteps'])
            matlab_finalTime = float(expected_values['finalTime'])
            matlab_finalRadius = float(expected_values['finalRadius'])
            matlab_alg = str(expected_values['options_alg'])
    except Exception as e:
        print(f"ERROR: Could not load MATLAB expected values: {e}")
        return False
    
    # Load Python results if available
    python_results = None
    if os.path.exists('jetEngine_python_results.pkl'):
        try:
            with open('jetEngine_python_results.pkl', 'rb') as f:
                python_data = pickle.load(f)
                python_results = python_data['results']
        except Exception as e:
            print(f"WARNING: Could not load Python results: {e}")
    
    print("\nMATLAB Expected Values:")
    print(f"  numSteps: {matlab_numSteps}")
    print(f"  finalTime: {matlab_finalTime:.10f}")
    print(f"  finalRadius: {matlab_finalRadius:.10e}")
    print(f"  options_alg: '{matlab_alg}'")
    
    if python_results:
        print("\nPython Actual Values:")
        print(f"  numSteps: {python_results['numSteps']}")
        if python_results['finalTime'] is not None:
            print(f"  finalTime: {python_results['finalTime']:.10f}")
        if python_results['finalRadius'] is not None:
            print(f"  finalRadius: {python_results['finalRadius']:.10e}")
        print(f"  options_alg: '{python_results['options_alg']}'")
        
        print("\nComparison:")
        # Compare values
        all_match = True
        
        if python_results['finalTime'] is not None:
            time_diff = abs(python_results['finalTime'] - matlab_finalTime)
            time_match = time_diff < 0.1
            print(f"  Final time: {'✓' if time_match else '✗'} (diff: {time_diff:.6f})")
            if not time_match:
                all_match = False
        
        if python_results['finalRadius'] is not None:
            radius_diff = abs(python_results['finalRadius'] - matlab_finalRadius)
            radius_rel_diff = radius_diff / max(abs(matlab_finalRadius), 1e-10)
            radius_match = radius_rel_diff < 0.1  # 10% tolerance
            print(f"  Final radius: {'✓' if radius_match else '✗'} (rel diff: {radius_rel_diff*100:.2f}%)")
            if not radius_match:
                all_match = False
        
        alg_match = python_results['options_alg'] == matlab_alg
        print(f"  Algorithm: {'✓' if alg_match else '✗'}")
        if not alg_match:
            all_match = False
        
        return all_match
    else:
        print("\nPython results not available for comparison")
        return False

def main():
    """Main workflow"""
    print("JetEngine Verification Workflow")
    print("This script ensures MATLAB and Python run the same way and produce matching results\n")
    
    # Step 1: Generate MATLAB expected values
    if not run_matlab_generate_expected():
        print("\n✗ Workflow failed at Step 1")
        return 1
    
    # Step 2: Run Python test
    if not run_python_test():
        print("\n✗ Workflow failed at Step 2")
        return 1
    
    # Step 3: Compare results
    if not compare_results():
        print("\n✗ Workflow failed at Step 3 - Results do not match")
        return 1
    
    print("\n" + "=" * 80)
    print("✓ All steps completed successfully - MATLAB and Python results match!")
    print("=" * 80)
    return 0

if __name__ == '__main__':
    sys.exit(main())
