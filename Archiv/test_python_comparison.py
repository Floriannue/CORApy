"""
Python test for comparing error_adm_horizon growth with MATLAB
Uses tensor order 2 to match MATLAB test
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from cora_python.contDynamics.nonlinearSys.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope.zonotope import Zonotope
import numpy as np
import time

def vanderPolEq(x, u):
    """Van der Pol oscillator dynamics"""
    mu = 1.0
    dx = np.zeros((2, 1))
    dx[0, 0] = x[1, 0]
    dx[1, 0] = mu * (1 - x[0, 0]**2) * x[1, 0] - x[0, 0] + u[0, 0]
    return dx

def main():
    print("=== Testing Intermediate Value Tracking with vanDerPol (Python) ===")
    print("Running reach_adaptive with tracking enabled...")
    
    # Create system
    sys = NonlinearSys(vanderPolEq, 2, 1, name='vanderPolEq')
    
    # Set up parameters
    params = {
        'tStart': 0.0,
        'tFinal': 5.0,  # Longer time to capture error_adm_horizon growth
        'R0': Zonotope(np.array([[1.0], [1.0]]), 0.1 * np.eye(2)),
        'U': Zonotope(np.zeros((1, 1)), np.array([]).reshape(0, 1))
    }
    
    # Set up options
    options = {
        'alg': 'lin-adaptive',
        'tensorOrder': 2,  # Use tensor order 2 to match MATLAB
        'traceIntermediateValues': True,
        'progress': True,
        'progressInterval': 10,
        'verbose': 0,
        'isHessianConst': False,
        'hessianCheck': False,
        'thirdOrderTensorempty': True  # No third order tensor needed
    }
    
    print(f"Time horizon: {params['tStart']:.1f} to {params['tFinal']:.1f}")
    print(f"Tensor order: {options['tensorOrder']}")
    print("This may take a while...\n")
    
    start_time = time.time()
    
    try:
        # Run reachability analysis
        R = sys.reach(params, options)
        elapsed_time = time.time() - start_time
        
        print(f"[OK] reach_adaptive completed in {elapsed_time:.2f} seconds")
        
        # Check for trace files
        import glob
        trace_files = glob.glob('intermediate_values_step*_inner_loop.txt')
        trace_files.sort(key=lambda x: int(x.split('step')[1].split('_')[0]))
        
        if trace_files:
            print(f"\n[OK] Found {len(trace_files)} trace file(s):")
            for i, fname in enumerate(trace_files[:10]):
                size = os.path.getsize(fname)
                print(f"  - {fname} ({size} bytes)")
            if len(trace_files) > 10:
                print(f"  ... and {len(trace_files) - 10} more")
            
            # Analyze error_adm_horizon growth
            print("\n=== Analyzing error_adm_horizon Growth ===")
            error_adm_horizon_values = []
            
            for fname in trace_files:
                try:
                    step_num = int(fname.split('step')[1].split('_')[0])
                    with open(fname, 'r') as f:
                        content = f.read()
                    
                    # Extract initial error_adm_horizon
                    import re
                    match = re.search(r'Initial error_adm_horizon:\s*\[\[([\d.e+-]+)\]', content)
                    if match:
                        val = float(match.group(1))
                        error_adm_horizon_values.append((step_num, val))
                except Exception as e:
                    print(f"Warning: Could not parse {fname}: {e}")
            
            if error_adm_horizon_values:
                error_adm_horizon_values.sort(key=lambda x: x[0])
                print(f"Found error_adm_horizon values for {len(error_adm_horizon_values)} steps")
                print("Step | error_adm_horizon_max")
                print("------------------------------")
                for step_num, val in error_adm_horizon_values[:20]:
                    print(f"{step_num:4d} | {val:.6e}")
                if len(error_adm_horizon_values) > 20:
                    print(f"... and {len(error_adm_horizon_values) - 20} more")
                
                # Growth analysis
                if len(error_adm_horizon_values) > 1:
                    first_val = error_adm_horizon_values[0][1]
                    last_val = error_adm_horizon_values[-1][1]
                    growth_factor = last_val / first_val if first_val > 0 else float('inf')
                    print("\nGrowth analysis:")
                    print(f"  First step: {first_val:.6e}")
                    print(f"  Last step:  {last_val:.6e}")
                    print(f"  Growth factor: {growth_factor:.2f}x")
                    if growth_factor > 10:
                        print("  [WARNING] Significant growth detected!")
        else:
            print("\n[WARNING] No trace files found. Tracking may not be working.")
        
        print("\n=== Test Complete ===")
        print("\nTo compare with MATLAB:")
        print("1. Run MATLAB with equivalent tracking enabled")
        print("2. Use: python compare_intermediate_values.py <matlab_file> <python_file> [tolerance]")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
