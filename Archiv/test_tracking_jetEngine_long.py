"""
test_tracking_jetEngine_long.py
Test script for intermediate value tracking with jetEngine model (longer time horizon)
This is designed to capture the error_adm_horizon growth issue.
"""

import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.models.Cora.contDynamics.nonlinearSys.models.jetEngine import jetEngine

def main():
    print("=== Testing Intermediate Value Tracking with jetEngine (Long Time Horizon) ===\n")
    
    # Create jetEngine nonlinear system
    sys = NonlinearSys(name='jetEngine', fun=jetEngine, states=2, inputs=1)
    
    # Set up parameters with longer time horizon to capture error growth
    params = {}
    params['tFinal'] = 5.0  # Longer time to capture error_adm_horizon growth
    params['R0'] = Zonotope(np.array([[1], [1]]), 0.1 * np.eye(2))
    params['U'] = Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0))
    
    # Set up options with tracking enabled
    options = {}
    options['alg'] = 'lin-adaptive'
    options['traceIntermediateValues'] = True  # Enable tracking
    options['progress'] = True
    options['progressInterval'] = 10  # Less frequent updates for longer run
    
    # Clean up any existing trace files
    import glob
    for f in glob.glob('intermediate_values_step*_inner_loop.txt'):
        try:
            os.remove(f)
        except:
            pass
    
    print("Running reach with tracking enabled...")
    print(f"Time horizon: 0 to {params['tFinal']}")
    print("This may take a while...\n")
    
    try:
        t_start = time.time()
        result = sys.reach(params, options)
        t_elapsed = time.time() - t_start
        
        # Handle different return types
        if isinstance(result, tuple):
            R, _, opt = result
        else:
            R = result
            opt = options
        
        print(f"\n[OK] reach completed in {t_elapsed:.2f} seconds")
        
        # Check for trace files
        trace_files = sorted(glob.glob('intermediate_values_step*_inner_loop.txt'))
        if trace_files:
            print(f"\n[OK] Found {len(trace_files)} trace file(s):")
            for i, fname in enumerate(trace_files[:10]):  # Show first 10
                size = os.path.getsize(fname)
                print(f"  - {fname} ({size} bytes)")
            if len(trace_files) > 10:
                print(f"  ... and {len(trace_files) - 10} more")
            
            # Analyze error_adm_horizon growth
            print("\n=== Analyzing error_adm_horizon Growth ===")
            error_adm_horizon_values = []
            for fname in trace_files:
                try:
                    with open(fname, 'r') as f:
                        content = f.read()
                        # Extract initial error_adm_horizon
                        if 'Initial error_adm_horizon:' in content:
                            line = [l for l in content.split('\n') if 'Initial error_adm_horizon:' in l][0]
                            # Parse the value (format: [[value1]\n [value2]])
                            import re
                            match = re.search(r'\[\[([\d.e+-]+)\]', line)
                            if match:
                                val = float(match.group(1))
                                step_num = int(re.search(r'step(\d+)', fname).group(1))
                                error_adm_horizon_values.append((step_num, val))
                except Exception as e:
                    pass
            
            if error_adm_horizon_values:
                error_adm_horizon_values.sort(key=lambda x: x[0])
                print(f"\nFound error_adm_horizon values for {len(error_adm_horizon_values)} steps:")
                print("Step | error_adm_horizon_max")
                print("-" * 30)
                for step, val in error_adm_horizon_values[:20]:  # Show first 20
                    print(f"{step:4d} | {val:.6e}")
                if len(error_adm_horizon_values) > 20:
                    print(f"... and {len(error_adm_horizon_values) - 20} more")
                
                # Check for growth
                if len(error_adm_horizon_values) > 1:
                    first_val = error_adm_horizon_values[0][1]
                    last_val = error_adm_horizon_values[-1][1]
                    growth_factor = last_val / first_val if first_val > 0 else float('inf')
                    print(f"\nGrowth analysis:")
                    print(f"  First step: {first_val:.6e}")
                    print(f"  Last step:  {last_val:.6e}")
                    print(f"  Growth factor: {growth_factor:.2f}x")
                    if growth_factor > 10:
                        print(f"  [WARNING] Significant growth detected!")
                        print(f"  Steps with largest growth:")
                        # Find steps with largest increases
                        increases = []
                        for i in range(1, len(error_adm_horizon_values)):
                            step1, val1 = error_adm_horizon_values[i-1]
                            step2, val2 = error_adm_horizon_values[i]
                            if val1 > 0:
                                increase = val2 / val1
                                increases.append((step2, increase, val1, val2))
                        increases.sort(key=lambda x: x[1], reverse=True)
                        for step, inc, v1, v2 in increases[:5]:
                            print(f"    Step {step}: {v1:.6e} -> {v2:.6e} ({inc:.2f}x)")
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
