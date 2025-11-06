"""
Test script for VNN-COMP run_instances

This tests all instances from the test benchmark.
"""

import sys
import os

# Add cora_python to path
sys.path.insert(0, os.path.dirname(__file__))

from cora_python.examples.nn.vnncomp.scripts.run_instances import run_instances

# Change to test benchmark directory
original_dir = os.getcwd()
test_dir = 'cora_python/examples/nn/vnncomp/data/vnncomp2025_benchmarks/benchmarks/test'
results_dir = 'test_vnncomp_results'

print('='*70)
print('Testing VNN-COMP run_instances')
print('='*70)
print(f'Test directory: {test_dir}')
print(f'Results directory: {results_dir}')
print('='*70)
print()

try:
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Change to test directory
    os.chdir(test_dir)
    
    # Run all instances
    num_verif, num_fals, num_unknown = run_instances('test', os.path.join(original_dir, results_dir))
    
    # Change back
    os.chdir(original_dir)
    
    print()
    print('='*70)
    print('ALL TESTS COMPLETED SUCCESSFULLY')
    print('='*70)
    print(f'Verified: {num_verif}')
    print(f'Falsified: {num_fals}')
    print(f'Unknown: {num_unknown}')
    print(f'Total: {num_verif + num_fals + num_unknown}')
    print('='*70)
    
except Exception as e:
    os.chdir(original_dir)
    print()
    print('='*70)
    print('TEST FAILED')
    print('='*70)
    print(f'Error: {e}')
    print('='*70)
    import traceback
    traceback.print_exc()
    sys.exit(1)

