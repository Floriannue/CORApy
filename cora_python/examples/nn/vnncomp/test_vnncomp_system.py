"""
Test script for VNN-COMP system

This tests a single instance from the test benchmark.
"""

import sys
import os

# Add cora_python to path
sys.path.insert(0, os.path.dirname(__file__))

from cora_python.examples.nn.vnncomp.run_instance import run_instance

# Test paths
bench_name = 'test'
model_path = 'cora_python/examples/nn/vnncomp/data/vnncomp2025_benchmarks/benchmarks/test/onnx/test_nano.onnx'
vnnlib_path = 'cora_python/examples/nn/vnncomp/data/vnncomp2025_benchmarks/benchmarks/test/vnnlib/test_nano.vnnlib'
results_path = 'test_vnncomp_results/test_nano.counterexample'
timeout = 60.0
verbose = True

print('='*70)
print('Testing VNN-COMP Python System')
print('='*70)
print(f'Benchmark: {bench_name}')
print(f'Model: {model_path}')
print(f'Spec: {vnnlib_path}')
print(f'Timeout: {timeout}s')
print('='*70)
print()

try:
    result_str, result = run_instance(bench_name, model_path, vnnlib_path, 
                                     results_path, timeout, verbose)
    
    print()
    print('='*70)
    print('TEST COMPLETED SUCCESSFULLY')
    print('='*70)
    print(f'Result: {result_str}')
    print(f'Time: {result.get("time", "N/A")}s')
    print('='*70)
    
except Exception as e:
    print()
    print('='*70)
    print('TEST FAILED')
    print('='*70)
    print(f'Error: {e}')
    print('='*70)
    import traceback
    traceback.print_exc()
    sys.exit(1)

