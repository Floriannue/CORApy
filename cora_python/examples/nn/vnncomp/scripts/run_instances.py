"""
run_instances - run all instances of a benchmark

Syntax:
    num_verif, num_fals, num_unknown = run_instances(bench_name, results_path)

Inputs:
    bench_name - name of the benchmark
    results_path - path to the results directory

Outputs:
    num_verif - number of verified instances
    num_fals - number of falsified instances
    num_unknown - number of unknown instances

References:
    [1] VNN-COMP'24

Authors:       Lukas Koller
Written:       11-August-2025
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import sys
import os
import csv
import time
from typing import Tuple, List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from run_instance import run_instance


def run_instances(bench_name: str, results_path: str) -> Tuple[int, int, int]:
    """
    Run all instances found in the current directory.
    
    Args:
        bench_name: Name of the benchmark
        results_path: Path to store results
        
    Returns:
        Tuple of (num_verified, num_falsified, num_unknown)
    """
    verbose = True
    
    # Obtain all instances from CSV
    filename = 'instances.csv'
    instances = []
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:  # model, vnnlib, timeout
                instances.append({
                    'model': row[0],
                    'vnnlib': row[1],
                    'timeout': float(row[2])
                })
    
    N = len(instances)
    
    # Init results
    bench_names = []
    models = []
    vnnlibs = []
    prep_times = []
    results = []
    verif_times = []
    
    # Count numbers
    num_verif = 0
    num_fals = 0
    num_unknown = 0
    num_verif_patches = 0
    
    # Specify the instance IDs
    instance_ids = range(N)  # All instances
    
    for i in instance_ids:
        print('__________________________________________________________________')
        print(f'INSTANCE ({i+1}/{N})')
        print('------------------------------------------------------------------')
        
        # Extract current instance
        instance = instances[i]
        model_path = instance['model']
        vnnlib_path = instance['vnnlib']
        timeout = instance['timeout']
        
        # Create instance filename
        import re
        model_match = re.search(r'([^/\\]+)(?=\.onnx$)', model_path)
        model_name = model_match.group(1) if model_match else 'unknown_model'
        
        # Special handling for safenlp benchmark
        if bench_name == 'safenlp':
            if 'medical' in model_path:
                model_name = f'medical_{model_name}'
            elif 'ruarobot' in model_path:
                model_name = f'ruarobot_{model_name}'
        
        vnnlib_match = re.search(r'([^/\\]+)(?=\.vnnlib$)', vnnlib_path)
        vnnlib_name = vnnlib_match.group(1) if vnnlib_match else 'unknown_vnnlib'
        
        instance_filename = os.path.join(results_path, f'{model_name}_{vnnlib_name}.counterexample')
        
        # Note: Python version integrates prepare_instance into run_instance
        # No separate prepare step needed
        
        total_time_start = time.time()
        
        # Run the current instance
        try:
            res_str, res = run_instance(bench_name, model_path, vnnlib_path,
                                       instance_filename, timeout, verbose)
        except Exception as e:
            print(f'Error running instance: {e}')
            res_str = 'unknown'
            res = {'time': 0, 'numVerified': 0}
        
        instance_time = time.time() - total_time_start
        
        # Delete counterexample file if not sat
        if res_str in ['unsat', 'unknown'] and os.path.exists(instance_filename):
            os.remove(instance_filename)
        
        # Store outputs
        bench_names.append(bench_name)
        models.append(f'vnncomp2025_benchmarks/benchmarks/{bench_name}/{model_path}')
        vnnlibs.append(f'vnncomp2025_benchmarks/benchmarks/{bench_name}/{vnnlib_path}')
        prep_times.append(0)  # No separate prep phase in Python
        results.append(res_str)
        verif_times.append(instance_time)
        
        # Increment counters
        if res_str == 'unsat':
            num_verif += 1
        elif res_str == 'sat':
            num_fals += 1
        else:
            num_unknown += 1
        
        if 'numVerified' in res:
            num_verif_patches += res['numVerified']
        
        print('------------------------------------------------------------------')
        print('__________________________________________________________________')
    
    # Print stats
    print('\n' + '='*70)
    print('STATS')
    print('='*70)
    if instance_ids:
        avg_patches = num_verif_patches / len(list(instance_ids))
        avg_time = sum(verif_times) / len(list(instance_ids))
        print(f'  avg. #Verified Branches: {avg_patches:.2f}')
        print(f'  avg. Time: {avg_time:.4f}s')
    print('='*70 + '\n')
    
    # Write results table to CSV
    results_file = os.path.join(results_path, 'results.csv')
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(bench_names)):
            writer.writerow([
                bench_names[i],
                models[i],
                vnnlibs[i],
                prep_times[i],
                results[i],
                verif_times[i]
            ])
    
    return num_verif, num_fals, num_unknown


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run all instances of a benchmark')
    parser.add_argument('bench_name', type=str, help='Benchmark name')
    parser.add_argument('results_path', type=str, help='Path to results directory')
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_path, exist_ok=True)
    
    num_v, num_f, num_u = run_instances(args.bench_name, args.results_path)
    
    print(f'\nFinal Results:')
    print(f'  Verified: {num_v}')
    print(f'  Falsified: {num_f}')
    print(f'  Unknown: {num_u}')

