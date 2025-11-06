"""
run_benchmarks - run all benchmarks

Syntax:
    run_benchmarks(benchmarks, data_path, results_path)

Inputs:
    benchmarks - names of the benchmarks
    data_path - path to the benchmark files
    results_path - path to the results directory

Outputs:
    None

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
from typing import List
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.run_instances import run_instances


def run_benchmarks(benchmarks: List[str], data_path: str, results_path: str):
    """
    Run all benchmarks in the list.
    
    Args:
        benchmarks: List of benchmark names to run
        data_path: Path to benchmark data
        results_path: Path to store results
    """
    # Get the base path
    base_path = os.getcwd()
    
    # Prepend the base path to the results path
    results_path = os.path.join(base_path, results_path)
    
    # Create results directory
    os.makedirs(results_path, exist_ok=True)
    
    # Specify base directories
    bench_paths = [
        os.path.join(data_path, 'vnncomp2025_benchmarks', 'benchmarks'),
        os.path.join(data_path, 'vnncomp2024_benchmarks', 'benchmarks'),
        os.path.join(data_path, 'vnncomp2022_benchmarks', 'benchmarks'),
    ]
    
    # List all benchmarks
    bench_dirs = []
    for bench_path in bench_paths:
        if os.path.exists(bench_path):
            for item in os.listdir(bench_path):
                item_path = os.path.join(bench_path, item)
                if os.path.isdir(item_path) and item in benchmarks:
                    bench_dirs.append({
                        'name': item,
                        'path': item_path
                    })
    
    for i, bench_dir in enumerate(bench_dirs):
        bench_name_i = bench_dir['name']
        bench_path_i = bench_dir['path']
        
        print('__________________________________________________________________')
        print('------------------------------------------------------------------')
        print(f'BENCHMARK {bench_name_i} ({i+1}/{len(bench_dirs)})')
        print('------------------------------------------------------------------')
        
        # Create a results directory for this benchmark
        bench_result_path = os.path.join(results_path, f'2025_{bench_name_i}')
        os.makedirs(bench_result_path, exist_ok=True)
        
        # Change directory to the current benchmark
        original_dir = os.getcwd()
        os.chdir(bench_path_i)
        
        try:
            # Run all instances of the benchmark
            num_verif, num_fals, num_unknown = run_instances(bench_name_i, bench_result_path)
            
            # Compute total number of instances
            total_num = num_verif + num_fals + num_unknown
            
            # Print summary
            print('\n' + '='*70)
            print(f'BENCHMARK {bench_name_i} RESULTS')
            print('='*70)
            if total_num > 0:
                print(f'  #Verified:  {num_verif}/{total_num} [{num_verif/total_num*100:.1f}%]')
                print(f'  #Falsified: {num_fals}/{total_num} [{num_fals/total_num*100:.1f}%]')
                print(f'  #Unknown:   {num_unknown}/{total_num} [{num_unknown/total_num*100:.1f}%]')
                print('-'*70)
                solved = num_verif + num_fals
                print(f'  Solved:     {solved}/{total_num} [{solved/total_num*100:.1f}%]')
            else:
                print('  No instances found')
            print('='*70)
            print('__________________________________________________________________\n')
            
        except Exception as e:
            print(f'Error running benchmark {bench_name_i}: {e}')
            import traceback
            traceback.print_exc()
        
        finally:
            # Go back to original directory
            os.chdir(original_dir)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run multiple VNN-COMP benchmarks')
    parser.add_argument('benchmarks', type=str, nargs='+', help='Benchmark names')
    parser.add_argument('--data-path', type=str, default='data', help='Path to benchmark data')
    parser.add_argument('--results-path', type=str, default='results', help='Path to results directory')
    
    args = parser.parse_args()
    
    run_benchmarks(args.benchmarks, args.data_path, args.results_path)

