"""
Compare results from two different VNN-COMP runs

This script compares results.csv files from two different evaluation runs
and checks for soundness issues.

Authors:       Lukas Koller
Written:       11-August-2025
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import os
import sys
import csv
from typing import Dict, List, Tuple


def compare_results(results_path1: str, results_path2: str):
    """
    Compare results from two different runs.
    
    Args:
        results_path1: Path to first results directory
        results_path2: Path to second results directory
    """
    # Find all benchmark result directories in first path
    if not os.path.exists(results_path1):
        print(f'Error: Results path 1 not found: {results_path1}')
        return
    
    bench_dirs = [d for d in os.listdir(results_path1) 
                  if os.path.isdir(os.path.join(results_path1, d))]
    
    for bench_name in bench_dirs:
        # Skip hidden directories
        if bench_name.startswith('.'):
            continue
        
        # Read first results file
        try:
            results1 = read_results_file(os.path.join(results_path1, bench_name, 'results.csv'))
            # Remove test_nano if present
            results1 = [r for r in results1 if not r['onnxPath'].endswith('test_nano.onnx')]
        except Exception as e:
            print(f'--- Cannot read results for "{results_path1}/{bench_name}"!')
            print(f'    Error: {e}')
            continue
        
        # Read second results file
        try:
            results2 = read_results_file(os.path.join(results_path2, bench_name, 'results.csv'))
            # Remove test_nano if present
            results2 = [r for r in results2 if not r['onnxPath'].endswith('test_nano.onnx')]
        except Exception as e:
            print(f'--- Cannot read results for "{results_path2}/{bench_name}"!')
            print(f'    Error: {e}')
            continue
        
        print('__________________________________________________________________')
        print('------------------------------------------------------------------')
        print(f'BENCHMARK {bench_name}')
        print('------------------------------------------------------------------')
        
        # Ensure same number of entries
        if len(results1) != len(results2):
            print(f'Warning: Different number of results: {len(results1)} vs {len(results2)}')
            min_len = min(len(results1), len(results2))
            results1 = results1[:min_len]
            results2 = results2[:min_len]
        
        # Initialize counters
        num_verif = [0, 0]
        num_fals = [0, 0]
        num_unknown = [0, 0]
        soundness_issues = []
        
        for j in range(len(results1)):
            result1j = results1[j]
            result2j = results2[j]
            
            # Ensure comparing same instance
            if result1j['name'] != result2j['name']:
                print(f'Warning: Mismatch at row {j}: {result1j["name"]} vs {result2j["name"]}')
            
            # Check for soundness issues
            if (result1j['result'] == 'unsat' and result2j['result'] == 'sat') or \
               (result1j['result'] == 'sat' and result2j['result'] == 'unsat'):
                soundness_issues.append((j, result1j, result2j))
                print(f'SOUNDNESS ISSUE at instance {j}:')
                print(f'  Results 1: {result1j["result"]}')
                print(f'  Results 2: {result2j["result"]}')
                print(f'  Model: {result1j["onnxPath"]}')
                print(f'  Spec: {result1j["vnnlibPath"]}')
            
            # Count results for first run
            if result1j['result'] == 'unsat':
                num_verif[0] += 1
            elif result1j['result'] == 'sat':
                num_fals[0] += 1
            else:  # unknown or timeout
                num_unknown[0] += 1
            
            # Count results for second run
            if result2j['result'] == 'unsat':
                num_verif[1] += 1
            elif result2j['result'] == 'sat':
                num_fals[1] += 1
            else:  # unknown or timeout
                num_unknown[1] += 1
            
            # Report unknown instances where second tool succeeded
            if result1j['result'] == 'unknown' and \
               result2j['result'] not in ['unknown', 'timeout']:
                print(f'unknown instance (results 2: {result2j["result"]}): {j}')
        
        # Compute totals
        total_num = [num_verif[0] + num_fals[0] + num_unknown[0],
                     num_verif[1] + num_fals[1] + num_unknown[1]]
        
        # Print summary for first results
        print()
        print('='*70)
        print(f'RESULTS 1 ({results_path1})')
        print('='*70)
        if total_num[0] > 0:
            print(f'  #Verified:  {num_verif[0]}/{total_num[0]} [{num_verif[0]/total_num[0]*100:.1f}%]')
            print(f'  #Falsified: {num_fals[0]}/{total_num[0]} [{num_fals[0]/total_num[0]*100:.1f}%]')
            print(f'  #Unknown:   {num_unknown[0]}/{total_num[0]} [{num_unknown[0]/total_num[0]*100:.1f}%]')
            print('-'*70)
            solved = num_verif[0] + num_fals[0]
            print(f'  Solved:     {solved}/{total_num[0]} [{solved/total_num[0]*100:.1f}%]')
        print('='*70)
        
        # Print summary for second results
        print()
        print('='*70)
        print(f'RESULTS 2 ({results_path2})')
        print('='*70)
        if total_num[1] > 0:
            print(f'  #Verified:  {num_verif[1]}/{total_num[1]} [{num_verif[1]/total_num[1]*100:.1f}%]')
            print(f'  #Falsified: {num_fals[1]}/{total_num[1]} [{num_fals[1]/total_num[1]*100:.1f}%]')
            print(f'  #Unknown:   {num_unknown[1]}/{total_num[1]} [{num_unknown[1]/total_num[1]*100:.1f}%]')
            print('-'*70)
            solved = num_verif[1] + num_fals[1]
            print(f'  Solved:     {solved}/{total_num[1]} [{solved/total_num[1]*100:.1f}%]')
        print('='*70)
        print()


def read_results_file(file_path: str) -> List[Dict[str, str]]:
    """
    Read a results CSV file.
    
    Args:
        file_path: Path to results.csv
        
    Returns:
        List of result dictionaries
    """
    results = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 6:
                results.append({
                    'name': row[0],
                    'onnxPath': row[1],
                    'vnnlibPath': row[2],
                    'prepTime': float(row[3]) if row[3] else 0.0,
                    'result': row[4],
                    'verifTime': float(row[5]) if row[5] else 0.0,
                })
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare VNN-COMP results from two runs')
    parser.add_argument('results_path1', type=str, help='First results directory path')
    parser.add_argument('results_path2', type=str, help='Second results directory path')
    
    args = parser.parse_args()
    
    compare_results(args.results_path1, args.results_path2)

