"""
Analyze how trueError grows across iterations within Run 1
"""

import re
import numpy as np

def extract_all_iterations(step_num, run_num):
    """Extract all iterations for a specific run"""
    fname = f'intermediate_values_step{step_num}_inner_loop.txt'
    
    try:
        with open(fname, 'r') as f:
            content = f.read()
        
        # Find the specific run
        run_pattern = rf'=== Run {run_num} ===(.*?)(?=== Run |=== Step Complete|$)'
        run_match = re.search(run_pattern, content, re.DOTALL)
        if not run_match:
            return []
        
        run_content = run_match.group(1)
        
        # Find all iterations
        iterations = []
        iter_pattern = r'--- Inner Loop Iteration (\d+) ---(.*?)(?=--- Inner Loop Iteration|=== Inner Loop Complete|$)'
        for match in re.finditer(iter_pattern, run_content, re.DOTALL):
            iter_num = int(match.group(1))
            iter_content = match.group(2)
            
            data = {'iter': iter_num}
            
            # Extract key values
            m = re.search(r'error_adm_max:\s*([\d.e+-]+)', iter_content)
            if m:
                data['error_adm_max'] = float(m.group(1))
            
            m = re.search(r'VerrorDyn_radius_max:\s*([\d.e+-]+)', iter_content)
            if m:
                data['VerrorDyn_radius_max'] = float(m.group(1))
            
            m = re.search(r'trueError_max:\s*([\d.e+-]+)', iter_content)
            if m:
                data['trueError_max'] = float(m.group(1))
            
            m = re.search(r'errorSec \(after quadMap\) radius_max:\s*([\d.e+-]+)', iter_content)
            if m:
                data['errorSec_radius_max'] = float(m.group(1))
            
            m = re.search(r'Z \(before quadMap\) radius_max:\s*([\d.e+-]+)', iter_content)
            if m:
                data['Z_radius_max'] = float(m.group(1))
            
            iterations.append(data)
        
        return iterations
    except Exception as e:
        print(f"Error reading {fname}: {e}")
        return []

def analyze_iteration_growth():
    """Analyze how trueError grows across iterations"""
    print("=== Analyzing trueError Growth Across Iterations ===\n")
    
    step450_iters = extract_all_iterations(450, 1)
    step451_iters = extract_all_iterations(451, 1)
    
    print("Step 450, Run 1:")
    for iter_data in step450_iters:
        print(f"  Iteration {iter_data['iter']}:")
        if 'error_adm_max' in iter_data:
            print(f"    error_adm_max: {iter_data['error_adm_max']:.6e}")
        if 'VerrorDyn_radius_max' in iter_data:
            print(f"    VerrorDyn_radius_max: {iter_data['VerrorDyn_radius_max']:.6e}")
        if 'trueError_max' in iter_data:
            print(f"    trueError_max: {iter_data['trueError_max']:.6e}")
        if 'errorSec_radius_max' in iter_data:
            print(f"    errorSec_radius_max: {iter_data['errorSec_radius_max']:.6e}")
        if 'Z_radius_max' in iter_data:
            print(f"    Z_radius_max: {iter_data['Z_radius_max']:.6e}")
    
    print("\nStep 451, Run 1:")
    for iter_data in step451_iters:
        print(f"  Iteration {iter_data['iter']}:")
        if 'error_adm_max' in iter_data:
            print(f"    error_adm_max: {iter_data['error_adm_max']:.6e}")
        if 'VerrorDyn_radius_max' in iter_data:
            print(f"    VerrorDyn_radius_max: {iter_data['VerrorDyn_radius_max']:.6e}")
        if 'trueError_max' in iter_data:
            print(f"    trueError_max: {iter_data['trueError_max']:.6e}")
        if 'errorSec_radius_max' in iter_data:
            print(f"    errorSec_radius_max: {iter_data['errorSec_radius_max']:.6e}")
        if 'Z_radius_max' in iter_data:
            print(f"    Z_radius_max: {iter_data['Z_radius_max']:.6e}")
    
    # Compare first and last iterations
    print("\n=== Growth Analysis ===")
    
    if len(step450_iters) > 0 and len(step451_iters) > 0:
        step450_first = step450_iters[0]
        step450_last = step450_iters[-1]
        step451_first = step451_iters[0]
        step451_last = step451_iters[-1]
        
        print("Step 450, Run 1:")
        if 'trueError_max' in step450_first and 'trueError_max' in step450_last:
            growth = step450_last['trueError_max'] / step450_first['trueError_max']
            print(f"  trueError growth (iter 1 -> last): {growth:.3f}x")
            print(f"    First: {step450_first['trueError_max']:.6e}")
            print(f"    Last:  {step450_last['trueError_max']:.6e}")
        
        print("\nStep 451, Run 1:")
        if 'trueError_max' in step451_first and 'trueError_max' in step451_last:
            growth = step451_last['trueError_max'] / step451_first['trueError_max']
            print(f"  trueError growth (iter 1 -> last): {growth:.3f}x")
            print(f"    First: {step451_first['trueError_max']:.6e}")
            print(f"    Last:  {step451_last['trueError_max']:.6e}")
        
        print("\nStep 450 -> Step 451:")
        if 'trueError_max' in step450_last and 'trueError_max' in step451_last:
            growth = step451_last['trueError_max'] / step450_last['trueError_max']
            print(f"  trueError growth (final): {growth:.3f}x")
            print(f"    Step 450 final: {step450_last['trueError_max']:.6e}")
            print(f"    Step 451 final: {step451_last['trueError_max']:.6e}")
        
        # Check error_adm growth
        if 'error_adm_max' in step450_first and 'error_adm_max' in step451_first:
            growth = step451_first['error_adm_max'] / step450_first['error_adm_max']
            print(f"\n  error_adm growth (initial): {growth:.3f}x")
            print(f"    Step 450 initial: {step450_first['error_adm_max']:.6e}")
            print(f"    Step 451 initial: {step451_first['error_adm_max']:.6e}")

if __name__ == '__main__':
    analyze_iteration_growth()
