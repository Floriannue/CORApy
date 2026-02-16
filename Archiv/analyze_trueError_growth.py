"""
Analyze why Run 1's trueError is growing so rapidly
"""

import re
import numpy as np

def extract_iteration_data(step_num, run_num, iter_num):
    """Extract data for a specific iteration"""
    fname = f'intermediate_values_step{step_num}_inner_loop.txt'
    
    try:
        with open(fname, 'r') as f:
            content = f.read()
        
        # Find the specific run and iteration
        run_pattern = rf'=== Run {run_num} ===(.*?)(?=== Run |=== Step Complete|$)'
        run_match = re.search(run_pattern, content, re.DOTALL)
        if not run_match:
            return None
        
        run_content = run_match.group(1)
        
        # Find the specific iteration
        iter_pattern = rf'--- Inner Loop Iteration {iter_num} ---(.*?)(?=--- Inner Loop Iteration|=== Inner Loop Complete|$)'
        iter_match = re.search(iter_pattern, run_content, re.DOTALL)
        if not iter_match:
            return None
        
        iter_content = iter_match.group(1)
        
        data = {}
        
        # Extract errorSec
        match = re.search(r'errorSec \(after quadMap\) radius_max:\s*([\d.e+-]+)', iter_content)
        if match:
            data['errorSec_radius_max'] = float(match.group(1))
        
        # Extract VerrorDyn
        match = re.search(r'VerrorDyn center:\s*\[([\d.e+-]+)\s+([\d.e+-]+)\]', iter_content)
        if match:
            data['VerrorDyn_center'] = [float(match.group(1)), float(match.group(2))]
        
        match = re.search(r'VerrorDyn radius:\s*\[([\d.e+-]+)\s+([\d.e+-]+)\]', iter_content)
        if match:
            data['VerrorDyn_radius'] = [float(match.group(1)), float(match.group(2))]
        
        match = re.search(r'VerrorDyn radius_max:\s*([\d.e+-]+)', iter_content)
        if match:
            data['VerrorDyn_radius_max'] = float(match.group(1))
        
        # Extract trueError
        match = re.search(r'trueError_max:\s*([\d.e+-]+)', iter_content)
        if match:
            data['trueError_max'] = float(match.group(1))
        
        # Extract Rmax
        match = re.search(r'Rmax radius_max:\s*([\d.e+-]+)', iter_content)
        if match:
            data['Rmax_radius_max'] = float(match.group(1))
        
        # Extract Z
        match = re.search(r'Z \(before quadMap\) radius_max:\s*([\d.e+-]+)', iter_content)
        if match:
            data['Z_radius_max'] = float(match.group(1))
        
        # Extract error_adm
        match = re.search(r'error_adm_max:\s*([\d.e+-]+)', iter_content)
        if match:
            data['error_adm_max'] = float(match.group(1))
        
        return data
    except Exception as e:
        print(f"Error reading {fname}: {e}")
        return None

def analyze_growth():
    """Analyze the growth of trueError components"""
    print("=== Analyzing trueError Growth Components ===\n")
    
    # Compare Step 450 and 451, Run 1, Iteration 1
    step450 = extract_iteration_data(450, 1, 1)
    step451 = extract_iteration_data(451, 1, 1)
    
    if not step450 or not step451:
        print("Could not extract data")
        return
    
    print("Step 450, Run 1, Iteration 1:")
    for key, value in step450.items():
        if isinstance(value, list):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.6e}")
    
    print("\nStep 451, Run 1, Iteration 1:")
    for key, value in step451.items():
        if isinstance(value, list):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.6e}")
    
    print("\n=== Growth Analysis ===")
    
    # Calculate growth factors
    if 'trueError_max' in step450 and 'trueError_max' in step451:
        growth = step451['trueError_max'] / step450['trueError_max']
        print(f"trueError_max growth: {growth:.3f}x")
    
    if 'VerrorDyn_radius_max' in step450 and 'VerrorDyn_radius_max' in step451:
        growth = step451['VerrorDyn_radius_max'] / step450['VerrorDyn_radius_max']
        print(f"VerrorDyn_radius_max growth: {growth:.3f}x")
    
    if 'errorSec_radius_max' in step450 and 'errorSec_radius_max' in step451:
        growth = step451['errorSec_radius_max'] / step450['errorSec_radius_max']
        print(f"errorSec_radius_max growth: {growth:.3f}x")
    
    if 'Z_radius_max' in step450 and 'Z_radius_max' in step451:
        growth = step451['Z_radius_max'] / step450['Z_radius_max']
        print(f"Z_radius_max growth: {growth:.3f}x")
    
    if 'Rmax_radius_max' in step450 and 'Rmax_radius_max' in step451:
        growth = step451['Rmax_radius_max'] / step450['Rmax_radius_max']
        print(f"Rmax_radius_max growth: {growth:.3f}x")
    
    if 'error_adm_max' in step450 and 'error_adm_max' in step451:
        growth = step451['error_adm_max'] / step450['error_adm_max']
        print(f"error_adm_max growth: {growth:.3f}x")
    
    print("\n=== Component Contribution ===")
    # trueError = |VerrorDyn.center()| + sum(|VerrorDyn.generators()|)
    # For Step 450:
    if 'VerrorDyn_center' in step450 and 'VerrorDyn_radius_max' in step450:
        center_contrib_450 = np.max(np.abs(step450['VerrorDyn_center']))
        radius_contrib_450 = step450['VerrorDyn_radius_max']
        print(f"Step 450 VerrorDyn:")
        print(f"  Center contribution: {center_contrib_450:.6e}")
        print(f"  Radius contribution: {radius_contrib_450:.6e}")
        print(f"  Total (approx): {center_contrib_450 + radius_contrib_450:.6e}")
        print(f"  Actual trueError_max: {step450['trueError_max']:.6e}")
    
    if 'VerrorDyn_center' in step451 and 'VerrorDyn_radius_max' in step451:
        center_contrib_451 = np.max(np.abs(step451['VerrorDyn_center']))
        radius_contrib_451 = step451['VerrorDyn_radius_max']
        print(f"\nStep 451 VerrorDyn:")
        print(f"  Center contribution: {center_contrib_451:.6e}")
        print(f"  Radius contribution: {radius_contrib_451:.6e}")
        print(f"  Total (approx): {center_contrib_451 + radius_contrib_451:.6e}")
        print(f"  Actual trueError_max: {step451['trueError_max']:.6e}")
        
        # Growth in components
        if 'VerrorDyn_center' in step450:
            center_contrib_450 = np.max(np.abs(step450['VerrorDyn_center']))
            center_growth = center_contrib_451 / center_contrib_450 if center_contrib_450 > 0 else float('inf')
            radius_growth = radius_contrib_451 / radius_contrib_450 if radius_contrib_450 > 0 else float('inf')
            print(f"\nComponent growth:")
            print(f"  Center growth: {center_growth:.3f}x")
            print(f"  Radius growth: {radius_growth:.3f}x")

if __name__ == '__main__':
    analyze_growth()
