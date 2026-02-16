"""
Compare MATLAB and Python trace files step by step
Identifies differences in intermediate values
"""

import re
import glob
import os
import sys

def extract_values_from_trace(fname):
    """Extract all tracked values from a trace file"""
    try:
        with open(fname, 'r') as f:
            content = f.read()
        
        values = {}
        
        # Extract step number
        step_match = re.search(r'Step (\d+)', content)
        if step_match:
            values['step'] = int(step_match.group(1))
        
        # Extract initial error_adm_horizon
        match = re.search(r'Initial error_adm_horizon:\s*\[\[([\d.e+-]+)\]', content)
        if match:
            values['initial_error_adm_horizon'] = float(match.group(1))
        
        # Extract Run 1 and Run 2 data
        run1_match = re.search(r'=== Run 1 ===(.*?)(?=== Run 2|=== Step Complete|$)', content, re.DOTALL)
        run2_match = re.search(r'=== Run 2 ===(.*?)(?=== Step Complete|$)', content, re.DOTALL)
        
        if run1_match:
            run1_content = run1_match.group(1)
            # Extract last iteration's values from Run 1
            iterations = re.findall(r'--- Inner Loop Iteration (\d+) ---(.*?)(?=--- Inner Loop Iteration|=== Inner Loop Complete|$)', 
                                   run1_content, re.DOTALL)
            if iterations:
                last_iter = iterations[-1][1]
                match = re.search(r'trueError_max:\s*([\d.e+-]+)', last_iter)
                if match:
                    values['run1_trueError_max'] = float(match.group(1))
                match = re.search(r'error_adm_max:\s*([\d.e+-]+)', last_iter)
                if match:
                    values['run1_error_adm_max'] = float(match.group(1))
                match = re.search(r'VerrorDyn_radius_max:\s*([\d.e+-]+)', last_iter)
                if match:
                    values['run1_VerrorDyn_radius_max'] = float(match.group(1))
        
        # Extract Run 1 error_adm_horizon update
        match = re.search(r'=== Run 1, Step \d+: error_adm_horizon Update ===.*?error_adm_horizon max:\s*([\d.e+-]+)', 
                          content, re.DOTALL)
        if match:
            values['run1_error_adm_horizon_set'] = float(match.group(1))
        
        if run2_match:
            run2_content = run2_match.group(1)
            # Extract last iteration's values from Run 2
            iterations = re.findall(r'--- Inner Loop Iteration (\d+) ---(.*?)(?=--- Inner Loop Iteration|=== Inner Loop Complete|$)', 
                                   run2_content, re.DOTALL)
            if iterations:
                last_iter = iterations[-1][1]
                match = re.search(r'trueError_max:\s*([\d.e+-]+)', last_iter)
                if match:
                    values['run2_trueError_max'] = float(match.group(1))
        
        # Extract Run 2 error_adm_Deltatopt update
        match = re.search(r'=== Run 2, Step \d+: error_adm_Deltatopt Update ===.*?error_adm_Deltatopt max:\s*([\d.e+-]+)', 
                          content, re.DOTALL)
        if match:
            values['run2_error_adm_Deltatopt'] = float(match.group(1))
        
        # Extract final values
        match = re.search(r'Final error_adm_horizon:\s*\[\[([\d.e+-]+)\]', content)
        if match:
            values['final_error_adm_horizon'] = float(match.group(1))
        
        match = re.search(r'Final error_adm_Deltatopt:\s*\[\[([\d.e+-]+)\]', content)
        if match:
            values['final_error_adm_Deltatopt'] = float(match.group(1))
        
        return values
    except Exception as e:
        print(f"Error reading {fname}: {e}")
        return None

def compare_traces(matlab_dir='.', python_dir='.', tolerance=1e-6):
    """Compare MATLAB and Python trace files"""
    print("=== Comparing MATLAB and Python Trace Files ===\n")
    
    # Find all trace files
    matlab_files = sorted(glob.glob(os.path.join(matlab_dir, 'intermediate_values_step*_inner_loop_matlab.txt')),
                         key=lambda x: int(re.search(r'step(\d+)', x).group(1)) if re.search(r'step(\d+)', x) else 0)
    python_files = sorted(glob.glob(os.path.join(python_dir, 'intermediate_values_step*_inner_loop.txt')),
                         key=lambda x: int(re.search(r'step(\d+)', x).group(1)) if re.search(r'step(\d+)', x) else 0)
    
    print(f"Found {len(matlab_files)} MATLAB trace files")
    print(f"Found {len(python_files)} Python trace files\n")
    
    # Find common steps
    matlab_steps = {int(re.search(r'step(\d+)', f).group(1)): f for f in matlab_files if re.search(r'step(\d+)', f)}
    python_steps = {int(re.search(r'step(\d+)', f).group(1)): f for f in python_files if re.search(r'step(\d+)', f)}
    
    common_steps = sorted(set(matlab_steps.keys()) & set(python_steps.keys()))
    print(f"Found {len(common_steps)} common steps\n")
    
    if not common_steps:
        print("No common steps found!")
        return
    
    # Compare each step - focus on early steps and explosion region
    differences = []
    # Compare first 50 steps, steps around 100, 200, 300, 400, 450, 451, 452, 500, and last 10 steps
    steps_to_compare = list(common_steps[:50]) + \
                      [s for s in common_steps if 95 <= s <= 105] + \
                      [s for s in common_steps if 195 <= s <= 205] + \
                      [s for s in common_steps if 295 <= s <= 305] + \
                      [s for s in common_steps if 395 <= s <= 405] + \
                      [s for s in common_steps if 445 <= s <= 455] + \
                      [s for s in common_steps if 495 <= s <= 505] + \
                      list(common_steps[-10:])
    steps_to_compare = sorted(set(steps_to_compare))
    
    for step in steps_to_compare:
        matlab_vals = extract_values_from_trace(matlab_steps[step])
        python_vals = extract_values_from_trace(python_steps[step])
        
        if not matlab_vals or not python_vals:
            continue
        
        step_diffs = []
        
        # Compare each value
        for key in ['initial_error_adm_horizon', 'run1_trueError_max', 'run1_error_adm_horizon_set', 
                   'run2_trueError_max', 'run2_error_adm_Deltatopt', 'final_error_adm_horizon']:
            if key in matlab_vals and key in python_vals:
                matlab_val = matlab_vals[key]
                python_val = python_vals[key]
                diff = abs(matlab_val - python_val)
                rel_diff = diff / max(abs(matlab_val), abs(python_val), 1e-15)
                
                if diff > tolerance and rel_diff > tolerance:
                    step_diffs.append({
                        'key': key,
                        'matlab': matlab_val,
                        'python': python_val,
                        'abs_diff': diff,
                        'rel_diff': rel_diff
                    })
        
        if step_diffs:
            differences.append({'step': step, 'diffs': step_diffs})
    
    # Report differences
    if differences:
        print(f"Found differences in {len(differences)} steps:\n")
        for diff_info in differences[:10]:  # Show first 10 steps with differences
            print(f"Step {diff_info['step']}:")
            for d in diff_info['diffs']:
                print(f"  {d['key']}:")
                print(f"    MATLAB: {d['matlab']:.6e}")
                print(f"    Python: {d['python']:.6e}")
                print(f"    Abs diff: {d['abs_diff']:.6e}")
                print(f"    Rel diff: {d['rel_diff']:.6e}")
            print()
    else:
        print("No significant differences found!")
    
    # Summary statistics
    if common_steps:
        print(f"\n=== Summary ===")
        print(f"Compared {len(steps_to_compare)} steps (including explosion region)")
        print(f"Steps with differences: {len(differences)}")
        if len(differences) == 0:
            print("All values match within tolerance!")

if __name__ == '__main__':
    compare_traces()
