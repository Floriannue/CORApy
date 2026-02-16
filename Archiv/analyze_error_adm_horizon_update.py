"""
Analyze error_adm_horizon update logic
Compare how it's updated in Python vs MATLAB
"""

import re
import glob

def extract_error_adm_horizon_flow(step_num):
    """Extract the flow of error_adm_horizon for a given step"""
    fname = f'intermediate_values_step{step_num}_inner_loop.txt'
    
    try:
        with open(fname, 'r') as f:
            content = f.read()
            
        result = {
            'step': step_num,
            'initial_error_adm_horizon': None,
            'iterations': []
        }
        
        # Extract initial error_adm_horizon
        match = re.search(r'Initial error_adm_horizon:\s*\[\[([\d.e+-]+)\]', content)
        if match:
            result['initial_error_adm_horizon'] = float(match.group(1))
        
        # Extract each iteration
        iterations = re.findall(r'--- Inner Loop Iteration (\d+) ---(.*?)(?=--- Inner Loop Iteration|\n=== Inner Loop Complete|$)', 
                               content, re.DOTALL)
        
        for iter_num, iter_content in iterations:
            iter_data = {
                'iteration': int(iter_num),
                'error_adm': None,
                'error_adm_max': None,
                'trueError': None,
                'trueError_max': None,
                'perfIndCurr': None,
                'error_adm_updated': None,
                'error_adm_max_updated': None
            }
            
            # Extract error_adm
            match = re.search(r'error_adm:\s*\[([\d.e+-]+)', iter_content)
            if match:
                iter_data['error_adm'] = float(match.group(1))
            
            match = re.search(r'error_adm_max:\s*([\d.e+-]+)', iter_content)
            if match:
                iter_data['error_adm_max'] = float(match.group(1))
            
            # Extract trueError
            match = re.search(r'trueError:\s*\[([\d.e+-]+)', iter_content)
            if match:
                iter_data['trueError'] = float(match.group(1))
            
            match = re.search(r'trueError_max:\s*([\d.e+-]+)', iter_content)
            if match:
                iter_data['trueError_max'] = float(match.group(1))
            
            # Extract perfIndCurr
            match = re.search(r'perfIndCurr:\s*([\d.e+-]+)', iter_content)
            if match:
                iter_data['perfIndCurr'] = float(match.group(1))
            
            # Extract updated error_adm
            match = re.search(r'error_adm updated:\s*\[([\d.e+-]+)', iter_content)
            if match:
                iter_data['error_adm_updated'] = float(match.group(1))
            
            match = re.search(r'error_adm_max updated:\s*([\d.e+-]+)', iter_content)
            if match:
                iter_data['error_adm_max_updated'] = float(match.group(1))
            
            result['iterations'].append(iter_data)
        
        return result
    except Exception as e:
        print(f"Error reading {fname}: {e}")
        return None

def analyze_update_logic():
    """Analyze how error_adm_horizon is updated"""
    print("=== Analyzing error_adm_horizon Update Logic ===\n")
    
    # Analyze Step 450 and 451 (before and at explosion)
    steps = [450, 451]
    
    for step in steps:
        data = extract_error_adm_horizon_flow(step)
        if not data:
            continue
        
        print(f"Step {step}:")
        print(f"  Initial error_adm_horizon: {data['initial_error_adm_horizon']:.6e}")
        print(f"  Iterations: {len(data['iterations'])}")
        
        for iter_data in data['iterations']:
            print(f"\n  Iteration {iter_data['iteration']}:")
            print(f"    error_adm (input): {iter_data['error_adm']:.6e}")
            print(f"    trueError (computed): {iter_data['trueError_max']:.6e}")
            print(f"    perfIndCurr: {iter_data['perfIndCurr']:.6e}")
            if iter_data['error_adm_updated']:
                print(f"    error_adm updated: {iter_data['error_adm_updated']:.6e}")
                ratio = iter_data['error_adm_updated'] / iter_data['trueError_max'] if iter_data['trueError_max'] > 0 else 0
                print(f"    Update ratio (error_adm_updated / trueError): {ratio:.6f}")
        
        # Check how error_adm_horizon is set for next step
        if data['iterations']:
            last_iter = data['iterations'][-1]
            if last_iter['error_adm_updated']:
                print(f"\n  Next step will start with error_adm_horizon ≈ {last_iter['error_adm_updated']:.6e}")
                if step == 450:
                    # Check what Step 451 actually started with
                    next_data = extract_error_adm_horizon_flow(step + 1)
                    if next_data:
                        print(f"  Step {step+1} actually started with: {next_data['initial_error_adm_horizon']:.6e}")
                        if next_data['initial_error_adm_horizon'] != last_iter['error_adm_updated']:
                            print(f"  [WARNING] Mismatch! Expected {last_iter['error_adm_updated']:.6e}, got {next_data['initial_error_adm_horizon']:.6e}")
                            print(f"  Difference: {abs(next_data['initial_error_adm_horizon'] - last_iter['error_adm_updated']):.6e}")
        
        print("\n" + "="*60 + "\n")
    
    # Analyze the update formula
    print("=== Update Formula Analysis ===\n")
    print("From code analysis:")
    print("  In inner loop: error_adm = 1.1 * trueError  (line 243 in MATLAB, line 243 in Python)")
    print("  After inner loop: options.error_adm_horizon = trueError  (line 336/348 in MATLAB, line 440/452 in Python)")
    print("\n  Key observation:")
    print("  - Inner loop updates error_adm = 1.1 * trueError")
    print("  - But error_adm_horizon is set to trueError (not 1.1 * trueError)")
    print("  - This means error_adm_horizon for next step = trueError from current step")
    print("  - If trueError grows, error_adm_horizon grows, creating feedback loop")
    
    # Check if there's a pattern
    print("\n=== Growth Pattern Analysis ===\n")
    trace_files = sorted(glob.glob('intermediate_values_step*_inner_loop.txt'),
                        key=lambda x: int(re.search(r'step(\d+)', x).group(1)))
    
    # Sample a few steps to see the pattern
    sample_steps = [1, 100, 200, 300, 400, 450, 451, 500, 600, 700, 800]
    print("Step | Initial error_adm_horizon | Final trueError | Ratio")
    print("-" * 70)
    
    for step_num in sample_steps:
        fname = f'intermediate_values_step{step_num}_inner_loop.txt'
        if fname in trace_files:
            data = extract_error_adm_horizon_flow(step_num)
            if data and data['iterations']:
                initial = data['initial_error_adm_horizon']
                final_trueError = data['iterations'][-1]['trueError_max']
                ratio = final_trueError / initial if initial > 0 else 0
                print(f"{step_num:4d} | {initial:20.6e} | {final_trueError:15.6e} | {ratio:6.2f}x")

if __name__ == '__main__':
    analyze_update_logic()
