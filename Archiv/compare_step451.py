"""
Compare Step 451 between Python trace and identify the source of error_adm_horizon
"""

import re

def extract_step_data(step_num):
    """Extract all relevant data from a step trace file"""
    fname = f'intermediate_values_step{step_num}_inner_loop.txt'
    
    try:
        with open(fname, 'r') as f:
            content = f.read()
        
        data = {
            'step': step_num,
            'initial_error_adm_horizon': None,
            'algorithm': None,
            'tensorOrder': None,
            'iterations': []
        }
        
        # Extract header
        match = re.search(r'Initial error_adm_horizon:\s*\[\[([\d.e+-]+)\]', content)
        if match:
            data['initial_error_adm_horizon'] = float(match.group(1))
        
        match = re.search(r'Algorithm:\s*(\w+)', content)
        if match:
            data['algorithm'] = match.group(1)
        
        match = re.search(r'TensorOrder:\s*(\d+)', content)
        if match:
            data['tensorOrder'] = int(match.group(1))
        
        # Extract iterations
        iter_blocks = re.findall(r'--- Inner Loop Iteration (\d+) ---(.*?)(?=--- Inner Loop Iteration|\n=== Inner Loop Complete|$)', 
                                content, re.DOTALL)
        
        for iter_num, iter_content in iter_blocks:
            iter_data = {}
            
            # Extract key values
            patterns = {
                'error_adm': r'error_adm:\s*\[([\d.e+-]+)',
                'error_adm_max': r'error_adm_max:\s*([\d.e+-]+)',
                'RallError_radius_max': r'RallError radius_max:\s*([\d.e+-]+)',
                'Rmax_radius_max': r'Rmax radius_max:\s*([\d.e+-]+)',
                'VerrorDyn_radius_max': r'VerrorDyn radius_max:\s*([\d.e+-]+)',
                'trueError_max': r'trueError_max:\s*([\d.e+-]+)',
                'perfIndCurr': r'perfIndCurr:\s*([\d.e+-]+)',
                'error_adm_updated': r'error_adm updated:\s*\[([\d.e+-]+)',
                'error_adm_max_updated': r'error_adm_max updated:\s*([\d.e+-]+)',
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, iter_content)
                if match:
                    iter_data[key] = float(match.group(1))
            
            iter_data['iteration'] = int(iter_num)
            data['iterations'].append(iter_data)
        
        return data
    except Exception as e:
        print(f"Error reading {fname}: {e}")
        return None

def compare_steps():
    """Compare Step 450 and 451 to find the source of the jump"""
    print("=== Step 450-451 Comparison ===\n")
    
    step450 = extract_step_data(450)
    step451 = extract_step_data(451)
    
    if not step450 or not step451:
        print("Could not read step data")
        return
    
    print(f"Step 450:")
    print(f"  Initial error_adm_horizon: {step450['initial_error_adm_horizon']:.6e}")
    if step450['iterations']:
        last_iter = step450['iterations'][-1]
        print(f"  Final iteration {last_iter['iteration']}:")
        print(f"    trueError_max: {last_iter.get('trueError_max', 'N/A'):.6e}" if 'trueError_max' in last_iter else "    trueError_max: N/A")
        print(f"    error_adm_max_updated: {last_iter.get('error_adm_max_updated', 'N/A'):.6e}" if 'error_adm_max_updated' in last_iter else "    error_adm_max_updated: N/A")
    
    print(f"\nStep 451:")
    print(f"  Initial error_adm_horizon: {step451['initial_error_adm_horizon']:.6e}")
    
    # Calculate the jump
    if step450['iterations'] and 'trueError_max' in step450['iterations'][-1]:
        expected = step450['iterations'][-1]['trueError_max']
        actual = step451['initial_error_adm_horizon']
        jump_ratio = actual / expected if expected > 0 else 0
        print(f"\n  Expected (from Step 450 trueError): {expected:.6e}")
        print(f"  Actual (Step 451 initial): {actual:.6e}")
        print(f"  Jump ratio: {jump_ratio:.2f}x")
        
        if jump_ratio > 1.1:
            print(f"  [WARNING] Large jump detected! error_adm_horizon is NOT being set to trueError")
            print(f"  This suggests error_adm_horizon is being set from a different source")
    
    # Check if it matches error_adm_updated
    if step450['iterations'] and 'error_adm_max_updated' in step450['iterations'][-1]:
        error_adm_updated = step450['iterations'][-1]['error_adm_max_updated']
        ratio = step451['initial_error_adm_horizon'] / error_adm_updated if error_adm_updated > 0 else 0
        print(f"\n  Step 450 error_adm_max_updated: {error_adm_updated:.6e}")
        print(f"  Ratio to Step 451 initial: {ratio:.2f}x")
        if abs(ratio - 1.0) < 0.01:
            print(f"  [MATCH] Step 451 starts with Step 450's error_adm_updated value!")
        elif abs(ratio - 1.1) < 0.01:
            print(f"  [MATCH] Step 451 starts with 1.1 * Step 450's error_adm_updated value!")
    
    # Analyze the update logic
    print("\n=== Update Logic Analysis ===")
    print("\nFrom code:")
    print("  1. Inner loop: error_adm = 1.1 * trueError")
    print("  2. After inner loop (line 440/452): options['error_adm_horizon'] = trueError")
    print("  3. Next step starts with: error_adm = options['error_adm_horizon']")
    print("\n  Expected flow:")
    print("    Step N: trueError -> error_adm_horizon for Step N+1")
    print("    Step N+1: error_adm = error_adm_horizon (from Step N)")
    
    # Check if there's a mismatch
    if step450['iterations']:
        step450_final_trueError = step450['iterations'][-1].get('trueError_max', None)
        if step450_final_trueError:
            expected_next = step450_final_trueError
            actual_next = step451['initial_error_adm_horizon']
            if abs(actual_next - expected_next) / expected_next > 0.1:
                print(f"\n  [ISSUE] Mismatch detected:")
                print(f"    Expected (Step 450 trueError): {expected_next:.6e}")
                print(f"    Actual (Step 451 initial): {actual_next:.6e}")
                print(f"    This suggests error_adm_horizon is NOT being set correctly!")

if __name__ == '__main__':
    compare_steps()
