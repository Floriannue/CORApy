"""
Analyze run == 2 tracking to understand error_adm_horizon source
"""

import re
import glob

def extract_step_data(step_num):
    """Extract all relevant data from a step trace file"""
    fname = f'intermediate_values_step{step_num}_inner_loop.txt'
    
    try:
        with open(fname, 'r') as f:
            content = f.read()
        
        data = {
            'step': step_num,
            'initial_error_adm_horizon': None,
            'final_error_adm_horizon': None,
            'final_error_adm_Deltatopt': None,
            'run1_final_trueError': None,
            'run2_final_trueError': None,
            'run1_error_adm_horizon_set': None,
            'iterations': []
        }
        
        # Extract initial error_adm_horizon
        match = re.search(r'Initial error_adm_horizon:\s*\[\[([\d.e+-]+)\]', content)
        if match:
            data['initial_error_adm_horizon'] = float(match.group(1))
        
        # Extract final values
        match = re.search(r'Final error_adm_horizon:\s*\[\[([\d.e+-]+)\]', content)
        if match:
            data['final_error_adm_horizon'] = float(match.group(1))
        
        match = re.search(r'Final error_adm_Deltatopt:\s*\[\[([\d.e+-]+)\]', content)
        if match:
            data['final_error_adm_Deltatopt'] = float(match.group(1))
        
        # Extract Run 1 error_adm_horizon update
        match = re.search(r'=== Run 1, Step \d+: error_adm_horizon Update ===.*?trueError_max:\s*([\d.e+-]+)', content, re.DOTALL)
        if match:
            data['run1_final_trueError'] = float(match.group(1))
            # Also get the value it was set to
            match2 = re.search(r'error_adm_horizon SET TO:.*?error_adm_horizon max:\s*([\d.e+-]+)', content, re.DOTALL)
            if match2:
                data['run1_error_adm_horizon_set'] = float(match2.group(1))
        
        # Extract Run 2 error_adm_Deltatopt update
        match = re.search(r'=== Run 2, Step \d+: error_adm_Deltatopt Update ===.*?trueError_max:\s*([\d.e+-]+)', content, re.DOTALL)
        if match:
            data['run2_final_trueError'] = float(match.group(1))
        
        return data
    except Exception as e:
        print(f"Error reading {fname}: {e}")
        return None

def analyze_run2_flow():
    """Analyze the flow of error_adm_horizon through run 1 and run 2"""
    print("=== Analyzing Run 1 -> Run 2 -> Next Step Flow ===\n")
    
    steps = [449, 450, 451]
    
    for step in steps:
        data = extract_step_data(step)
        if not data:
            continue
        
        print(f"Step {step}:")
        print(f"  Initial error_adm_horizon: {data['initial_error_adm_horizon']:.6e}")
        if data['run1_final_trueError']:
            print(f"  Run 1 final trueError: {data['run1_final_trueError']:.6e}")
        if data['run1_error_adm_horizon_set']:
            print(f"  Run 1 sets error_adm_horizon to: {data['run1_error_adm_horizon_set']:.6e}")
        if data['run2_final_trueError']:
            print(f"  Run 2 final trueError: {data['run2_final_trueError']:.6e}")
        if data['final_error_adm_Deltatopt']:
            print(f"  Final error_adm_Deltatopt: {data['final_error_adm_Deltatopt']:.6e}")
        if data['final_error_adm_horizon']:
            print(f"  Final error_adm_horizon: {data['final_error_adm_horizon']:.6e}")
        
        # Check next step
        if step < 451:
            next_data = extract_step_data(step + 1)
            if next_data:
                print(f"\n  -> Step {step+1} starts with:")
                print(f"    Initial error_adm_horizon: {next_data['initial_error_adm_horizon']:.6e}")
                
                # Compare
                if data['final_error_adm_horizon'] and next_data['initial_error_adm_horizon']:
                    ratio = next_data['initial_error_adm_horizon'] / data['final_error_adm_horizon']
                    print(f"    Ratio to Step {step} final: {ratio:.3f}x")
                    
                    # Check if it matches run 1's error_adm_horizon
                    if data['run1_error_adm_horizon_set']:
                        ratio1 = next_data['initial_error_adm_horizon'] / data['run1_error_adm_horizon_set']
                        print(f"    Ratio to Step {step} Run 1 error_adm_horizon: {ratio1:.3f}x")
                        if abs(ratio1 - 1.0) < 0.01:
                            print(f"    [MATCH] Step {step+1} starts with Step {step}'s Run 1 error_adm_horizon!")
                    
                    # Check if it matches run 2 trueError
                    if data['run2_final_trueError']:
                        ratio2 = next_data['initial_error_adm_horizon'] / data['run2_final_trueError']
                        print(f"    Ratio to Step {step} Run 2 trueError: {ratio2:.3f}x")
                        if abs(ratio2 - 1.0) < 0.01:
                            print(f"    [MATCH] Step {step+1} starts with Step {step}'s Run 2 trueError!")
        
        print("\n" + "="*60 + "\n")

if __name__ == '__main__':
    analyze_run2_flow()
