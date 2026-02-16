"""Check final values from Python run"""
import glob
import re
import os

print("=== Checking Python Final Values ===\n")

# Check trace files
trace_files = glob.glob('intermediate_values_step*_inner_loop.txt')
trace_files = [f for f in trace_files if '_matlab' not in f]

if trace_files:
    print(f"Found {len(trace_files)} Python trace files")
    
    # Get step numbers
    step_nums = []
    for fname in trace_files:
        match = re.search(r'step(\d+)', fname)
        if match:
            step_nums.append(int(match.group(1)))
    
    if step_nums:
        print(f"Step numbers range: {min(step_nums)} to {max(step_nums)}")
        print(f"Total unique steps: {len(set(step_nums))}")
    
    # Check last trace file
    step_nums.sort()
    last_step = step_nums[-1]
    last_file = f'intermediate_values_step{last_step}_inner_loop.txt'
    
    if os.path.exists(last_file):
        print(f"\nLast trace file: {last_file}")
        
        with open(last_file, 'r') as f:
            content = f.read()
        
        # Extract final error_adm_horizon
        match = re.search(r'Final error_adm_horizon:\s*\[\[([\d.e+-]+)\]', content)
        if match:
            print(f"Final error_adm_horizon: {match.group(1)}")
        
        # Extract step number
        match = re.search(r'Step (\d+)', content)
        if match:
            print(f"Step number in file: {match.group(1)}")

print("\n=== Expected Final Time ===")
print("tFinal should be: 2.0")
print("If Python did more steps, check termination logic")
