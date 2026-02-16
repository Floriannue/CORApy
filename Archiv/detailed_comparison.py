"""Detailed comparison of MATLAB and Python intermediate values"""
import re
import glob
import numpy as np

def extract_all_values(fname):
    """Extract all numeric values from trace file"""
    values = {}
    try:
        with open(fname, 'r') as f:
            content = f.read()
        
        # Extract all numeric values with their labels
        patterns = {
            'error_adm_max': r'error_adm_max:\s*([\d.e+-]+)',
            'RallError_radius_max': r'RallError radius_max:\s*([\d.e+-]+)',
            'Rmax_radius_max': r'Rmax radius_max:\s*([\d.e+-]+)',
            'VerrorDyn_radius_max': r'VerrorDyn radius_max:\s*([\d.e+-]+)',
            'trueError_max': r'trueError_max:\s*([\d.e+-]+)',
            'perfIndCurr': r'perfIndCurr:\s*([\d.e+-]+)',
            'initial_error_adm_horizon': r'Initial error_adm_horizon:.*?([\d.e+-]+)',
            'final_error_adm_horizon': r'Final error_adm_horizon:.*?([\d.e+-]+)',
        }
        
        for key, pattern in patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                # For error_adm_max, get the FIRST iteration's value (not last)
                # This matches the comparison script logic
                if key == 'error_adm_max':
                    # Get first iteration's error_adm_max
                    first_iter_match = re.search(r'--- Inner Loop Iteration 1 ---(.*?)(?=--- Inner Loop Iteration 2|--- Inner Loop Complete|===)', content, re.DOTALL)
                    if first_iter_match:
                        first_iter_content = first_iter_match.group(1)
                        first_match = re.search(pattern, first_iter_content)
                        if first_match:
                            try:
                                values[key] = float(first_match.group(1))
                            except:
                                pass
                    # Fallback to last match if first iteration not found
                    if key not in values and matches:
                        try:
                            values[key] = float(matches[0])  # First match, not last
                        except:
                            pass
                else:
                    # Get the last match (most recent value) for other keys
                    try:
                        values[key] = float(matches[-1])
                    except:
                        pass
        
        # Extract arrays
        array_patterns = {
            'error_adm': r'error_adm:\s*\[(.*?)\]',
            'trueError': r'trueError:\s*\[(.*?)\]',
        }
        
        for key, pattern in array_patterns.items():
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    arr_str = match.group(1).replace(';', ',').replace('\n', ' ')
                    # Parse array
                    arr = [float(x.strip()) for x in arr_str.split(',') if x.strip() and not x.strip().startswith('[')]
                    if arr:
                        values[key] = np.array(arr)
                        values[f'{key}_max'] = np.max(arr)
                except:
                    pass
        
        return values
    except Exception as e:
        print(f"Error reading {fname}: {e}")
        return {}

def compare_values(matlab_vals, python_vals, step, tolerance=1e-10):
    """Compare values and report differences"""
    differences = []
    
    # Compare scalar values
    for key in set(list(matlab_vals.keys()) + list(python_vals.keys())):
        if isinstance(matlab_vals.get(key), (int, float)) and isinstance(python_vals.get(key), (int, float)):
            matlab_val = matlab_vals[key]
            python_val = python_vals[key]
            
            if np.isnan(matlab_val) and np.isnan(python_val):
                continue
            
            if np.isinf(matlab_val) and np.isinf(python_val):
                continue
            
            diff = abs(matlab_val - python_val)
            rel_diff = diff / max(abs(matlab_val), abs(python_val), 1e-15)
            
            if diff > tolerance and rel_diff > tolerance:
                differences.append({
                    'key': key,
                    'matlab': matlab_val,
                    'python': python_val,
                    'abs_diff': diff,
                    'rel_diff': rel_diff
                })
    
    return differences

# Compare all steps
print("=== Detailed Comparison: MATLAB vs Python ===\n")

matlab_files = sorted(glob.glob('intermediate_values_step*_inner_loop_matlab.txt'),
                     key=lambda x: int(re.search(r'step(\d+)', x).group(1)))
python_files = sorted(glob.glob('intermediate_values_step*_inner_loop.txt'),
                     key=lambda x: int(re.search(r'step(\d+)', x).group(1)))

print(f"Found {len(matlab_files)} MATLAB files and {len(python_files)} Python files\n")

all_differences = []
for i, (mfile, pfile) in enumerate(zip(matlab_files[:37], python_files[:37])):
    step = i + 1
    matlab_vals = extract_all_values(mfile)
    python_vals = extract_all_values(pfile)
    
    diffs = compare_values(matlab_vals, python_vals, step, tolerance=1e-12)
    if diffs:
        all_differences.append({'step': step, 'diffs': diffs})
        if len(all_differences) <= 5:  # Show first 5 steps with differences
            print(f"Step {step} differences:")
            for d in diffs:
                print(f"  {d['key']}:")
                print(f"    MATLAB: {d['matlab']:.15e}")
                print(f"    Python: {d['python']:.15e}")
                print(f"    Abs diff: {d['abs_diff']:.15e}")
                print(f"    Rel diff: {d['rel_diff']:.15e}")
            print()

if all_differences:
    print(f"\n=== Summary ===")
    print(f"Steps with differences: {len(all_differences)}")
    print(f"Total differences found: {sum(len(d['diffs']) for d in all_differences)}")
else:
    print("\n[OK] All values match within tolerance (1e-12)!")
