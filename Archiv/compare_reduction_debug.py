"""
Compare reduction debug data between Python and MATLAB to find divergence
"""
import pickle
import scipy.io
import numpy as np

# Load Python debug data
try:
    with open('initReach_debug.pkl', 'rb') as f:
        python_debug = pickle.load(f)
    print("✓ Loaded Python debug data")
except FileNotFoundError:
    print("✗ Python debug file not found")
    python_debug = None

# Load MATLAB debug data
try:
    matlab_data = scipy.io.loadmat('initReach_debug.mat')
    matlab_debug = matlab_data['debug_data']
    print("✓ Loaded MATLAB debug data")
except FileNotFoundError:
    print("✗ MATLAB debug file not found")
    matlab_debug = None

if python_debug is None or matlab_debug is None:
    print("\nCannot compare - missing debug files")
    exit(1)

print(f"\nPython entries: {len(python_debug)}")
print(f"MATLAB entries: {len(matlab_debug)}")

# Compare Step 1 Run 1 (first reduction call)
# This is where Rend.tp is created from Rhom_tp
step = 1
run = 1

print(f"\n{'='*80}")
print(f"Comparing Step {step} Run {run} reduction")
print(f"{'='*80}")

# Find matching entries
python_entry = None
matlab_entry = None

for entry in python_debug:
    if entry.get('step') == step and entry.get('run') == run:
        python_entry = entry
        break

# MATLAB struct access
if isinstance(matlab_debug, np.ndarray):
    if matlab_debug.size > 0:
        # Try to find matching entry
        for i in range(len(matlab_debug)):
            try:
                entry = matlab_debug[i]
                if isinstance(entry, np.ndarray) and entry.size > 0:
                    entry = entry.item()
                if hasattr(entry, 'step') and hasattr(entry, 'run'):
                    if entry.step == step and entry.run == run:
                        matlab_entry = entry
                        break
            except:
                pass

if python_entry is None:
    print(f"✗ Python entry for Step {step} Run {run} not found")
    print("Available Python entries:")
    for entry in python_debug:
        print(f"  Step {entry.get('step')}, Run {entry.get('run')}")
    exit(1)

if matlab_entry is None:
    print(f"✗ MATLAB entry for Step {step} Run {run} not found")
    if isinstance(matlab_debug, np.ndarray) and matlab_debug.size > 0:
        print("Available MATLAB entries:")
        for i in range(min(5, len(matlab_debug))):
            try:
                entry = matlab_debug[i]
                if isinstance(entry, np.ndarray) and entry.size > 0:
                    entry = entry.item()
                if hasattr(entry, 'step'):
                    print(f"  Step {entry.step}, Run {getattr(entry, 'run', '?')}")
            except:
                pass
    exit(1)

print(f"\n✓ Found matching entries")

# Compare key values
print(f"\n{'Key Value':<30} {'Python':<25} {'MATLAB':<25} {'Match':<10}")
print(f"{'-'*90}")

def compare_value(key, py_val, mat_val, tolerance=1e-10):
    """Compare a value and return match status"""
    if isinstance(py_val, (list, np.ndarray)) and isinstance(mat_val, (list, np.ndarray)):
        py_arr = np.asarray(py_val)
        mat_arr = np.asarray(mat_val)
        if py_arr.shape != mat_arr.shape:
            return False, f"Shape mismatch: {py_arr.shape} vs {mat_arr.shape}"
        match = np.allclose(py_arr, mat_arr, atol=tolerance)
        if match:
            return True, "✓"
        else:
            diff = np.abs(py_arr - mat_arr)
            max_diff = np.max(diff)
            return False, f"Max diff: {max_diff:.2e}"
    else:
        try:
            py_float = float(py_val)
            mat_float = float(mat_val)
            match = abs(py_float - mat_float) < tolerance
            if match:
                return True, "✓"
            else:
                return False, f"Diff: {abs(py_float - mat_float):.2e}"
        except:
            return py_val == mat_val, "?"

# Compare Rhom_tp before reduction
py_rhom_tp = python_entry.get('Rhom_tp_before_reduction')
mat_rhom_tp = getattr(matlab_entry, 'Rhom_tp_before_reduction', None)

if py_rhom_tp is not None and mat_rhom_tp is not None:
    py_gens = py_rhom_tp.shape[1] if hasattr(py_rhom_tp, 'shape') else len(py_rhom_tp[0]) if isinstance(py_rhom_tp, list) else None
    if isinstance(mat_rhom_tp, np.ndarray):
        mat_gens = mat_rhom_tp.shape[1] if mat_rhom_tp.ndim > 1 else 1
    else:
        mat_gens = None
    match, status = compare_value('Rhom_tp generators', py_gens, mat_gens)
    print(f"{'Rhom_tp generators':<30} {str(py_gens):<25} {str(mat_gens):<25} {status:<10}")

# Compare Rend.tp after reduction
py_rend_tp = python_entry.get('Rend_tp_after_reduction')
mat_rend_tp = getattr(matlab_entry, 'Rend_tp_after_reduction', None)

if py_rend_tp is not None and mat_rend_tp is not None:
    py_gens = py_rend_tp.shape[1] if hasattr(py_rend_tp, 'shape') else len(py_rend_tp[0]) if isinstance(py_rend_tp, list) else None
    if isinstance(mat_rend_tp, np.ndarray):
        mat_gens = mat_rend_tp.shape[1] if mat_rend_tp.ndim > 1 else 1
    else:
        mat_gens = None
    match, status = compare_value('Rend.tp generators', py_gens, mat_gens)
    print(f"{'Rend.tp generators':<30} {str(py_gens):<25} {str(mat_gens):<25} {status:<10}")

# Compare reduction parameters
keys_to_compare = [
    'diagpercent',
    'dHmax',
    'nrG',
    'last0Idx',
    'redIdx',
    'redIdx_0based',
    'dHerror',
    'gredIdx_len',
]

for key in keys_to_compare:
    py_val = python_entry.get(key)
    mat_val = getattr(matlab_entry, key, None)
    
    if py_val is not None or mat_val is not None:
        match, status = compare_value(key, py_val, mat_val)
        py_str = str(py_val) if py_val is not None else "None"
        mat_str = str(mat_val) if mat_val is not None else "None"
        if len(py_str) > 24:
            py_str = py_str[:21] + "..."
        if len(mat_str) > 24:
            mat_str = mat_str[:21] + "..."
        print(f"{key:<30} {py_str:<25} {mat_str:<25} {status:<10}")

# Compare h array
py_h = python_entry.get('h_computed')
mat_h = getattr(matlab_entry, 'h_computed', None)

if py_h is not None and mat_h is not None:
    py_arr = np.asarray(py_h)
    if isinstance(mat_h, np.ndarray):
        mat_arr = mat_h
    else:
        mat_arr = np.asarray(mat_h)
    
    if py_arr.shape == mat_arr.shape:
        match = np.allclose(py_arr, mat_arr, atol=1e-10)
        max_diff = np.max(np.abs(py_arr - mat_arr)) if not match else 0
        status = "✓" if match else f"Max diff: {max_diff:.2e}"
        print(f"{'h_computed':<30} {f'Shape {py_arr.shape}':<25} {f'Shape {mat_arr.shape}':<25} {status:<10}")
        
        # Show first few values
        print(f"\nFirst 5 h values:")
        print(f"  Python: {py_arr[:5]}")
        print(f"  MATLAB: {mat_arr[:5]}")
    else:
        print(f"{'h_computed':<30} {f'Shape {py_arr.shape}':<25} {f'Shape {mat_arr.shape}':<25} {'Shape mismatch':<10}")

# Compare redIdx_arr
py_redIdx_arr = python_entry.get('redIdx_arr')
mat_redIdx_arr = getattr(matlab_entry, 'redIdx_arr', None)

if py_redIdx_arr is not None and mat_redIdx_arr is not None:
    py_arr = np.asarray(py_redIdx_arr)
    if isinstance(mat_redIdx_arr, np.ndarray):
        mat_arr = mat_redIdx_arr
    else:
        mat_arr = np.asarray(mat_redIdx_arr)
    
    match = np.array_equal(py_arr, mat_arr)
    status = "✓" if match else "✗"
    print(f"\n{'redIdx_arr':<30} {str(py_arr):<25} {str(mat_arr):<25} {status:<10}")

# Compare h_le_dHmax
py_h_le = python_entry.get('h_le_dHmax')
mat_h_le = getattr(matlab_entry, 'h_le_dHmax', None)

if py_h_le is not None and mat_h_le is not None:
    py_arr = np.asarray(py_h_le)
    if isinstance(mat_h_le, np.ndarray):
        mat_arr = mat_h_le
    else:
        mat_arr = np.asarray(mat_h_le)
    
    match = np.array_equal(py_arr, mat_arr)
    status = "✓" if match else "✗"
    py_count = np.sum(py_arr)
    mat_count = np.sum(mat_arr)
    print(f"\n{'h_le_dHmax (count)':<30} {py_count:<25} {mat_count:<25} {status:<10}")
    print(f"  Python: {py_arr}")
    print(f"  MATLAB: {mat_arr}")

print(f"\n{'='*80}")
