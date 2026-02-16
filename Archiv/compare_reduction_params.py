"""
Compare reduction parameters between Python and MATLAB to find divergence.
"""
import pickle
import scipy.io
import numpy as np

print("=" * 80)
print("COMPARING REDUCTION PARAMETERS")
print("=" * 80)

# Load Python tracking data
try:
    with open('upstream_python_log.pkl', 'rb') as f:
        python_log = pickle.load(f)
    print("[OK] Loaded Python log")
except Exception as e:
    print(f"[ERROR] Error loading Python log: {e}")
    python_log = None

# Load MATLAB tracking data
try:
    matlab_data = scipy.io.loadmat('upstream_matlab_log.mat', struct_as_record=False, squeeze_me=True)
    matlab_log = matlab_data['upstreamLog']
    print("[OK] Loaded MATLAB log")
except Exception as e:
    print(f"[ERROR] Error loading MATLAB log: {e}")
    matlab_log = None

if python_log is None or matlab_log is None:
    print("Cannot proceed without both logs")
    exit(1)

# Find Step 2 Run 2 initReach_tracking entries (where Rlintp divergence occurs)
# Note: Step 2 Run 2 may use timeStepequalHorizon path, but Python still tracks initReach
print("\n" + "=" * 80)
print("STEP 2 RUN 2 - REDUCTION PARAMETERS")
print("=" * 80)

python_entry = None
matlab_entry = None

# Python log is a dict with 'upstreamLog' key containing a list
if isinstance(python_log, dict) and 'upstreamLog' in python_log:
    python_upstream = python_log['upstreamLog']
    for entry in python_upstream:
        if hasattr(entry, 'step') and hasattr(entry, 'run'):
            if entry.step == 2 and entry.run == 2:
                if hasattr(entry, 'initReach_tracking'):
                    python_entry = entry.initReach_tracking
                    break
        elif isinstance(entry, dict):
            if entry.get('step') == 2 and entry.get('run') == 2:
                if 'initReach_tracking' in entry:
                    python_entry = entry['initReach_tracking']
                    break

# MATLAB log is a numpy array of mat_struct objects
if isinstance(matlab_log, np.ndarray):
    for entry in matlab_log:
        if hasattr(entry, 'step') and hasattr(entry, 'run'):
            if entry.step == 2 and entry.run == 2:
                if hasattr(entry, 'initReach_tracking'):
                    it = entry.initReach_tracking
                    if isinstance(it, np.ndarray) and it.size > 0:
                        matlab_entry = it[0]
                        break
                    elif not isinstance(it, np.ndarray):
                        matlab_entry = it
                        break

if python_entry is None:
    print("[ERROR] Could not find Python Step 2 Run 2 initReach_tracking")
else:
    print("[OK] Found Python entry")

if matlab_entry is None:
    print("[ERROR] Could not find MATLAB Step 2 Run 2 initReach_tracking")
    print("Note: Step 2 Run 2 uses timeStepequalHorizon path, so initReach_adaptive")
    print("may not be called. Checking if we can use Step 1 Run 1 data instead...")
    # Try Step 1 Run 1 as fallback
    for entry in matlab_log:
        if hasattr(entry, 'step') and hasattr(entry, 'run'):
            if entry.step == 1 and entry.run == 1:
                if hasattr(entry, 'initReach_tracking'):
                    it = entry.initReach_tracking
                    if isinstance(it, np.ndarray) and it.size > 0:
                        matlab_entry = it[0]
                        print("[OK] Using Step 1 Run 1 data (used by Step 2 Run 2)")
                        break
                    elif not isinstance(it, np.ndarray):
                        matlab_entry = it
                        print("[OK] Using Step 1 Run 1 data (used by Step 2 Run 2)")
                        break
else:
    print("[OK] Found MATLAB entry")

if python_entry is None or matlab_entry is None:
    print("Cannot compare - missing entries")
    exit(1)

# Compare reduction parameters from initReach_tracking
print("\n" + "-" * 80)
print("REDUCTION PARAMETERS COMPARISON")
print("=" * 80)

# Get initReach_tracking from both
py_it = None
if isinstance(python_entry, dict):
    py_it = python_entry.get('initReach_tracking')
else:
    py_it = getattr(python_entry, 'initReach_tracking', None)

mat_it = None
if hasattr(matlab_entry, 'initReach_tracking'):
    it = matlab_entry.initReach_tracking
    if isinstance(it, np.ndarray) and it.size > 0:
        mat_it = it[0]
    elif not isinstance(it, np.ndarray):
        mat_it = it

if py_it is not None and mat_it is not None:
    
    # Compare key parameters (stored with 'reduction_' prefix in initReach_tracking)
    params_to_compare = ['reduction_dHmax', 'reduction_h_computed', 'reduction_redIdx', 
                         'reduction_gredIdx', 'reduction_nrG', 'reduction_last0Idx']
    
    for param in params_to_compare:
        # Handle dict access for Python
        if isinstance(py_it, dict):
            py_val = py_it.get(param)
        else:
            py_val = getattr(py_it, param, None)
        
        # Handle MATLAB struct access
        if hasattr(mat_it, param):
            mat_val = getattr(mat_it, param)
        else:
            mat_val = None
        
        if py_val is None or mat_val is None:
            print(f"\n{param}: Missing in one or both (Python: {py_val is not None}, MATLAB: {mat_val is not None})")
            continue
        
        # Convert to numpy arrays for comparison
        if isinstance(py_val, (list, np.ndarray)):
            py_val = np.asarray(py_val)
        if isinstance(mat_val, (list, np.ndarray)):
            mat_val = np.asarray(mat_val)
        
        print(f"\n{param}:")
        print(f"  Python: {py_val}")
        print(f"  MATLAB: {mat_val}")
        
        if isinstance(py_val, np.ndarray) and isinstance(mat_val, np.ndarray):
            if py_val.shape == mat_val.shape:
                if np.allclose(py_val, mat_val, rtol=1e-10, atol=1e-12):
                    print(f"  [MATCH]")
                else:
                    print(f"  [MISMATCH]")
                    print(f"  Difference: {np.abs(py_val - mat_val)}")
            else:
                print(f"  [SHAPE MISMATCH]: Python {py_val.shape} vs MATLAB {mat_val.shape}")
        elif py_val == mat_val:
            print(f"  [MATCH]")
        else:
            print(f"  [MISMATCH]")
else:
    print("\n[ERROR] Cannot compare - missing initReach_tracking in one or both")
    if py_it is None:
        print("  Python initReach_tracking: MISSING")
    else:
        print("  Python initReach_tracking: FOUND")
    if mat_it is None:
        print("  MATLAB initReach_tracking: MISSING")
    else:
        print("  MATLAB initReach_tracking: FOUND")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
