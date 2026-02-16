"""Compare optimaldeltat inputs and outputs between Python and MATLAB"""
import pickle
import scipy.io
import numpy as np

print("=" * 80)
print("COMPARING optimaldeltat INPUTS AND OUTPUTS")
print("=" * 80)

# Python
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']

# Find Step 2 Run 2 (where divergence occurs)
py_entry = None
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 2 and entry.get('run') == 2:
        py_entry = entry
        break

if py_entry is None:
    print("[ERROR] Could not find Python Step 2 Run 2")
    exit(1)

print("[OK] Found Python Step 2 Run 2")

# Check if optimaldeltat tracking exists
if 'optimaldeltatLog' in py_entry:
    opt_log = py_entry['optimaldeltatLog']
    print(f"[OK] Found {len(opt_log)} optimaldeltat entries")
    
    # Show the last entry (most recent)
    if len(opt_log) > 0:
        last_entry = opt_log[-1]
        print("\nLast optimaldeltat entry:")
        for key, val in last_entry.items():
            if isinstance(val, (list, np.ndarray)):
                arr = np.asarray(val)
                if arr.size <= 10:
                    print(f"  {key}: {val}")
                else:
                    print(f"  {key}: shape={arr.shape}, min={np.min(arr):.6e}, max={np.max(arr):.6e}")
            else:
                print(f"  {key}: {val}")
else:
    print("[INFO] No optimaldeltatLog found in Python entry")
    print("       This tracking needs to be enabled in linReach_adaptive")

# MATLAB - check if we have optimaldeltat tracking
matlab_data = scipy.io.loadmat('upstream_matlab_log.mat', struct_as_record=False, squeeze_me=True)
matlab_log = matlab_data['upstreamLog']

mat_entry = None
for entry in matlab_log:
    if hasattr(entry, 'step') and hasattr(entry, 'run'):
        if entry.step == 2 and entry.run == 2:
            mat_entry = entry
            break

if mat_entry is None:
    print("\n[INFO] Could not find MATLAB Step 2 Run 2")
    print("       Trying Step 1 Run 1 (used by Step 2 Run 2 via timeStepequalHorizon)...")
    for entry in matlab_log:
        if hasattr(entry, 'step') and hasattr(entry, 'run'):
            if entry.step == 1 and entry.run == 1:
                mat_entry = entry
                break

if mat_entry is None:
    print("[ERROR] Could not find MATLAB entry")
    exit(1)

print(f"\n[OK] Found MATLAB entry (Step {mat_entry.step}, Run {mat_entry.run})")

# Check MATLAB optimaldeltat tracking
if hasattr(mat_entry, 'optimaldeltatLog'):
    mat_opt_log = mat_entry.optimaldeltatLog
    if isinstance(mat_opt_log, np.ndarray) and mat_opt_log.size > 0:
        print(f"[OK] Found MATLAB optimaldeltatLog with {mat_opt_log.size} entries")
        if mat_opt_log.size > 0:
            last_entry = mat_opt_log[-1] if isinstance(mat_opt_log, np.ndarray) else mat_opt_log
            print("\nLast MATLAB optimaldeltat entry:")
            if hasattr(last_entry, '_fieldnames'):
                for field in last_entry._fieldnames:
                    val = getattr(last_entry, field, None)
                    if isinstance(val, (list, np.ndarray)):
                        arr = np.asarray(val)
                        if arr.size <= 10:
                            print(f"  {field}: {val}")
                        else:
                            print(f"  {field}: shape={arr.shape}, min={np.min(arr):.6e}, max={np.max(arr):.6e}")
                    else:
                        print(f"  {field}: {val}")
    else:
        print("[INFO] MATLAB optimaldeltatLog is empty")
else:
    print("[INFO] No optimaldeltatLog found in MATLAB entry")

# Compare initReach_tracking timeStep values
print("\n" + "=" * 80)
print("COMPARING TIME STEP VALUES")
print("=" * 80)

if 'initReach_tracking' in py_entry:
    py_timeStep = py_entry['initReach_tracking'].get('timeStep')
    print(f"Python Step 2 Run 2 timeStep: {py_timeStep}")
else:
    print("[ERROR] No initReach_tracking in Python entry")

if hasattr(mat_entry, 'initReach_tracking'):
    it = mat_entry.initReach_tracking
    if isinstance(it, np.ndarray) and it.size > 0:
        mat_timeStep = getattr(it[0], 'timeStep', None)
        print(f"MATLAB Step {mat_entry.step} Run {mat_entry.run} timeStep: {mat_timeStep}")
    elif not isinstance(it, np.ndarray):
        mat_timeStep = getattr(it, 'timeStep', None)
        print(f"MATLAB Step {mat_entry.step} Run {mat_entry.run} timeStep: {mat_timeStep}")
    else:
        print("[ERROR] MATLAB initReach_tracking is empty")
else:
    print("[ERROR] No initReach_tracking in MATLAB entry")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("To compare optimaldeltat inputs, we need to:")
print("1. Enable optimaldeltat tracking in both Python and MATLAB")
print("2. Compare Rstart, Rerror_h, finitehorizon, varphi, zetaP values")
print("3. Verify _aux_optimaldeltat produces same output for same inputs")
