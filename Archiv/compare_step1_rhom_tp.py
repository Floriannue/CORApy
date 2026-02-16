"""Compare Step 1's Rhom_tp before reduction."""
import pickle
import scipy.io
import numpy as np

# Load Python log
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

# Load MATLAB log
matlab_log = scipy.io.loadmat('upstream_matlab_log.mat', squeeze_me=True, struct_as_record=False)
matlab_upstream_log = matlab_log['upstreamLog']

# Find Step 1 Run 2 entries (which have initReach_tracking)
python_step1 = None
for entry in python_log.get('upstreamLog', []):
    if entry.get('step') == 1 and entry.get('run') == 2:
        python_step1 = entry
        break

matlab_step1 = None
for entry in matlab_upstream_log:
    if hasattr(entry, 'step') and entry.step == 1:
        run = entry.run if hasattr(entry, 'run') else None
        if isinstance(run, np.ndarray):
            run = run.item() if run.size == 1 else run
        if run == 2:
            matlab_step1 = entry
            break

if python_step1 is None or matlab_step1 is None:
    print("ERROR: Could not find Step 1 Run 2 entries")
    exit(1)

# Get initReach_tracking
py_initReach = python_step1.get('initReach_tracking')
matlab_initReach = matlab_step1.initReach_tracking if hasattr(matlab_step1, 'initReach_tracking') else None

if py_initReach is None or matlab_initReach is None:
    print("ERROR: initReach_tracking not found")
    exit(1)

if isinstance(matlab_initReach, np.ndarray):
    if matlab_initReach.size == 0:
        print("ERROR: MATLAB initReach_tracking is empty")
        exit(1)
    if matlab_initReach.size == 1:
        matlab_initReach = matlab_initReach.item()

def get_ml_value(ml_obj, field):
    if not hasattr(ml_obj, field):
        return None
    val = getattr(ml_obj, field)
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return None
        if val.size == 1:
            return val.item()
        return val
    return val

print("=" * 80)
print("Step 1 Run 2: Rhom_tp and Rend.tp Comparison")
print("=" * 80)
print()

# Compare Rhom_tp (before reduction)
py_rhom_tp_gen = py_initReach.get('Rhom_tp_num_generators', 0)
matlab_rhom_tp_gen = get_ml_value(matlab_initReach, 'Rhom_tp_num_generators')
print(f"Rhom_tp (before reduction):")
print(f"  Python: {py_rhom_tp_gen} generators")
print(f"  MATLAB: {matlab_rhom_tp_gen} generators")
if py_rhom_tp_gen != matlab_rhom_tp_gen:
    print(f"  -> MISMATCH: diff = {matlab_rhom_tp_gen - py_rhom_tp_gen}")
else:
    print(f"  -> MATCH")
print()

# Compare Rend.tp (after reduction)
py_rend_tp_gen = py_initReach.get('Rend_tp_num_generators', 0)
matlab_rend_tp_gen = get_ml_value(matlab_initReach, 'Rend_tp_num_generators')
print(f"Rend.tp (after reduction):")
print(f"  Python: {py_rend_tp_gen} generators")
print(f"  MATLAB: {matlab_rend_tp_gen} generators")
if py_rend_tp_gen != matlab_rend_tp_gen:
    print(f"  -> MISMATCH: diff = {matlab_rend_tp_gen - py_rend_tp_gen}")
    print(f"  -> Reduction removed {py_rhom_tp_gen - py_rend_tp_gen} generators in Python")
    print(f"  -> Reduction removed {matlab_rhom_tp_gen - matlab_rend_tp_gen} generators in MATLAB")
else:
    print(f"  -> MATCH")
