"""Compare initReach_adaptive debug values from files."""
import pickle
import scipy.io
import numpy as np

# Load Python debug file
try:
    with open('initReach_debug.pkl', 'rb') as f:
        python_debug = pickle.load(f)
    # Handle case where it's a single dict instead of a list
    if isinstance(python_debug, dict):
        python_debug = [python_debug]
except FileNotFoundError:
    print("ERROR: initReach_debug.pkl not found")
    exit(1)
except Exception as e:
    print(f"ERROR loading Python debug file: {e}")
    exit(1)

# Load MATLAB debug file
try:
    matlab_data = scipy.io.loadmat('initReach_debug.mat', squeeze_me=True, struct_as_record=False)
    matlab_debug = matlab_data['debug_data']
    # Handle case where it's a single struct or array
    if isinstance(matlab_debug, np.ndarray):
        if matlab_debug.size == 1:
            matlab_debug = [matlab_debug.item()]
        else:
            matlab_debug = [matlab_debug[i].item() if isinstance(matlab_debug[i], np.ndarray) else matlab_debug[i] for i in range(matlab_debug.size)]
    elif not isinstance(matlab_debug, (list, tuple)):
        matlab_debug = [matlab_debug]
except FileNotFoundError:
    print("ERROR: initReach_debug.mat not found")
    exit(1)
except Exception as e:
    print(f"ERROR loading MATLAB debug file: {e}")
    exit(1)

print("=" * 80)
print("initReach_adaptive Debug Comparison")
print("=" * 80)
print()

# Find Step 1 Run 1 entries
py_step1_run1 = None
for entry in python_debug:
    if entry.get('step') == 1 and entry.get('run') == 1:
        py_step1_run1 = entry
        break

matlab_step1_run1 = None
for entry in matlab_debug:
    if isinstance(entry, np.ndarray):
        entry = entry.item()
    step = entry.step if hasattr(entry, 'step') else None
    run = entry.run if hasattr(entry, 'run') else None
    if isinstance(step, np.ndarray):
        step = step.item() if step.size == 1 else step
    if isinstance(run, np.ndarray):
        run = run.item() if run.size == 1 else run
    if step == 1 and run == 1:
        matlab_step1_run1 = entry
        break

if py_step1_run1 is None:
    print("ERROR: Python Step 1 Run 1 not found")
    print(f"Available Python entries: {[(e.get('step'), e.get('run')) for e in python_debug]}")
    exit(1)

if matlab_step1_run1 is None:
    print("ERROR: MATLAB Step 1 Run 1 not found")
    print(f"Available MATLAB entries: {[(getattr(e.item() if isinstance(e, np.ndarray) else e, 'step', None), getattr(e.item() if isinstance(e, np.ndarray) else e, 'run', None)) for e in matlab_debug]}")
    exit(1)

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

print("Step 1 Run 1 Comparison:")
print()

# Compare Rhom_tp (before reduction)
py_rhom_tp_gen = py_step1_run1.get('Rhom_tp_num_generators', 0)
matlab_rhom_tp_gen = get_ml_value(matlab_step1_run1, 'Rhom_tp_num_generators')
print(f"Rhom_tp (before reduction):")
print(f"  Python: {py_rhom_tp_gen} generators")
print(f"  MATLAB: {matlab_rhom_tp_gen} generators")
if py_rhom_tp_gen != matlab_rhom_tp_gen:
    print(f"  -> MISMATCH: diff = {matlab_rhom_tp_gen - py_rhom_tp_gen}")
    print(f"  -> THIS IS THE SOURCE OF THE DIVERGENCE!")
else:
    print(f"  -> MATCH")
print()

# Compare Rend.tp (after reduction)
py_rend_tp_gen = py_step1_run1.get('Rend_tp_num_generators', 0)
matlab_rend_tp_gen = get_ml_value(matlab_step1_run1, 'Rend_tp_num_generators')
print(f"Rend.tp (after reduction):")
print(f"  Python: {py_rend_tp_gen} generators")
print(f"  MATLAB: {matlab_rend_tp_gen} generators")
if py_rend_tp_gen != matlab_rend_tp_gen:
    print(f"  -> MISMATCH: diff = {matlab_rend_tp_gen - py_rend_tp_gen}")
    print(f"  -> Reduction removed {py_rhom_tp_gen - py_rend_tp_gen} generators in Python")
    print(f"  -> Reduction removed {matlab_rhom_tp_gen - matlab_rend_tp_gen} generators in MATLAB")
else:
    print(f"  -> MATCH")
print()

# Compare other components
print("Other components:")
py_rstart_gen = py_step1_run1.get('Rstart_num_generators', 0)
matlab_rstart_gen = get_ml_value(matlab_step1_run1, 'Rstart_num_generators')
print(f"  Rstart: Python {py_rstart_gen} vs MATLAB {matlab_rstart_gen}")

py_rtrans_gen = py_step1_run1.get('Rtrans_num_generators', 0)
matlab_rtrans_gen = get_ml_value(matlab_step1_run1, 'Rtrans_num_generators')
print(f"  Rtrans: Python {py_rtrans_gen} vs MATLAB {matlab_rtrans_gen}")

py_inputCorr_gen = py_step1_run1.get('inputCorr_num_generators', 0)
matlab_inputCorr_gen = get_ml_value(matlab_step1_run1, 'inputCorr_num_generators')
print(f"  inputCorr: Python {py_inputCorr_gen} vs MATLAB {matlab_inputCorr_gen}")

py_redFactor = py_step1_run1.get('redFactor')
matlab_redFactor = get_ml_value(matlab_step1_run1, 'redFactor')
print(f"  redFactor: Python {py_redFactor} vs MATLAB {matlab_redFactor}")
