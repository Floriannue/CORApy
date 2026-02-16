"""Compare Step 1's initReach_adaptive inputs and outputs between Python and MATLAB."""
import pickle
import scipy.io
import numpy as np

# Load Python log
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

# Load MATLAB log
matlab_log = scipy.io.loadmat('upstream_matlab_log.mat', squeeze_me=True, struct_as_record=False)
matlab_upstream_log = matlab_log['upstreamLog']

# Find Step 1 entries (use Run 1, as that's what becomes Rtp_h for Step 2)
python_step1 = None
for entry in python_log.get('upstreamLog', []):
    if entry.get('step') == 1 and entry.get('run') == 1:
        python_step1 = entry
        break

matlab_step1 = None
for entry in matlab_upstream_log:
    if hasattr(entry, 'step') and entry.step == 1:
        run = entry.run if hasattr(entry, 'run') else None
        if isinstance(run, np.ndarray):
            run = run.item() if run.size == 1 else run
        if run == 1:
            matlab_step1 = entry
            break

if python_step1 is None:
    print("ERROR: Could not find Python Step 1 Run 1 entry")
    exit(1)

if matlab_step1 is None:
    print("ERROR: Could not find MATLAB Step 1 Run 1 entry")
    exit(1)

print("=" * 80)
print("Step 1 Run 1: initReach_adaptive Comparison")
print("=" * 80)
print()

# Get initReach_tracking
py_initReach = python_step1.get('initReach_tracking')
matlab_initReach = matlab_step1.initReach_tracking if hasattr(matlab_step1, 'initReach_tracking') else None

if py_initReach is None:
    print("ERROR: Python initReach_tracking NOT FOUND")
    exit(1)

if matlab_initReach is None or (isinstance(matlab_initReach, np.ndarray) and matlab_initReach.size == 0):
    print("ERROR: MATLAB initReach_tracking NOT FOUND or empty")
    exit(1)

if isinstance(matlab_initReach, np.ndarray):
    matlab_initReach = matlab_initReach.item()

# Helper function to get MATLAB value
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

# Compare Rstart (input)
print("--- Rstart (input to initReach_adaptive) ---")
py_rstart_gen = py_initReach.get('Rstart_num_generators', 0)
matlab_rstart_gen = get_ml_value(matlab_initReach, 'Rstart_num_generators')
if matlab_rstart_gen is None:
    print("MATLAB: Rstart_num_generators NOT FOUND")
else:
    print(f"Python: {py_rstart_gen} generators")
    print(f"MATLAB: {matlab_rstart_gen} generators")
    if py_rstart_gen == matlab_rstart_gen:
        print(f"  -> MATCH: Both have {py_rstart_gen} generators")
    else:
        print(f"  -> MISMATCH: Python {py_rstart_gen} vs MATLAB {matlab_rstart_gen} (diff: {matlab_rstart_gen - py_rstart_gen})")
print()

# Compare Rhom_tp (before reduction)
print("--- Rhom_tp (before reduction) ---")
py_rhom_tp_gen = py_initReach.get('Rhom_tp_num_generators', 0)
matlab_rhom_tp_gen = get_ml_value(matlab_initReach, 'Rhom_tp_num_generators')
if matlab_rhom_tp_gen is None:
    print("MATLAB: Rhom_tp_num_generators NOT FOUND")
else:
    print(f"Python: {py_rhom_tp_gen} generators")
    print(f"MATLAB: {matlab_rhom_tp_gen} generators")
    if py_rhom_tp_gen == matlab_rhom_tp_gen:
        print(f"  -> MATCH: Both have {py_rhom_tp_gen} generators")
    else:
        print(f"  -> MISMATCH: Python {py_rhom_tp_gen} vs MATLAB {matlab_rhom_tp_gen} (diff: {matlab_rhom_tp_gen - py_rhom_tp_gen})")
print()

# Compare Rhom (before reduction)
print("--- Rhom (before reduction) ---")
py_rhom_gen = py_initReach.get('Rhom_num_generators', 0)
matlab_rhom_gen = get_ml_value(matlab_initReach, 'Rhom_num_generators')
if matlab_rhom_gen is None:
    print("MATLAB: Rhom_num_generators NOT FOUND")
else:
    print(f"Python: {py_rhom_gen} generators")
    print(f"MATLAB: {matlab_rhom_gen} generators")
    if py_rhom_gen == matlab_rhom_gen:
        print(f"  -> MATCH: Both have {py_rhom_gen} generators")
    else:
        print(f"  -> MISMATCH: Python {py_rhom_gen} vs MATLAB {matlab_rhom_gen} (diff: {matlab_rhom_gen - py_rhom_gen})")
print()

# Compare Rend.tp (output - becomes Rlintp for Step 2)
print("--- Rend.tp (output - becomes Rlintp for Step 2) ---")
py_rend_tp_gen = py_initReach.get('Rend_tp_num_generators', 0)
matlab_rend_tp_gen = get_ml_value(matlab_initReach, 'Rend_tp_num_generators')
if matlab_rend_tp_gen is None:
    print("MATLAB: Rend_tp_num_generators NOT FOUND")
else:
    print(f"Python: {py_rend_tp_gen} generators")
    print(f"MATLAB: {matlab_rend_tp_gen} generators")
    if py_rend_tp_gen == matlab_rend_tp_gen:
        print(f"  -> MATCH: Both have {py_rend_tp_gen} generators")
    else:
        print(f"  -> MISMATCH: Python {py_rend_tp_gen} vs MATLAB {matlab_rend_tp_gen} (diff: {matlab_rend_tp_gen - py_rend_tp_gen})")
print()

# Compare Rend.ti (output - becomes Rlinti for Step 2)
print("--- Rend.ti (output - becomes Rlinti for Step 2) ---")
py_rend_ti_gen = py_initReach.get('Rend_ti_num_generators', 0)
matlab_rend_ti_gen = get_ml_value(matlab_initReach, 'Rend_ti_num_generators')
if matlab_rend_ti_gen is None:
    print("MATLAB: Rend_ti_num_generators NOT FOUND")
else:
    print(f"Python: {py_rend_ti_gen} generators")
    print(f"MATLAB: {matlab_rend_ti_gen} generators")
    if py_rend_ti_gen == matlab_rend_ti_gen:
        print(f"  -> MATCH: Both have {py_rend_ti_gen} generators")
    else:
        print(f"  -> MISMATCH: Python {py_rend_ti_gen} vs MATLAB {matlab_rend_ti_gen} (diff: {matlab_rend_ti_gen - py_rend_ti_gen})")
print()

# Compare other components
print("--- Other Components ---")
py_inputCorr_gen = py_initReach.get('inputCorr_num_generators', 0)
matlab_inputCorr_gen = get_ml_value(matlab_initReach, 'inputCorr_num_generators')
if matlab_inputCorr_gen is not None:
    print(f"inputCorr: Python {py_inputCorr_gen} vs MATLAB {matlab_inputCorr_gen}")

py_rtrans_gen = py_initReach.get('Rtrans_num_generators', 0)
matlab_rtrans_gen = get_ml_value(matlab_initReach, 'Rtrans_num_generators')
if matlab_rtrans_gen is not None:
    print(f"Rtrans: Python {py_rtrans_gen} vs MATLAB {matlab_rtrans_gen}")

py_redFactor = py_initReach.get('redFactor')
matlab_redFactor = get_ml_value(matlab_initReach, 'redFactor')
if matlab_redFactor is not None:
    print(f"redFactor: Python {py_redFactor} vs MATLAB {matlab_redFactor}")

print()
print("=" * 80)
print("Summary:")
print("=" * 80)
print("The divergence chain:")
if py_rstart_gen != matlab_rstart_gen:
    print(f"  - Rstart: Python {py_rstart_gen} vs MATLAB {matlab_rstart_gen} (diff: {matlab_rstart_gen - py_rstart_gen})")
if py_rhom_tp_gen != matlab_rhom_tp_gen:
    print(f"  - Rhom_tp: Python {py_rhom_tp_gen} vs MATLAB {matlab_rhom_tp_gen} (diff: {matlab_rhom_tp_gen - py_rhom_tp_gen})")
if py_rhom_gen != matlab_rhom_gen:
    print(f"  - Rhom: Python {py_rhom_gen} vs MATLAB {matlab_rhom_gen} (diff: {matlab_rhom_gen - py_rhom_gen})")
if py_rend_tp_gen != matlab_rend_tp_gen:
    print(f"  - Rend.tp: Python {py_rend_tp_gen} vs MATLAB {matlab_rend_tp_gen} (diff: {matlab_rend_tp_gen - py_rend_tp_gen})")
    print(f"    -> This is the source of Step 2's Rlintp divergence!")
