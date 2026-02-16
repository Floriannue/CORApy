"""Compare Rlintp generator counts between Python and MATLAB"""
import pickle
import scipy.io
import numpy as np

print("=" * 80)
print("COMPARING Rlintp GENERATOR COUNTS")
print("=" * 80)

# Python
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']

# Find Step 2 Run 2
py_entry = None
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 2 and entry.get('run') == 2:
        py_entry = entry
        break

if py_entry is None:
    print("[ERROR] Could not find Python Step 2 Run 2")
    exit(1)

print("[OK] Found Python Step 2 Run 2")

# MATLAB
matlab_data = scipy.io.loadmat('upstream_matlab_log.mat', struct_as_record=False, squeeze_me=True)
matlab_log = matlab_data['upstreamLog']

mat_entry = None
for entry in matlab_log:
    if hasattr(entry, 'step') and hasattr(entry, 'run'):
        if entry.step == 2 and entry.run == 2:
            mat_entry = entry
            break

if mat_entry is None:
    print("[ERROR] Could not find MATLAB Step 2 Run 2")
    exit(1)

print("[OK] Found MATLAB Step 2 Run 2")

# Compare Rlintp generator counts
print("\n" + "=" * 80)
print("RLINTP GENERATOR COUNT COMPARISON")
print("=" * 80)

# Python Rlintp
if 'Rlintp_tracking' in py_entry:
    py_rlintp = py_entry['Rlintp_tracking']
    py_rlintp_num = py_rlintp.get('num_generators')
    py_rlintp_center = np.asarray(py_rlintp.get('center', [])).flatten()
    py_rlintp_gens = np.asarray(py_rlintp.get('generators', []))
    
    print(f"\nPython Rlintp:")
    print(f"  Num generators: {py_rlintp_num}")
    print(f"  Center: {py_rlintp_center}")
    print(f"  Generators shape: {py_rlintp_gens.shape}")
else:
    print("\n[ERROR] No Rlintp_tracking in Python entry")
    py_rlintp_num = None

# MATLAB Rlintp
if hasattr(mat_entry, 'Rlintp_tracking'):
    mat_rlintp = mat_entry.Rlintp_tracking
    if isinstance(mat_rlintp, np.ndarray) and mat_rlintp.size > 0:
        mat_rlintp = mat_rlintp[0]
    mat_rlintp_num = getattr(mat_rlintp, 'num_generators', None)
    mat_rlintp_center = np.asarray(getattr(mat_rlintp, 'center', [])).flatten()
    mat_rlintp_gens = np.asarray(getattr(mat_rlintp, 'generators', []))
    
    print(f"\nMATLAB Rlintp:")
    print(f"  Num generators: {mat_rlintp_num}")
    print(f"  Center: {mat_rlintp_center}")
    print(f"  Generators shape: {mat_rlintp_gens.shape}")
else:
    print("\n[ERROR] No Rlintp_tracking in MATLAB entry")
    mat_rlintp_num = None

# Compare
if py_rlintp_num is not None and mat_rlintp_num is not None:
    print(f"\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"Python: {py_rlintp_num} generators")
    print(f"MATLAB: {mat_rlintp_num} generators")
    if py_rlintp_num == mat_rlintp_num:
        print(f"[MATCH]")
    else:
        print(f"[MISMATCH] - Difference: {abs(py_rlintp_num - mat_rlintp_num)} generators")
        print(f"\nThis is the divergence we need to fix!")

# Also check initReach_tracking for Rend_tp
print("\n" + "=" * 80)
print("REND_TP GENERATOR COUNT COMPARISON")
print("=" * 80)

if 'initReach_tracking' in py_entry:
    py_it = py_entry['initReach_tracking']
    py_rend_tp_num = py_it.get('Rend_tp_num_generators')
    py_rhom_tp_num = py_it.get('Rhom_tp_num_generators')
    
    print(f"\nPython initReach_tracking:")
    print(f"  Rhom_tp (before reduction): {py_rhom_tp_num} generators")
    print(f"  Rend_tp (after reduction): {py_rend_tp_num} generators")
    if py_rhom_tp_num and py_rend_tp_num:
        print(f"  Reduced: {py_rhom_tp_num - py_rend_tp_num} generators")

if hasattr(mat_entry, 'initReach_tracking'):
    mat_it = mat_entry.initReach_tracking
    if isinstance(mat_it, np.ndarray) and mat_it.size > 0:
        mat_it = mat_it[0]
    mat_rend_tp_num = getattr(mat_it, 'Rend_tp_num_generators', None)
    mat_rhom_tp_num = getattr(mat_it, 'Rhom_tp_num_generators', None)
    
    print(f"\nMATLAB initReach_tracking:")
    print(f"  Rhom_tp (before reduction): {mat_rhom_tp_num} generators")
    print(f"  Rend_tp (after reduction): {mat_rend_tp_num} generators")
    if mat_rhom_tp_num and mat_rend_tp_num:
        print(f"  Reduced: {mat_rhom_tp_num - mat_rend_tp_num} generators")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("Rlintp is computed from Rend.tp, so if Rend.tp has different generator counts,")
print("Rlintp will also have different generator counts.")
