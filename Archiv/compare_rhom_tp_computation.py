"""Compare Rhom_tp computation inputs and intermediates"""
import pickle
import scipy.io
import numpy as np

print("=" * 80)
print("COMPARING Rhom_tp COMPUTATION")
print("=" * 80)

# Python Step 2 Run 2
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']
py_entry = None
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 2 and entry.get('run') == 2:
        if 'initReach_tracking' in entry:
            py_entry = entry['initReach_tracking']
            break

# MATLAB - try Step 2 Run 2, then Step 1 Run 1 as fallback
matlab_data = scipy.io.loadmat('upstream_matlab_log.mat', struct_as_record=False, squeeze_me=True)
matlab_log = matlab_data['upstreamLog']

mat_entry = None
mat_source = None
# Try Step 2 Run 2 first
for entry in matlab_log:
    if hasattr(entry, 'step') and hasattr(entry, 'run'):
        if entry.step == 2 and entry.run == 2:
            if hasattr(entry, 'initReach_tracking'):
                it = entry.initReach_tracking
                if isinstance(it, np.ndarray) and it.size > 0:
                    mat_entry = it[0]
                    mat_source = "Step 2 Run 2"
                    break
                elif not isinstance(it, np.ndarray):
                    mat_entry = it
                    mat_source = "Step 2 Run 2"
                    break

# Fallback to Step 1 Run 1
if mat_entry is None:
    for entry in matlab_log:
        if hasattr(entry, 'step') and hasattr(entry, 'run'):
            if entry.step == 1 and entry.run == 1:
                if hasattr(entry, 'initReach_tracking'):
                    it = entry.initReach_tracking
                    if isinstance(it, np.ndarray) and it.size > 0:
                        mat_entry = it[0]
                        mat_source = "Step 1 Run 1 (used by Step 2 Run 2)"
                        break
                    elif not isinstance(it, np.ndarray):
                        mat_entry = it
                        mat_source = "Step 1 Run 1 (used by Step 2 Run 2)"
                        break

if py_entry is None:
    print("[ERROR] Could not find Python Step 2 Run 2")
    exit(1)

if mat_entry is None:
    print("[ERROR] Could not find MATLAB entry")
    exit(1)

print(f"[OK] Python Step 2 Run 2")
print(f"[OK] MATLAB {mat_source}")

# Compare inputs to Rhom_tp computation
# Rhom_tp = eAt * Rstart + Rtrans

print("\n" + "=" * 80)
print("COMPARING INPUTS TO Rhom_tp COMPUTATION")
print("=" * 80)

# Rstart
py_rstart_center = np.asarray(py_entry.get('Rstart_center', [])).flatten()
py_rstart_gens = np.asarray(py_entry.get('Rstart_generators', []))
mat_rstart_center = np.asarray(getattr(mat_entry, 'Rstart_center', [])).flatten()
mat_rstart_gens = np.asarray(getattr(mat_entry, 'Rstart_generators', []))

print(f"\nRstart center:")
print(f"  Python: {py_rstart_center}")
print(f"  MATLAB: {mat_rstart_center}")
if len(py_rstart_center) == len(mat_rstart_center):
    center_diff = np.abs(py_rstart_center - mat_rstart_center)
    print(f"  Difference: {center_diff}")
    print(f"  Max diff: {np.max(center_diff)}")
    if np.max(center_diff) < 1e-10:
        print(f"  [MATCH]")
    else:
        print(f"  [MISMATCH]")

print(f"\nRstart generators:")
print(f"  Python shape: {py_rstart_gens.shape}")
print(f"  MATLAB shape: {mat_rstart_gens.shape}")
if py_rstart_gens.shape == mat_rstart_gens.shape:
    gens_diff = np.abs(py_rstart_gens - mat_rstart_gens)
    print(f"  Max difference: {np.max(gens_diff)}")
    print(f"  Mean difference: {np.mean(gens_diff)}")
    if np.max(gens_diff) < 1e-10:
        print(f"  [MATCH]")
    else:
        print(f"  [MISMATCH]")
        # Show which generators differ most
        col_diffs = np.sum(gens_diff, axis=0)
        print(f"  Column differences: {col_diffs}")

# eAt
py_eAt = np.asarray(py_entry.get('eAt', []))
mat_eAt = np.asarray(getattr(mat_entry, 'eAt', []))

print(f"\neAt (exponential matrix):")
print(f"  Python shape: {py_eAt.shape}")
print(f"  MATLAB shape: {mat_eAt.shape}")
if py_eAt.shape == mat_eAt.shape:
    eAt_diff = np.abs(py_eAt - mat_eAt)
    print(f"  Max difference: {np.max(eAt_diff)}")
    print(f"  Mean difference: {np.mean(eAt_diff)}")
    if np.max(eAt_diff) < 1e-10:
        print(f"  [MATCH]")
    else:
        print(f"  [MISMATCH]")
        print(f"  eAt values:")
        print(f"    Python:\n{py_eAt}")
        print(f"    MATLAB:\n{mat_eAt}")

# Rtrans
py_rtrans_center = np.asarray(py_entry.get('Rtrans_center', [])).flatten()
py_rtrans_gens = np.asarray(py_entry.get('Rtrans_generators', []))
mat_rtrans_center = np.asarray(getattr(mat_entry, 'Rtrans_center', [])).flatten()
mat_rtrans_gens = np.asarray(getattr(mat_entry, 'Rtrans_generators', []))

print(f"\nRtrans center:")
print(f"  Python: {py_rtrans_center}")
print(f"  MATLAB: {mat_rtrans_center}")
if len(py_rtrans_center) == len(mat_rtrans_center):
    center_diff = np.abs(py_rtrans_center - mat_rtrans_center)
    print(f"  Difference: {center_diff}")
    if np.max(center_diff) < 1e-10:
        print(f"  [MATCH]")
    else:
        print(f"  [MISMATCH]")

print(f"\nRtrans generators:")
print(f"  Python shape: {py_rtrans_gens.shape}")
print(f"  MATLAB shape: {mat_rtrans_gens.shape}")
if py_rtrans_gens.shape == mat_rtrans_gens.shape:
    gens_diff = np.abs(py_rtrans_gens - mat_rtrans_gens)
    print(f"  Max difference: {np.max(gens_diff)}")
    if np.max(gens_diff) < 1e-10:
        print(f"  [MATCH]")
    else:
        print(f"  [MISMATCH]")

# Compare resulting Rhom_tp
py_rhom_tp_center = np.asarray(py_entry.get('Rhom_tp_center', [])).flatten()
py_rhom_tp_gens = np.asarray(py_entry.get('Rhom_tp_generators', []))
mat_rhom_tp_center = np.asarray(getattr(mat_entry, 'Rhom_tp_center', [])).flatten()
mat_rhom_tp_gens = np.asarray(getattr(mat_entry, 'Rhom_tp_generators', []))

print("\n" + "=" * 80)
print("COMPARING RESULTING Rhom_tp")
print("=" * 80)

print(f"\nRhom_tp center:")
print(f"  Python: {py_rhom_tp_center}")
print(f"  MATLAB: {mat_rhom_tp_center}")
if len(py_rhom_tp_center) == len(mat_rhom_tp_center):
    center_diff = np.abs(py_rhom_tp_center - mat_rhom_tp_center)
    print(f"  Difference: {center_diff}")
    if np.max(center_diff) < 1e-10:
        print(f"  [MATCH]")
    else:
        print(f"  [MISMATCH]")

print(f"\nRhom_tp generators:")
print(f"  Python shape: {py_rhom_tp_gens.shape}")
print(f"  MATLAB shape: {mat_rhom_tp_gens.shape}")
if py_rhom_tp_gens.shape == mat_rhom_tp_gens.shape:
    gens_diff = np.abs(py_rhom_tp_gens - mat_rhom_tp_gens)
    print(f"  Max difference: {np.max(gens_diff)}")
    print(f"  Mean difference: {np.mean(gens_diff)}")
    if np.max(gens_diff) < 1e-10:
        print(f"  [MATCH]")
    else:
        print(f"  [MISMATCH]")
        # Show which generators differ most
        col_diffs = np.sum(gens_diff, axis=0)
        print(f"  Column (generator) differences: {col_diffs}")
        print(f"  Generator values:")
        print(f"    Python:\n{py_rhom_tp_gens}")
        print(f"    MATLAB:\n{mat_rhom_tp_gens}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("This comparison shows where the divergence occurs in the computation chain.")
print("The first component that differs is likely the root cause.")
