"""Compare Python Run 1 vs Run 2 to find what differs"""
import pickle
import numpy as np

print("=" * 80)
print("COMPARING PYTHON RUN 1 vs RUN 2")
print("Run 1: 5->4 generators (matches MATLAB)")
print("Run 2: 5->2 generators (differs from MATLAB)")
print("=" * 80)

with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']

# Find Step 4 Run 1 and Run 2 (both have 5 generators input, different outputs)
py_run1 = None
py_run2 = None

for entry in python_upstream:
    if isinstance(entry, dict):
        step = entry.get('step')
        run = entry.get('run')
        if step == 4 and run == 1 and 'initReach_tracking' in entry:
            it = entry['initReach_tracking']
            if it.get('Rhom_tp_num_generators') == 5:
                py_run1 = it
        elif step == 4 and run == 2 and 'initReach_tracking' in entry:
            it = entry['initReach_tracking']
            if it.get('Rhom_tp_num_generators') == 5:
                py_run2 = it

if py_run1 is None or py_run2 is None:
    print("[ERROR] Could not find both Run 1 and Run 2 entries")
    exit(1)

print("[OK] Found Step 4 Run 1 and Run 2")

# Compare all tracked values
print("\n" + "=" * 80)
print("COMPARING ALL TRACKED VALUES")
print("=" * 80)

keys_to_compare = [
    'Rstart_center', 'Rstart_generators', 'Rstart_num_generators',
    'eAt', 'F', 'inputCorr_center', 'inputCorr_generators', 'inputCorr_num_generators',
    'Rtrans_center', 'Rtrans_generators', 'Rtrans_num_generators',
    'Rhom_tp_center', 'Rhom_tp_generators', 'Rhom_tp_num_generators',
    'Rhom_center', 'Rhom_generators', 'Rhom_num_generators',
    'redFactor', 'timeStep'
]

for key in keys_to_compare:
    val1 = py_run1.get(key)
    val2 = py_run2.get(key)
    
    if val1 is None or val2 is None:
        print(f"\n{key}:")
        print(f"  Run 1: {val1}")
        print(f"  Run 2: {val2}")
        continue
    
    # Convert to numpy arrays for comparison
    # Handle special types
    if hasattr(val1, '__class__') and 'Interval' in str(type(val1)):
        print(f"\n{key}: [SKIP] Interval type - cannot compare directly")
        continue
    
    try:
        arr1 = np.asarray(val1)
        arr2 = np.asarray(val2)
    except (TypeError, ValueError):
        print(f"\n{key}: [SKIP] Cannot convert to array")
        continue
    
    if arr1.shape != arr2.shape:
        print(f"\n{key}: [SHAPE MISMATCH]")
        print(f"  Run 1 shape: {arr1.shape}")
        print(f"  Run 2 shape: {arr2.shape}")
    else:
        diff = np.abs(arr1 - arr2)
        max_diff = np.max(diff)
        if max_diff < 1e-12:
            print(f"\n{key}: [MATCH] (max diff: {max_diff:.2e})")
        else:
            print(f"\n{key}: [DIFFERENT] (max diff: {max_diff:.2e})")
            if arr1.ndim <= 2 and arr1.size <= 20:
                print(f"  Run 1: {arr1}")
                print(f"  Run 2: {arr2}")
            elif arr1.ndim == 2:
                # Show column-wise differences for matrices
                col_diffs = np.sum(diff, axis=0)
                print(f"  Column differences: {col_diffs}")
                print(f"  Max column diff: {np.max(col_diffs):.2e}")

# Focus on the key computation: Rhom_tp = eAt * Rstart + Rtrans
print("\n" + "=" * 80)
print("ANALYZING Rhom_tp COMPUTATION")
print("Rhom_tp = eAt * Rstart + Rtrans")
print("=" * 80)

# Get components
eAt1 = np.asarray(py_run1.get('eAt'))
eAt2 = np.asarray(py_run2.get('eAt'))
rstart_center1 = np.asarray(py_run1.get('Rstart_center')).flatten()
rstart_center2 = np.asarray(py_run2.get('Rstart_center')).flatten()
rstart_gens1 = np.asarray(py_run1.get('Rstart_generators'))
rstart_gens2 = np.asarray(py_run2.get('Rstart_generators'))
rtrans_center1 = np.asarray(py_run1.get('Rtrans_center')).flatten()
rtrans_center2 = np.asarray(py_run2.get('Rtrans_center')).flatten()
rtrans_gens1 = np.asarray(py_run1.get('Rtrans_generators'))
rtrans_gens2 = np.asarray(py_run2.get('Rtrans_generators'))

# Compute expected Rhom_tp centers
rhom_tp_center1_expected = eAt1 @ rstart_center1 + rtrans_center1
rhom_tp_center2_expected = eAt2 @ rstart_center2 + rtrans_center2

rhom_tp_center1_actual = np.asarray(py_run1.get('Rhom_tp_center')).flatten()
rhom_tp_center2_actual = np.asarray(py_run2.get('Rhom_tp_center')).flatten()

print(f"\nRhom_tp center computation:")
print(f"  Run 1 expected: {rhom_tp_center1_expected}")
print(f"  Run 1 actual:   {rhom_tp_center1_actual}")
print(f"  Run 1 diff:     {np.abs(rhom_tp_center1_expected - rhom_tp_center1_actual)}")
print(f"  Run 2 expected: {rhom_tp_center2_expected}")
print(f"  Run 2 actual:   {rhom_tp_center2_actual}")
print(f"  Run 2 diff:     {np.abs(rhom_tp_center2_expected - rhom_tp_center2_actual)}")

# Compare generator computation
# Rhom_tp generators = eAt * Rstart_generators + Rtrans_generators
if rstart_gens1.shape == rstart_gens2.shape and rtrans_gens1.shape == rtrans_gens2.shape:
    rhom_tp_gens1_expected = eAt1 @ rstart_gens1
    rhom_tp_gens1_expected = np.hstack([rhom_tp_gens1_expected, rtrans_gens1])
    
    rhom_tp_gens2_expected = eAt2 @ rstart_gens2
    rhom_tp_gens2_expected = np.hstack([rhom_tp_gens2_expected, rtrans_gens2])
    
    rhom_tp_gens1_actual = np.asarray(py_run1.get('Rhom_tp_generators'))
    rhom_tp_gens2_actual = np.asarray(py_run2.get('Rhom_tp_generators'))
    
    print(f"\nRhom_tp generators computation:")
    print(f"  Run 1 expected shape: {rhom_tp_gens1_expected.shape}")
    print(f"  Run 1 actual shape:   {rhom_tp_gens1_actual.shape}")
    print(f"  Run 2 expected shape: {rhom_tp_gens2_expected.shape}")
    print(f"  Run 2 actual shape:   {rhom_tp_gens2_actual.shape}")
    
    if rhom_tp_gens1_expected.shape == rhom_tp_gens1_actual.shape:
        diff1 = np.abs(rhom_tp_gens1_expected - rhom_tp_gens1_actual)
        print(f"  Run 1 max diff: {np.max(diff1):.2e}")
    if rhom_tp_gens2_expected.shape == rhom_tp_gens2_actual.shape:
        diff2 = np.abs(rhom_tp_gens2_expected - rhom_tp_gens2_actual)
        print(f"  Run 2 max diff: {np.max(diff2):.2e}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("The first component that differs between Run 1 and Run 2")
print("is likely the root cause of the different reduction results.")
