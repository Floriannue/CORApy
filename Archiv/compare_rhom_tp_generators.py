"""Compare Rhom_tp generator matrices between Run 1 and Run 2"""
import pickle
import numpy as np

print("=" * 80)
print("COMPARING Rhom_tp GENERATOR MATRICES")
print("=" * 80)

with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']

# Find Step 4 Run 1 and Run 2 (both have 5 generators input)
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

# Compare generator matrices
run1_gens = np.asarray(py_run1.get('Rhom_tp_generators'))
run2_gens = np.asarray(py_run2.get('Rhom_tp_generators'))

print(f"\nRun 1 Rhom_tp generators shape: {run1_gens.shape}")
print(f"Run 2 Rhom_tp generators shape: {run2_gens.shape}")

if run1_gens.shape == run2_gens.shape:
    diff = np.abs(run1_gens - run2_gens)
    max_diff = np.max(diff)
    print(f"\nGenerator matrix difference:")
    print(f"  Max absolute difference: {max_diff}")
    print(f"  Mean absolute difference: {np.mean(diff)}")
    if max_diff > 1e-10:
        print(f"  [DIFFERENT] Generator matrices differ!")
        print(f"  This explains why reduction produces different results")
        # Show which generators differ most
        col_diffs = np.sum(diff, axis=0)
        print(f"  Column (generator) differences: {col_diffs}")
    else:
        print(f"  [SAME] Generator matrices are identical")
else:
    print(f"  [SHAPE MISMATCH]")

# Compare centers
run1_center = np.asarray(py_run1.get('Rhom_tp_center')).flatten()
run2_center = np.asarray(py_run2.get('Rhom_tp_center')).flatten()

print(f"\nRun 1 Rhom_tp center: {run1_center}")
print(f"Run 2 Rhom_tp center: {run2_center}")
center_diff = np.abs(run1_center - run2_center)
print(f"Center difference: {center_diff}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("If the generator matrices differ between Run 1 and Run 2,")
print("this explains why Python Run 1 reduces to 4 (like MATLAB)")
print("but Python Run 2 reduces to 2 (different from MATLAB).")
print("\nThe root cause is likely in how Rhom_tp is computed,")
print("not in the reduction algorithm itself.")
