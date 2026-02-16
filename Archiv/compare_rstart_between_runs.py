"""Compare Rstart values between Run 1 and Run 2 to understand the divergence"""
import pickle
import numpy as np

print("=" * 80)
print("COMPARING Rstart VALUES BETWEEN RUNS")
print("=" * 80)

with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']

# Find Step 2 Run 1 and Run 2
run1_entry = None
run2_entry = None

for entry in python_upstream:
    if isinstance(entry, dict):
        step = entry.get('step')
        run = entry.get('run')
        if step == 2 and run == 1:
            run1_entry = entry
        elif step == 2 and run == 2:
            run2_entry = entry

if run1_entry is None or run2_entry is None:
    print("[ERROR] Could not find both Run 1 and Run 2 entries for Step 2")
    exit(1)

print("[OK] Found Step 2 Run 1 and Run 2")

# Compare Rstart from initReach_tracking
print("\n" + "=" * 80)
print("COMPARING Rstart FROM initReach_tracking")
print("=" * 80)

if 'initReach_tracking' in run1_entry:
    r1_it = run1_entry['initReach_tracking']
    r1_rstart_center = np.asarray(r1_it.get('Rstart_center', [])).flatten()
    r1_rstart_gens = np.asarray(r1_it.get('Rstart_generators', []))
    r1_rstart_num = r1_it.get('Rstart_num_generators')
    r1_timeStep = r1_it.get('timeStep')
    
    print(f"\nRun 1:")
    print(f"  Rstart center: {r1_rstart_center}")
    print(f"  Rstart norm: {np.linalg.norm(r1_rstart_center):.6f}")
    print(f"  Rstart generators shape: {r1_rstart_gens.shape}")
    print(f"  Rstart num generators: {r1_rstart_num}")
    print(f"  timeStep: {r1_timeStep}")
else:
    print("[ERROR] No initReach_tracking in Run 1")
    exit(1)

if 'initReach_tracking' in run2_entry:
    r2_it = run2_entry['initReach_tracking']
    r2_rstart_center = np.asarray(r2_it.get('Rstart_center', [])).flatten()
    r2_rstart_gens = np.asarray(r2_it.get('Rstart_generators', []))
    r2_rstart_num = r2_it.get('Rstart_num_generators')
    r2_timeStep = r2_it.get('timeStep')
    
    print(f"\nRun 2:")
    print(f"  Rstart center: {r2_rstart_center}")
    print(f"  Rstart norm: {np.linalg.norm(r2_rstart_center):.6f}")
    print(f"  Rstart generators shape: {r2_rstart_gens.shape}")
    print(f"  Rstart num generators: {r2_rstart_num}")
    print(f"  timeStep: {r2_timeStep}")
else:
    print("[ERROR] No initReach_tracking in Run 2")
    exit(1)

# Compare
print(f"\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

center_diff = np.abs(r1_rstart_center - r2_rstart_center)
center_diff_pct = (center_diff / np.abs(r1_rstart_center)) * 100 if np.any(r1_rstart_center != 0) else np.zeros_like(center_diff)

print(f"\nRstart center difference:")
print(f"  Absolute: {center_diff}")
print(f"  Percentage: {center_diff_pct}%")
print(f"  Max absolute diff: {np.max(center_diff):.6f}")
print(f"  Max percentage diff: {np.max(center_diff_pct):.1f}%")

if r1_rstart_gens.shape == r2_rstart_gens.shape:
    gens_diff = np.abs(r1_rstart_gens - r2_rstart_gens)
    print(f"\nRstart generators difference:")
    print(f"  Max absolute diff: {np.max(gens_diff):.6f}")
    print(f"  Mean absolute diff: {np.mean(gens_diff):.6f}")
    if np.max(gens_diff) < 1e-10:
        print(f"  [MATCH] Generators are identical")
    else:
        print(f"  [DIFFERENT] Generators differ")
        # Show column-wise differences
        col_diffs = np.sum(gens_diff, axis=0)
        print(f"  Column (generator) differences: {col_diffs}")
else:
    print(f"\nRstart generators: [SHAPE MISMATCH]")
    print(f"  Run 1 shape: {r1_rstart_gens.shape}")
    print(f"  Run 2 shape: {r2_rstart_gens.shape}")

timeStep_diff = abs(r1_timeStep - r2_timeStep)
timeStep_diff_pct = (timeStep_diff / r1_timeStep) * 100 if r1_timeStep > 0 else 0
print(f"\ntimeStep difference:")
print(f"  Absolute: {timeStep_diff:.6f}")
print(f"  Percentage: {timeStep_diff_pct:.1f}%")

# Check what Rstart should be (from previous step's reduced Rtp)
print("\n" + "=" * 80)
print("CHECKING WHAT Rstart SHOULD BE")
print("=" * 80)
print("Rstart for Step 2 should be the reduced Rtp from Step 1")

# Find Step 1 Run 1 and Run 2
step1_run1 = None
step1_run2 = None

for entry in python_upstream:
    if isinstance(entry, dict):
        step = entry.get('step')
        run = entry.get('run')
        if step == 1 and run == 1:
            step1_run1 = entry
        elif step == 1 and run == 2:
            step1_run2 = entry

if step1_run1 and 'Rtp_after_reduction' in step1_run1:
    s1r1_rtp = step1_run1['Rtp_after_reduction']
    s1r1_rtp_center = np.asarray(s1r1_rtp.get('center', [])).flatten()
    s1r1_rtp_num = s1r1_rtp.get('num_generators')
    
    print(f"\nStep 1 Run 1 reduced Rtp (should be Rstart for Step 2 Run 1):")
    print(f"  Center: {s1r1_rtp_center}")
    print(f"  Norm: {np.linalg.norm(s1r1_rtp_center):.6f}")
    print(f"  Num generators: {s1r1_rtp_num}")
    
    # Compare with Step 2 Run 1 Rstart
    if len(s1r1_rtp_center) == len(r1_rstart_center):
        diff = np.abs(s1r1_rtp_center - r1_rstart_center)
        print(f"  Difference from Step 2 Run 1 Rstart: {diff}")
        print(f"  Max diff: {np.max(diff):.6f}")
        if np.max(diff) < 1e-10:
            print(f"  [MATCH] Step 2 Run 1 Rstart matches Step 1 Run 1 reduced Rtp")
        else:
            print(f"  [MISMATCH] Step 2 Run 1 Rstart differs from Step 1 Run 1 reduced Rtp")

if step1_run2 and 'Rtp_after_reduction' in step1_run2:
    s1r2_rtp = step1_run2['Rtp_after_reduction']
    s1r2_rtp_center = np.asarray(s1r2_rtp.get('center', [])).flatten()
    s1r2_rtp_num = s1r2_rtp.get('num_generators')
    
    print(f"\nStep 1 Run 2 reduced Rtp (should be Rstart for Step 2 Run 2):")
    print(f"  Center: {s1r2_rtp_center}")
    print(f"  Norm: {np.linalg.norm(s1r2_rtp_center):.6f}")
    print(f"  Num generators: {s1r2_rtp_num}")
    
    # Compare with Step 2 Run 2 Rstart
    if len(s1r2_rtp_center) == len(r2_rstart_center):
        diff = np.abs(s1r2_rtp_center - r2_rstart_center)
        print(f"  Difference from Step 2 Run 2 Rstart: {diff}")
        print(f"  Max diff: {np.max(diff):.6f}")
        if np.max(diff) < 1e-10:
            print(f"  [MATCH] Step 2 Run 2 Rstart matches Step 1 Run 2 reduced Rtp")
        else:
            print(f"  [MISMATCH] Step 2 Run 2 Rstart differs from Step 1 Run 2 reduced Rtp")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("This comparison shows:")
print("1. Whether Rstart differs between Run 1 and Run 2")
print("2. Whether Rstart matches the previous step's reduced Rtp")
print("3. How the difference propagates to timeStep selection")
