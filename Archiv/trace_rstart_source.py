"""Trace where Rstart comes from for Step 2 Run 2"""
import pickle
import numpy as np

print("=" * 80)
print("TRACING Rstart SOURCE FOR STEP 2 RUN 2")
print("=" * 80)

with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']

# Find Step 2 Run 2
s2r2 = None
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 2 and entry.get('run') == 2:
        s2r2 = entry
        break

if not s2r2 or 'initReach_tracking' not in s2r2:
    print("[ERROR] Could not find Step 2 Run 2 with initReach_tracking")
    exit(1)

s2r2_rstart_center = np.asarray(s2r2['initReach_tracking'].get('Rstart_center', [])).flatten()
s2r2_rstart_norm = np.linalg.norm(s2r2_rstart_center)
print(f"Step 2 Run 2 Rstart:")
print(f"  Center: {s2r2_rstart_center}")
print(f"  Norm: {s2r2_rstart_norm:.6f}")

# Check Step 1's Rtp_after_reduction
print("\n" + "=" * 80)
print("CHECKING STEP 1 REDUCED Rtp VALUES")
print("=" * 80)

step1_entries = []
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 1:
        step1_entries.append(entry)

print(f"Found {len(step1_entries)} Step 1 entries")

for entry in step1_entries:
    run = entry.get('run')
    print(f"\nStep 1 Run {run}:")
    
    if 'Rtp_after_reduction' in entry:
        rtp = entry['Rtp_after_reduction']
        rtp_center = np.asarray(rtp.get('center', [])).flatten()
        rtp_norm = np.linalg.norm(rtp_center)
        rtp_num = rtp.get('num_generators')
        
        print(f"  Rtp_after_reduction:")
        print(f"    Center: {rtp_center}")
        print(f"    Norm: {rtp_norm:.6f}")
        print(f"    Num generators: {rtp_num}")
        
        # Compare with Step 2 Run 2 Rstart
        if len(rtp_center) == len(s2r2_rstart_center):
            diff = np.abs(rtp_center - s2r2_rstart_center)
            print(f"    Difference from Step 2 Run 2 Rstart: {diff}")
            print(f"    Max diff: {np.max(diff):.6f}")
            if np.max(diff) < 1e-10:
                print(f"    [MATCH] This is the source of Step 2 Run 2 Rstart!")
            else:
                print(f"    [MISMATCH]")
    else:
        print(f"  No Rtp_after_reduction tracking")
    
    # Also check Rtp_before_reduction
    if 'Rtp_before_reduction' in entry:
        rtp = entry['Rtp_before_reduction']
        rtp_center = np.asarray(rtp.get('center', [])).flatten()
        rtp_norm = np.linalg.norm(rtp_center)
        rtp_num = rtp.get('num_generators')
        
        print(f"  Rtp_before_reduction:")
        print(f"    Center: {rtp_center}")
        print(f"    Norm: {rtp_norm:.6f}")
        print(f"    Num generators: {rtp_num}")

# Check if Step 2 Run 2 uses timeStepequalHorizon
print("\n" + "=" * 80)
print("CHECKING timeStepequalHorizon PATH")
print("=" * 80)

if 'timeStepequalHorizon_used' in s2r2:
    print(f"Step 2 Run 2 timeStepequalHorizon_used: {s2r2['timeStepequalHorizon_used']}")
    if s2r2['timeStepequalHorizon_used']:
        print("  [USES timeStepequalHorizon] - Should reuse Step 1 Run 1 results")
        if 'Rtp_h_tracking' in s2r2:
            rtp_h = s2r2['Rtp_h_tracking']
            print(f"  Rtp_h num_generators: {rtp_h.get('num_generators', 'N/A')}")
else:
    print("Step 2 Run 2: No timeStepequalHorizon_used flag")
    print("  [DOES NOT use timeStepequalHorizon] - Calls initReach_adaptive")

# Check Step 1 Run 2's initReach_tracking to see what Rstart it used
print("\n" + "=" * 80)
print("STEP 1 RUN 2 - WHAT Rstart DID IT USE?")
print("=" * 80)

s1r2 = None
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 1 and entry.get('run') == 2:
        s1r2 = entry
        break

if s1r2 and 'initReach_tracking' in s1r2:
    s1r2_rstart_center = np.asarray(s1r2['initReach_tracking'].get('Rstart_center', [])).flatten()
    s1r2_rstart_norm = np.linalg.norm(s1r2_rstart_center)
    s1r2_timeStep = s1r2['initReach_tracking'].get('timeStep')
    
    print(f"Step 1 Run 2 Rstart:")
    print(f"  Center: {s1r2_rstart_center}")
    print(f"  Norm: {s1r2_rstart_norm:.6f}")
    print(f"  timeStep: {s1r2_timeStep}")
    
    # Compare with Step 2 Run 2
    if len(s1r2_rstart_center) == len(s2r2_rstart_center):
        diff = np.abs(s1r2_rstart_center - s2r2_rstart_center)
        print(f"  Difference from Step 2 Run 2 Rstart: {diff}")
        print(f"  Max diff: {np.max(diff):.6f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("This trace shows:")
print("1. What Step 2 Run 2's Rstart value is")
print("2. Which Step 1 reduced Rtp it should match")
print("3. Whether timeStepequalHorizon path is used")
print("4. How Rstart values differ between Step 1 Run 2 and Step 2 Run 2")
