"""Analyze the Rstart chain to understand how it's set between steps"""
import pickle
import numpy as np

print("=" * 80)
print("ANALYZING Rstart CHAIN")
print("=" * 80)

with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']

# Find entries for Step 1 and Step 2
entries = {}
for entry in python_upstream:
    if isinstance(entry, dict):
        step = entry.get('step')
        run = entry.get('run')
        if step in [1, 2] and run in [1, 2]:
            key = f"Step{step}_Run{run}"
            entries[key] = entry

print(f"Found entries: {list(entries.keys())}")

# Check what tracking data is available
print("\n" + "=" * 80)
print("AVAILABLE TRACKING DATA")
print("=" * 80)

for key, entry in sorted(entries.items()):
    print(f"\n{key}:")
    tracking_keys = [k for k in entry.keys() if 'tracking' in k.lower() or k in ['Rtp_after_reduction', 'Rtp_before_reduction']]
    if tracking_keys:
        for tk in tracking_keys:
            print(f"  - {tk}")
    else:
        print(f"  - No tracking data")

# Focus on Step 2 Run 2 (where we have initReach_tracking)
print("\n" + "=" * 80)
print("STEP 2 RUN 2 - DETAILED ANALYSIS")
print("=" * 80)

if 'Step2_Run2' in entries:
    s2r2 = entries['Step2_Run2']
    
    if 'initReach_tracking' in s2r2:
        it = s2r2['initReach_tracking']
        rstart_center = np.asarray(it.get('Rstart_center', [])).flatten()
        rstart_norm = np.linalg.norm(rstart_center)
        timeStep = it.get('timeStep')
        
        print(f"Rstart center: {rstart_center}")
        print(f"Rstart norm: {rstart_norm:.6f}")
        print(f"timeStep: {timeStep}")
        
        # Check what this should be (Step 1's reduced Rtp)
        if 'Step1_Run1' in entries:
            s1r1 = entries['Step1_Run1']
            if 'Rtp_after_reduction' in s1r1:
                s1r1_rtp = s1r1['Rtp_after_reduction']
                s1r1_rtp_center = np.asarray(s1r1_rtp.get('center', [])).flatten()
                s1r1_rtp_norm = np.linalg.norm(s1r1_rtp_center)
                
                print(f"\nStep 1 Run 1 reduced Rtp (expected Rstart for Step 2):")
                print(f"  Center: {s1r1_rtp_center}")
                print(f"  Norm: {s1r1_rtp_norm:.6f}")
                
                if len(s1r1_rtp_center) == len(rstart_center):
                    diff = np.abs(s1r1_rtp_center - rstart_center)
                    print(f"  Difference: {diff}")
                    print(f"  Max diff: {np.max(diff):.6f}")
                    if np.max(diff) < 1e-10:
                        print(f"  [MATCH]")
                    else:
                        print(f"  [MISMATCH] - This explains the divergence!")
        
        if 'Step1_Run2' in entries:
            s1r2 = entries['Step1_Run2']
            if 'Rtp_after_reduction' in s1r2:
                s1r2_rtp = s1r2['Rtp_after_reduction']
                s1r2_rtp_center = np.asarray(s1r2_rtp.get('center', [])).flatten()
                s1r2_rtp_norm = np.linalg.norm(s1r2_rtp_center)
                
                print(f"\nStep 1 Run 2 reduced Rtp:")
                print(f"  Center: {s1r2_rtp_center}")
                print(f"  Norm: {s1r2_rtp_norm:.6f}")
                
                if len(s1r2_rtp_center) == len(rstart_center):
                    diff = np.abs(s1r2_rtp_center - rstart_center)
                    print(f"  Difference from Step 2 Run 2 Rstart: {diff}")
                    print(f"  Max diff: {np.max(diff):.6f}")
                    if np.max(diff) < 1e-10:
                        print(f"  [MATCH] Step 2 Run 2 Rstart matches Step 1 Run 2 reduced Rtp")
                    else:
                        print(f"  [MISMATCH] Step 2 Run 2 Rstart differs from Step 1 Run 2 reduced Rtp")

# Check timeStepequalHorizon usage
print("\n" + "=" * 80)
print("timeStepequalHorizon USAGE")
print("=" * 80)

for key, entry in sorted(entries.items()):
    if 'timeStepequalHorizon_used' in entry:
        used = entry['timeStepequalHorizon_used']
        print(f"{key}: timeStepequalHorizon_used = {used}")
        if used and 'Rtp_h_tracking' in entry:
            rtp_h = entry['Rtp_h_tracking']
            print(f"  Rtp_h num_generators: {rtp_h.get('num_generators', 'N/A')}")
    else:
        print(f"{key}: No timeStepequalHorizon_used flag")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("This analysis shows:")
print("1. How Rstart is set from previous step's reduced Rtp")
print("2. Whether timeStepequalHorizon path is used")
print("3. Where the divergence in Rstart values originates")
