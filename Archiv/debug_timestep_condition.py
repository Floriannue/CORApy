"""Debug why timeStepequalHorizon condition is not satisfied"""
import pickle
import numpy as np

print("=" * 80)
print("DEBUGGING timeStepequalHorizon CONDITION")
print("=" * 80)

with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']

# Find Step 2 entries
step2_entries = []
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 2:
        step2_entries.append(entry)

print(f"Found {len(step2_entries)} Step 2 entries")

# Check each run
for entry in step2_entries:
    run = entry.get('run')
    timeStepequalHorizon_used = entry.get('timeStepequalHorizon_used', False)
    print(f"\nStep 2 Run {run}:")
    print(f"  timeStepequalHorizon_used: {timeStepequalHorizon_used}")
    
    # Try to get timeStep and finitehorizon from tracking
    if 'initReach_tracking' in entry:
        it = entry['initReach_tracking']
        timeStep = it.get('timeStep')
        if timeStep:
            print(f"  timeStep: {timeStep}")

# Check Step 1 Run 1 to see what finitehorizon should be
step1_run1 = None
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 1 and entry.get('run') == 1:
        step1_run1 = entry
        break

if step1_run1:
    print(f"\nStep 1 Run 1:")
    if 'initReach_tracking' in step1_run1:
        it = step1_run1['initReach_tracking']
        timeStep = it.get('timeStep')
        if timeStep:
            print(f"  timeStep: {timeStep}")
            print(f"  This should be finitehorizon for Step 2")

print("\n" + "=" * 80)
print("The condition checks if timeStep == finitehorizon after Run 1.")
print("If they differ, timeStepequalHorizon remains False.")
print("Need to check the actual values during execution.")
