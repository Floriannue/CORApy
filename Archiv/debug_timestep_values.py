"""Debug actual timeStep and finitehorizon values"""
import pickle
import numpy as np

print("=" * 80)
print("DEBUGGING timeStep vs finitehorizon VALUES")
print("=" * 80)

# We need to add debug output to linReach_adaptive to see the actual values
# For now, let's check what we can infer from the tracking data

with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']

# Find Step 2 Run 1 to see what timeStep was set to
step2_run1 = None
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 2 and entry.get('run') == 1:
        step2_run1 = entry
        break

if step2_run1:
    print("Step 2 Run 1:")
    if 'initReach_tracking' in step2_run1:
        it = step2_run1['initReach_tracking']
        timeStep = it.get('timeStep')
        if timeStep:
            print(f"  timeStep after Run 1: {timeStep}")

# Find Step 1 to see what finitehorizon should be
step1_entries = []
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 1:
        step1_entries.append(entry)

print(f"\nStep 1 entries: {len(step1_entries)}")
for entry in step1_entries:
    run = entry.get('run')
    if 'initReach_tracking' in entry:
        it = entry['initReach_tracking']
        timeStep = it.get('timeStep')
        if timeStep:
            print(f"  Step 1 Run {run}: timeStep = {timeStep}")

# The finitehorizon for Step 2 should be computed from Step 1's finitehorizon
# finitehorizon[2] = finitehorizon[1] * (1 + varphi[1] - zetaphi[minorder])
# But we don't have direct access to these values in the log

print("\n" + "=" * 80)
print("SOLUTION")
print("=" * 80)
print("The check needs to happen with a tolerance that accounts for")
print("the fact that _aux_optimaldeltat may return a value different from finitehorizon.")
print("We should check if the difference is small relative to finitehorizon itself.")
print("Or we need to check BEFORE timeStep is modified by _aux_optimaldeltat.")
