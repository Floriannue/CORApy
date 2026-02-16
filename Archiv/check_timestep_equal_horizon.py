"""Check if Python Step 2 Run 2 uses timeStepequalHorizon path"""
import pickle
import numpy as np

print("=" * 80)
print("CHECKING timeStepequalHorizon PATH USAGE")
print("=" * 80)

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

# Check if timeStepequalHorizon was used
timeStepequalHorizon_used = py_entry.get('timeStepequalHorizon_used', False)
print(f"\ntimeStepequalHorizon_used: {timeStepequalHorizon_used}")

if timeStepequalHorizon_used:
    print("[OK] Python Step 2 Run 2 uses timeStepequalHorizon path")
    if 'Rtp_h_tracking' in py_entry:
        rtp_h = py_entry['Rtp_h_tracking']
        print(f"  Rtp_h generators: {rtp_h.get('num_generators')}")
else:
    print("[WARNING] Python Step 2 Run 2 does NOT use timeStepequalHorizon path")
    print("  This means it calls initReach_adaptive directly")
    print("  This is different from MATLAB, which uses timeStepequalHorizon path")

# Check Step 1 Run 1 to see what finitehorizon was set
step1_run1 = None
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 1 and entry.get('run') == 1:
        step1_run1 = entry
        break

if step1_run1:
    print(f"\nStep 1 Run 1:")
    if 'initReach_tracking' in step1_run1:
        it = step1_run1['initReach_tracking']
        print(f"  Rend_tp generators: {it.get('Rend_tp_num_generators')}")
        print(f"  This is what should be reused in Step 2 Run 2")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
if not timeStepequalHorizon_used:
    print("Python Step 2 Run 2 should use timeStepequalHorizon path but doesn't.")
    print("This causes it to compute a different Rhom_tp, leading to different reduction results.")
    print("\nNeed to check why the condition 'timeStep == finitehorizon' is not satisfied.")
