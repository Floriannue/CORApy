"""Analyze time step pattern across runs to understand the divergence"""
import pickle
import numpy as np

print("=" * 80)
print("ANALYZING TIME STEP PATTERN")
print("=" * 80)

with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']

# Collect all Run 1 and Run 2 entries with initReach_tracking
run1_data = []
run2_data = []

for entry in python_upstream:
    if isinstance(entry, dict) and 'initReach_tracking' in entry:
        it = entry['initReach_tracking']
        step = entry.get('step')
        run = entry.get('run')
        timeStep = it.get('timeStep')
        rhom_tp_gens = it.get('Rhom_tp_num_generators')
        rend_tp_gens = it.get('Rend_tp_num_generators')
        rstart_center = np.asarray(it.get('Rstart_center', [])).flatten()
        
        if timeStep and rhom_tp_gens == 5:  # Focus on 5 generator inputs
            data = {
                'step': step,
                'run': run,
                'timeStep': timeStep,
                'rend_tp_gens': rend_tp_gens,
                'rstart_center': rstart_center,
                'rstart_norm': np.linalg.norm(rstart_center)
            }
            
            if run == 1:
                run1_data.append(data)
            elif run == 2:
                run2_data.append(data)

print(f"Found {len(run1_data)} Run 1 entries with 5 generators")
print(f"Found {len(run2_data)} Run 2 entries with 5 generators")

# Analyze pattern
print("\n" + "=" * 80)
print("RUN 1 PATTERN (matches MATLAB: 5->4 generators)")
print("=" * 80)
if run1_data:
    timeSteps1 = [d['timeStep'] for d in run1_data]
    rend_gens1 = [d['rend_tp_gens'] for d in run1_data]
    rstart_norms1 = [d['rstart_norm'] for d in run1_data]
    
    print(f"Time steps: min={min(timeSteps1):.6f}, max={max(timeSteps1):.6f}, mean={np.mean(timeSteps1):.6f}")
    print(f"Rend_tp generators: {set(rend_gens1)}")
    print(f"Rstart center norms: min={min(rstart_norms1):.6f}, max={max(rstart_norms1):.6f}, mean={np.mean(rstart_norms1):.6f}")
    
    # Show first few
    print("\nFirst 5 Run 1 entries:")
    for d in run1_data[:5]:
        print(f"  Step {d['step']}: timeStep={d['timeStep']:.6f}, Rstart_norm={d['rstart_norm']:.6f}, Rend_tp={d['rend_tp_gens']} gens")

print("\n" + "=" * 80)
print("RUN 2 PATTERN (differs from MATLAB: 5->2 generators)")
print("=" * 80)
if run2_data:
    timeSteps2 = [d['timeStep'] for d in run2_data]
    rend_gens2 = [d['rend_tp_gens'] for d in run2_data]
    rstart_norms2 = [d['rstart_norm'] for d in run2_data]
    
    print(f"Time steps: min={min(timeSteps2):.6f}, max={max(timeSteps2):.6f}, mean={np.mean(timeSteps2):.6f}")
    print(f"Rend_tp generators: {set(rend_gens2)}")
    print(f"Rstart center norms: min={min(rstart_norms2):.6f}, max={max(rstart_norms2):.6f}, mean={np.mean(rstart_norms2):.6f}")
    
    # Show first few
    print("\nFirst 5 Run 2 entries:")
    for d in run2_data[:5]:
        print(f"  Step {d['step']}: timeStep={d['timeStep']:.6f}, Rstart_norm={d['rstart_norm']:.6f}, Rend_tp={d['rend_tp_gens']} gens")

# Compare Run 1 vs Run 2 for same step
print("\n" + "=" * 80)
print("COMPARING RUN 1 vs RUN 2 FOR SAME STEPS")
print("=" * 80)

# Find steps that appear in both runs
steps_both = set([d['step'] for d in run1_data]) & set([d['step'] for d in run2_data])

if steps_both:
    print(f"Found {len(steps_both)} steps in both runs")
    print("\nComparison (first 10 steps):")
    for step in sorted(list(steps_both))[:10]:
        r1 = next((d for d in run1_data if d['step'] == step), None)
        r2 = next((d for d in run2_data if d['step'] == step), None)
        
        if r1 and r2:
            ts_diff = r2['timeStep'] - r1['timeStep']
            ts_diff_pct = (ts_diff / r1['timeStep']) * 100 if r1['timeStep'] > 0 else 0
            rstart_diff = r2['rstart_norm'] - r1['rstart_norm']
            
            print(f"\n  Step {step}:")
            print(f"    Run 1: timeStep={r1['timeStep']:.6f}, Rstart_norm={r1['rstart_norm']:.6f}, Rend_tp={r1['rend_tp_gens']} gens")
            print(f"    Run 2: timeStep={r2['timeStep']:.6f}, Rstart_norm={r2['rstart_norm']:.6f}, Rend_tp={r2['rend_tp_gens']} gens")
            print(f"    Diff:  timeStep={ts_diff:+.6f} ({ts_diff_pct:+.1f}%), Rstart_norm={rstart_diff:+.6f}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("This analysis shows the relationship between:")
print("  - Rstart center norm (input to optimaldeltat)")
print("  - Selected timeStep (output from optimaldeltat)")
print("  - Final Rend_tp generator count (after reduction)")
print("\nIf Rstart differs between Python Run 2 and MATLAB Run 2,")
print("this explains the time step divergence and subsequent reduction difference.")
