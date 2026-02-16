"""check_upstream_tracking - Check what upstream tracking data is available"""

import pickle

with open('upstream_python_log.pkl', 'rb') as f:
    data = pickle.load(f)
    log = data['upstreamLog']

print(f"Total entries: {len(log)}")

# Check for entries with Z_before_quadmap
entries_with_z = [e for e in log if e.get('Z_before_quadmap')]
print(f"Entries with Z_before_quadmap: {len(entries_with_z)}")

# Check for entries with errorSec
entries_with_es = [e for e in log if e.get('errorSec_before_combine')]
print(f"Entries with errorSec_before_combine: {len(entries_with_es)}")

# Check for entries with errorLagr
entries_with_el = [e for e in log if e.get('errorLagr_before_combine')]
print(f"Entries with errorLagr_before_combine: {len(entries_with_el)}")

# Check for entries with VerrorDyn
entries_with_vd = [e for e in log if e.get('VerrorDyn_before_reduce')]
print(f"Entries with VerrorDyn_before_reduce: {len(entries_with_vd)}")

# Check steps and runs
if entries_with_z:
    steps = sorted(set(e.get('step', 0) for e in entries_with_z))
    runs = sorted(set(e.get('run', 0) for e in entries_with_z))
    print(f"\nSteps with Z tracking: {steps[:10]}... (total: {len(steps)})")
    print(f"Runs with Z tracking: {runs}")
    print(f"\nSample entry (step {entries_with_z[0].get('step')}, run {entries_with_z[0].get('run')}):")
    print(f"  Keys: {list(entries_with_z[0].keys())}")
    if entries_with_z[0].get('Z_before_quadmap'):
        z = entries_with_z[0]['Z_before_quadmap']
        print(f"  Z_before_quadmap keys: {list(z.keys()) if isinstance(z, dict) else 'Not a dict'}")
        if isinstance(z, dict) and 'radius_max' in z:
            print(f"  Z radius_max: {z['radius_max']}")
