"""Debug Python Step 2 entries to see what's in the log."""
import pickle

# Load Python log
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

upstream_log = python_log.get('upstreamLog', [])

# Find all Step 2 entries
step2_entries = []
for i, entry in enumerate(upstream_log):
    if entry.get('step') == 2:
        step2_entries.append((i, entry))

print(f"Found {len(step2_entries)} Step 2 entries")
print()

for idx, (i, entry) in enumerate(step2_entries):
    print(f"Step 2 Entry {idx+1} (index {i}):")
    print(f"  run: {entry.get('run')}")
    print(f"  Has Rlintp_tracking: {'Rlintp_tracking' in entry}")
    print(f"  Has Rerror_tracking: {'Rerror_tracking' in entry}")
    print(f"  Has Rtp_final_tracking: {'Rtp_final_tracking' in entry}")
    
    if 'Rlintp_tracking' in entry:
        rlintp = entry['Rlintp_tracking']
        if isinstance(rlintp, dict):
            print(f"  Rlintp_tracking.num_generators: {rlintp.get('num_generators')}")
        else:
            print(f"  Rlintp_tracking: {type(rlintp)}")
    
    if 'Rerror_tracking' in entry:
        rerror = entry['Rerror_tracking']
        if isinstance(rerror, dict):
            print(f"  Rerror_tracking.num_generators: {rerror.get('num_generators')}")
        else:
            print(f"  Rerror_tracking: {type(rerror)}")
    
    print()
