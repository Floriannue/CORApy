"""check_quadmat_tracking - Check if quadMat tracking is working"""

import pickle
import os

python_file = 'upstream_python_log.pkl'
if os.path.exists(python_file):
    with open(python_file, 'rb') as f:
        data = pickle.load(f)
    log = data['upstreamLog']
    entries = [e for e in log if e.get('quadmat_tracking')]
    print(f'Entries with quadmat_tracking: {len(entries)}')
    if entries:
        e = entries[0]
        print(f'Step: {e.get("step")}')
        qm = e.get('quadmat_tracking')
        print(f'quadmat_tracking type: {type(qm)}')
        if qm:
            print(f'quadmat_tracking length: {len(qm)}')
            if len(qm) > 0:
                dim, info = qm[0]
                print(f'First entry: dim={dim}, info keys={list(info.keys())}')
                if 'dense_diag' in info:
                    print(f'  dense_diag: {info["dense_diag"]}')
                if 'after_convert_diag' in info:
                    print(f'  after_convert_diag: {info["after_convert_diag"]}')
    else:
        print('No entries with quadmat_tracking found')
else:
    print(f'File not found: {python_file}')
