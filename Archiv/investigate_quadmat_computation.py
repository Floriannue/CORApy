"""investigate_quadmat_computation - Investigate exact quadMat computation differences"""

import numpy as np
import pickle
import os

print("=" * 80)
print("INVESTIGATING quadMat COMPUTATION")
print("=" * 80)

# Load Python log
python_file = 'upstream_python_log.pkl'
if os.path.exists(python_file):
    with open(python_file, 'rb') as f:
        python_data = pickle.load(f)
    python_upstream = python_data.get('upstreamLog', [])
    
    # Find entries with quadMat tracking
    entries_with_quadmat = [e for e in python_upstream if e.get('quadmat_tracking')]
    print(f"\n[OK] Found {len(entries_with_quadmat)} entries with quadMat tracking")
    
    if entries_with_quadmat:
        # Get Step 3 entry
        step3_entries = [e for e in entries_with_quadmat if e.get('step') == 3]
        if step3_entries:
            entry = step3_entries[-1]  # Last one (converged)
            print(f"\nStep 3 quadMat data:")
            
            quadmat_tracking = entry.get('quadmat_tracking')
            if quadmat_tracking:
                for dim, info in quadmat_tracking:
                    print(f"\n  Dimension {dim}:")
                    print(f"    Type: {info.get('type')}")
                    print(f"    Is interval: {info.get('is_interval')}")
                    print(f"    Is sparse: {info.get('is_sparse')}")
                    
                    if 'dense_diag' in info:
                        diag = info['dense_diag']
                        print(f"    Diagonal: {diag}")
                    
                    if 'dense_full' in info:
                        full = info['dense_full']
                        print(f"    Full matrix shape: {full.shape}")
                        print(f"    Full matrix:\n{full}")
                        print(f"    Full matrix max: {np.max(np.abs(full))}")
                        
                        # Check if matrix is symmetric
                        is_symmetric = np.allclose(full, full.T)
                        print(f"    Is symmetric: {is_symmetric}")
                        
                        # Check diagonal values
                        if full.shape[0] > 1:
                            main_diag = np.diag(full)
                            print(f"    Main diagonal: {main_diag}")
                            
                            # Check submatrix diagonal (what's actually used)
                            gens = len(info.get('dense_diag', []))
                            if gens > 0 and full.shape[0] > gens:
                                sub_diag = np.diag(full[1:gens+1, 1:gens+1])
                                print(f"    Submatrix diagonal (used): {sub_diag}")
                                print(f"    Expected from tracking: {info.get('dense_diag')}")
                                if not np.allclose(sub_diag, info.get('dense_diag', [])):
                                    print(f"    [WARNING] Diagonal mismatch!")
                    
                    # Check H values
                    H_before = entry.get('H_before_quadmap')
                    if H_before and len(H_before) > dim:
                        H_info = H_before[dim]
                        if isinstance(H_info, dict) and 'matrix' in H_info:
                            H_mat = H_info['matrix']
                            print(f"\n    H[{dim}] matrix:")
                            print(f"      Shape: {H_mat.shape}")
                            print(f"      Matrix:\n{H_mat}")
                            
                            # Recompute quadMat manually
                            Z_before = entry.get('Z_before_quadmap')
                            if Z_before:
                                Z_center = Z_before.get('center')
                                Z_gens = Z_before.get('generators')
                                if Z_center is not None and Z_gens is not None:
                                    Zmat = np.hstack([np.asarray(Z_center), np.asarray(Z_gens)])
                                    print(f"\n    Zmat:")
                                    print(f"      Shape: {Zmat.shape}")
                                    print(f"      Zmat:\n{Zmat}")
                                    
                                    # Compute quadMat
                                    quadMat_recomputed = Zmat.T @ H_mat @ Zmat
                                    print(f"\n    Recomputed quadMat:")
                                    print(f"      Shape: {quadMat_recomputed.shape}")
                                    print(f"      quadMat:\n{quadMat_recomputed}")
                                    
                                    # Compare with tracked value
                                    if 'dense_full' in info:
                                        tracked_full = info['dense_full']
                                        diff = np.abs(quadMat_recomputed - tracked_full)
                                        max_diff = np.max(diff)
                                        print(f"\n    Comparison with tracked value:")
                                        print(f"      Max absolute difference: {max_diff:.10e}")
                                        if max_diff > 1e-10:
                                            print(f"      [WARNING] Recomputed differs from tracked!")
                                            print(f"      Difference matrix:\n{diff}")
                                        else:
                                            print(f"      [OK] Recomputed matches tracked")
        else:
            print("\n[WARNING] No Step 3 entries found")
    else:
        print("\n[WARNING] No entries with quadMat tracking found")
else:
    print(f"\n[ERROR] Python log file not found: {python_file}")

print("\n" + "=" * 80)
