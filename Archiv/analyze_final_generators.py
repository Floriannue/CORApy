"""Analyze how final generators are computed"""
import numpy as np

print("=" * 80)
print("ANALYZING FINAL GENERATOR COMPUTATION")
print("=" * 80)

# Simulate the scenario from Step 2 Run 2
# Python: Rhom_tp has 5 generators → Rend_tp has 2 generators
# MATLAB: Rhom_tp has 5 generators → Rend_tp has 4 generators

n = 2  # 2D zonotope
nrG = 5  # 5 generators initially
last0Idx = 0  # No zero generators
redIdx_python = 3  # Python reduces 3 generators
redIdx_matlab = 1  # MATLAB reduces 1 generator

print(f"\nScenario:")
print(f"  n (dimensions): {n}")
print(f"  nrG (initial generators): {nrG}")
print(f"  last0Idx: {last0Idx}")
print(f"  redIdx (Python): {redIdx_python}")
print(f"  redIdx (MATLAB): {redIdx_matlab}")

# Simulate gensred
gensred = np.random.rand(n, nrG) * 0.01

# Python case
print("\n" + "-" * 80)
print("PYTHON CASE (redIdx = 3)")
print("-" * 80)

if redIdx_python > 0:
    Gred_py = np.sum(gensred[:, :redIdx_python], axis=1, keepdims=True)
else:
    Gred_py = np.zeros((n, 1))

Gzeros = np.zeros((n, 1))  # Assume no zero generators
Gred_total_py = (Gred_py + Gzeros).flatten()
G_diag_py = np.diag(Gred_total_py)
G_diag_py = G_diag_py[:, np.any(G_diag_py, axis=0)]

# Unreduced generators
idx = np.arange(nrG)  # Simplified - actual idx would be sorted
if last0Idx + redIdx_python < len(idx):
    gunred_idx_py = idx[last0Idx + redIdx_python:]
    gunred_idx_sorted_py = np.sort(gunred_idx_py)
    Gunred_py = gensred[:, gunred_idx_sorted_py]
else:
    Gunred_py = np.array([]).reshape(n, 0)

G_new_py = np.hstack([Gunred_py, G_diag_py]) if Gunred_py.size > 0 else G_diag_py

print(f"  Gred shape: {Gred_py.shape}")
print(f"  G_diag shape: {G_diag_py.shape}")
print(f"  Gunred shape: {Gunred_py.shape}")
print(f"  G_new shape: {G_new_py.shape}")
print(f"  Final generators (Python): {G_new_py.shape[1]}")

# MATLAB case
print("\n" + "-" * 80)
print("MATLAB CASE (redIdx = 1)")
print("-" * 80)

if redIdx_matlab > 0:
    Gred_mat = np.sum(gensred[:, :redIdx_matlab], axis=1, keepdims=True)
else:
    Gred_mat = np.zeros((n, 1))

Gred_total_mat = (Gred_mat + Gzeros).flatten()
G_diag_mat = np.diag(Gred_total_mat)
G_diag_mat = G_diag_mat[:, np.any(G_diag_mat, axis=0)]

# Unreduced generators
if last0Idx + redIdx_matlab < len(idx):
    gunred_idx_mat = idx[last0Idx + redIdx_matlab:]
    gunred_idx_sorted_mat = np.sort(gunred_idx_mat)
    Gunred_mat = gensred[:, gunred_idx_sorted_mat]
else:
    Gunred_mat = np.array([]).reshape(n, 0)

G_new_mat = np.hstack([Gunred_mat, G_diag_mat]) if Gunred_mat.size > 0 else G_diag_mat

print(f"  Gred shape: {Gred_mat.shape}")
print(f"  G_diag shape: {G_diag_mat.shape}")
print(f"  Gunred shape: {Gunred_mat.shape}")
print(f"  G_new shape: {G_new_mat.shape}")
print(f"  Final generators (MATLAB): {G_new_mat.shape[1]}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print(f"Python reduces {redIdx_python} generators -> {G_new_py.shape[1]} final generators")
print(f"MATLAB reduces {redIdx_matlab} generators -> {G_new_mat.shape[1]} final generators")
print(f"\nThe difference comes from redIdx: {redIdx_python} vs {redIdx_matlab}")
print("This means Python's h array has more values <= dHmax than MATLAB's")
