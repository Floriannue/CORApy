import numpy as np

N = 100

# Old method (loop-based)
mat_old = np.zeros((N, N))
mat_old[0, 0] = 2
mat_old[0, 1] = -1
mat_old[N-1, N-2] = -1
mat_old[N-1, N-1] = 1
for r in range(1, N-1):
    mat_old[r, 0+(r-1)] = -1
    mat_old[r, 1+(r-1)] = 2
    mat_old[r, 2+(r-1)] = -1

# New method (vectorized)
mat_new = np.diag(2 * np.ones(N))
mat_new[-1, -1] = 1
mat_new += np.diag(-1 * np.ones(N-1), -1)
mat_new += np.diag(-1 * np.ones(N-1), 1)

print('Tridiagonal construction test:', 'PASSED' if np.allclose(mat_old, mat_new) else 'FAILED')
print('Max difference:', np.max(np.abs(mat_old - mat_new)))
