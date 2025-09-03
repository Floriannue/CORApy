"""
Debug script to trace center calculation in splitDepFactor
"""
import numpy as np
from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope

# Test the same case as MATLAB
c = np.array([-1, 3])
G = np.array([[2, 0, 1], [1, 2, 1]])
E = np.array([[1, 0, 2], [0, 1, 1]])
GI = np.array([]).reshape(2, 0)
pZ = PolyZonotope(c, G, GI, E)

print('Original pZ:')
print(f'  c: {pZ.c.flatten()}')
print(f'  G: {pZ.G}')
print(f'  E: {pZ.E}')

# Manually trace through splitDepFactor logic
ind = 1  # factor to split
polyOrd = 2

# Find selected dependent factor
ind_mask = (pZ.id == ind).flatten()
E_ind = pZ.E[ind_mask, :]
print(f'\nE_ind: {E_ind}')

# Determine generators to split
genInd = ((E_ind > 0) & (E_ind <= polyOrd)).flatten()
print(f'genInd: {genInd}')

# Create polynomial coefficients
polyCoeff1 = [None] * max(2, polyOrd)
polyCoeff2 = [None] * max(2, polyOrd)

# Create pascal triangle
P = [[1, 1]]
for i in range(2, polyOrd + 1):
    prev_row = P[i-2]
    new_row = [1]
    for j in range(1, i):
        new_row.append(prev_row[j-1] + prev_row[j])
    new_row.append(1)
    P.append(new_row)

print(f'Pascal triangle P: {P}')

for i in range(1, polyOrd + 1):
    Pi = P[i-1]
    polyCoeff1[i-1] = (0.5**i) * np.array(Pi)
    polyCoeff2[i-1] = (0.5**i) * np.array(Pi) * ((-1)**np.arange(i+1))
    print(f'i={i}: Pi={Pi}')
    print(f'  polyCoeff1[{i-1}] = {polyCoeff1[i-1]}')
    print(f'  polyCoeff2[{i-1}] = {polyCoeff2[i-1]}')

# Initialize centers
c1 = pZ.c.copy()
c2 = pZ.c.copy()
print(f'\nInitial centers: c1={c1.flatten()}, c2={c2.flatten()}')

# Calculate hout
hout = np.sum(~genInd)
for i in range(1, polyOrd + 1):
    numExpi = np.sum(E_ind == i)
    hout = hout + len(P[i-1]) * numExpi

print(f'hout: {hout}')

# Create output matrices
G1 = np.full((len(c1), hout), np.nan)
G2 = np.full((len(c2), hout), np.nan)
Eout = np.full((pZ.E.shape[0], hout), np.nan)

# Fill in non-split generators
h = 0
dh = np.sum(~genInd)
if dh > 0:
    G1[:, h:h+dh] = pZ.G[:, ~genInd]
    G2[:, h:h+dh] = pZ.G[:, ~genInd]
    Eout[:, h:h+dh] = pZ.E[:, ~genInd]
    h = h + dh

print(f'After non-split generators: h={h}')
print(f'G1[:, 0:{h}]: {G1[:, 0:h]}')
print(f'G2[:, 0:{h}]: {G2[:, 0:h]}')

# Process each polynomial order
for i in range(1, polyOrd + 1):
    coef1 = polyCoeff1[i-1]
    coef2 = polyCoeff2[i-1]
    
    expi = (E_ind == i).flatten()
    dh = len(coef1) * np.sum(expi)
    
    print(f'\ni={i}: expi={expi}, dh={dh}')
    print(f'coef1={coef1}, coef2={coef2}')
    
    if np.sum(expi) > 0:
        # Kronecker product equivalent
        G1_cols = []
        G2_cols = []
        for j in range(len(coef1)):
            G1_cols.append(coef1[j] * pZ.G[:, expi])
            G2_cols.append(coef2[j] * pZ.G[:, expi])
        
        if G1_cols:
            G1[:, h:h+dh] = np.hstack(G1_cols)
            G2[:, h:h+dh] = np.hstack(G2_cols)
        
        print(f'G1[:, {h}:{h+dh}]: {G1[:, h:h+dh]}')
        print(f'G2[:, {h}:{h+dh}]: {G2[:, h:h+dh]}')
        
        # Repeat E columns
        E_repeated = np.tile(pZ.E[:, expi], (1, i+1))
        Eout[:, h:h+dh] = E_repeated
        Eout[ind_mask, h:h+dh] = np.kron(np.arange(i+1), np.ones(np.sum(expi)))
        
        h = h + dh
        genInd = np.logical_xor(genInd, expi)

print(f'\nFinal G1: {G1}')
print(f'Final G2: {G2}')
print(f'Final Eout: {Eout}')

# Add generators with all-zero exponent matrix to center
temp = np.sum(Eout, axis=0)
genInd_zero = temp == 0
print(f'\ngenInd_zero: {genInd_zero}')
print(f'G1[:, genInd_zero]: {G1[:, genInd_zero]}')
print(f'G2[:, genInd_zero]: {G2[:, genInd_zero]}')

c1_final = c1 + np.sum(G1[:, genInd_zero], axis=1, keepdims=True)
c2_final = c2 + np.sum(G2[:, genInd_zero], axis=1, keepdims=True)

print(f'\nFinal centers:')
print(f'c1_final: {c1_final.flatten()}')
print(f'c2_final: {c2_final.flatten()}')
