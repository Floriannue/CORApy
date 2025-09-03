"""
Debug script for splitDepFactor - compare with MATLAB
"""
import numpy as np
from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope

# Test the same case as MATLAB
c = np.array([-1, 3])
G = np.array([[2, 0, 1], [1, 2, 1]])
E = np.array([[1, 0, 2], [0, 1, 1]])
GI = np.array([]).reshape(2, 0)
pZ = PolyZonotope(c, G, GI, E)

print('Python Original pZ:')
print(f'  c: {pZ.c.flatten()}')
print(f'  G shape: {pZ.G.shape}')
print(f'  E shape: {pZ.E.shape}')
print(f'  id: {pZ.id.flatten()}')

# Show generator lengths
len_gen = np.sum(pZ.G**2, axis=0)
print(f'  Generator lengths: {len_gen}')
ind = np.argmax(len_gen)
print(f'  Longest generator index: {ind}')

# Show which factor has largest exponent in longest generator
factor_idx = np.argmax(pZ.E[:, ind])
factor = pZ.id[factor_idx, 0]
print(f'  Longest generator column: {ind}')
print(f'  E[:, {ind}]: {pZ.E[:, ind]}')
print(f'  Factor with largest exponent index: {factor_idx}')
print(f'  Factor value: {factor}')

print('\nCalling splitDepFactor with factor =', factor)
pZsplit = pZ.splitDepFactor(factor)

print(f'\nSplit into {len(pZsplit)} parts:')
for i, pz in enumerate(pZsplit):
    print(f'  pZsplit[{i}]:')
    print(f'    c: {pz.c.flatten()}')
    print(f'    G shape: {pz.G.shape}')
    print(f'    E shape: {pz.E.shape}')
    print(f'    G:')
    print(pz.G)
    print(f'    E:')
    print(pz.E)

print('\nPython splitDepFactor test completed!')
