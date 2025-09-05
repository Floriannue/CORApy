#!/usr/bin/env python3
"""
Debug script to compare minMaxDiffOrder between MATLAB and Python
"""

import sys
sys.path.append('cora_python')
from nn.layers.nonlinear.nnReLULayer import nnReLULayer
from nn.nnHelper.leastSquarePolyFunc import leastSquarePolyFunc
from nn.nnHelper.minMaxDiffOrder import minMaxDiffOrder
from nn.nnHelper.getDerInterval import getDerInterval
import numpy as np

# Create ReLU layer
layer = nnReLULayer()

# Test parameters
l, u = -1, 1
order = 2

# Get the coefficients from regression
num_points = 10 * (order + 1)
x = np.linspace(l, u, num_points)
y = layer.f(x)
coeffs = leastSquarePolyFunc(x, y, order)

# Get derivative bounds
df_l, df_u = layer.getDerBounds(l, u)

print("=== Python minMaxDiffOrder Debug ===")
print(f"l: {l}, u: {u}")
print(f"coeffs: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
print(f"df_l: {df_l:.8f}, df_u: {df_u:.8f}")

# Call minMaxDiffOrder
diffl, diffu = minMaxDiffOrder(coeffs, l, u, layer.f, df_l, df_u)

print("minMaxDiffOrder results:")
print(f"  diffl: {diffl:.8f}")
print(f"  diffu: {diffu:.8f}")

# Compute final values
diffc = (diffl + diffu) / 2
coeffs_final = coeffs.copy()
coeffs_final[-1] = coeffs_final[-1] + diffc
d_final = diffu - diffc

print("Final computation:")
print(f"  diffc: {diffc:.8f}")
print(f"  coeffs_final: [{coeffs_final[0]:.8f}, {coeffs_final[1]:.8f}, {coeffs_final[2]:.8f}]")
print(f"  d_final: {d_final:.8f}")

# Let's also test the getDerInterval function
der2l, der2u = getDerInterval(coeffs, l, u)
print("\ngetDerInterval results:")
print(f"  der2l: {der2l:.8f}")
print(f"  der2u: {der2u:.8f}")

# Test the derivative calculation
der = np.max(np.abs([
    df_l - -der2l,
    df_l - -der2u,
    df_u - -der2l,
    df_u - -der2u
]))
print(f"  der: {der:.8f}")
