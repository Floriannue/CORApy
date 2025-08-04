"""
test_zonotope_intersectStrip_volumeOptimum - unit test function of 
intersectStrip to check whether the optimum volume is found.
According to [1], the volume of the resulting zonotope is a convex
function.

References:
    [1] T. Alamo, J. M. Bravo, and E. F. Camacho. Guaranteed
        state estimation by zonotopes. Automatica, 41(6):1035–1043,
        2005.

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       23-December-2020 (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
import pytest
from itertools import product
from cora_python.contSet.zonotope import Zonotope


def test_zonotope_intersectStrip_volumeOptimum():
    """Test volume optimization property of intersectStrip"""
    # Assume true
    res = True
    
    # Simple 2D example which can be easily visualized
    # Zonotope
    Z = Zonotope(
        np.array([[-0.0889], [-0.2158]]),
        np.array([[-0.5526, -1.2237, -0.0184, -0.1200],
                  [1.6579, 3.6711, -0.0447, 0.0200]])
    )
    
    # Strip
    C = np.array([[-2, 1]])
    y_strip = np.array([[4.83]])
    sigma = np.array([[0.2]])
    
    # Full factorial test of different lambda values
    # Step size and increments
    delta = 0.1
    increment = int(2/delta) + 1
    # Combinations - MATLAB uses combinator(increment,2,'p','r') for permutations with replacement
    # This is equivalent to generating all combinations of 2 elements from increment values with replacement
    lambda_range = np.linspace(-1, 1, increment)
    combs = []
    for i in range(increment):
        for j in range(increment):
            combs.append([lambda_range[i], lambda_range[j]])
    lambdaMat = np.array(combs)
    
    vol = []
    for i in range(len(lambdaMat)):
        # Obtain zonotopes 
        lambda_val = lambdaMat[i, :].reshape(-1, 1)
        Zres = Z.intersectStrip(C, sigma, y_strip, lambda_val)
        # Compute volume
        vol.append(Zres.volume_())
    
    # Compute result of optimization 
    Zopt = Z.intersectStrip(C, sigma, y_strip, 'alamo-volume')
    
    # Check if volume of optimal solution is below brute force method
    volOpt = Zopt.volume_()
    assert volOpt < 1.01 * min(vol)
    
    # Optional: Create plot
    # figure;
    # view([124 34]);
    # grid('on');
    # hold on;
    # 
    # # create surface
    # [X,Y] = meshgrid(-1:delta:1,-1:delta:1);
    # Z = reshape(vol,size(X));
    # surf(X,Y,Z,'LineStyle','none');
    
    assert res


if __name__ == "__main__":
    test_zonotope_intersectStrip_volumeOptimum() 