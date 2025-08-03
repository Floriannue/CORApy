"""
test_zonotope_volume - unit test function of volume

Authors: Matthias Althoff, Mark Wetzlinger
         Automatic python translation: Florian Nüssel BA 2025
Written: 26-July-2016
Last update: 01-May-2020 (MW, add second case)
            09-September-2020 (MA, approximate computation added)
Python translation: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def test_zonotope_volume():
    """Test zonotope volume function"""
    
    # empty set
    Z_empty = Zonotope.empty(2)
    assert Z_empty.volume_('exact') == 0
    
    # 2D zonotope
    # In MATLAB: zonotope([-4, -3, -2, -1; 1, 2, 3, 4])
    # This means: center = [-4; 1], generators = [-3, -2, -1; 2, 3, 4]
    c = np.array([-4, 1])
    G = np.array([[-3, -2, -1], [2, 3, 4]])  # 2 x 3 matrix
    Z1 = Zonotope(c, G)
    
    vol = Z1.volume_('exact')
    true_vol = 80
    assert np.isclose(vol, true_vol), f"Expected {true_vol}, got {vol}"
    
    # compare to interval
    I1 = Z1.interval()
    volInt = I1.volume()
    # convert back to zonotope
    IZ1 = Zonotope(I1)
    volIntZon = IZ1.volume_()  # has to be equal to interval volume
    
    assert vol < volInt, f"Zonotope volume {vol} should be less than interval volume {volInt}"
    assert np.isclose(volIntZon, volInt), f"Interval zonotope volume {volIntZon} should equal interval volume {volInt}"
    
    # approximate computation
    # order reduction
    volApprox_red = Z1.volume_('reduce', order=1)
    true_vol_approx_red = 122.8162136821466106
    assert np.isclose(volApprox_red, true_vol_approx_red), f"Expected {true_vol_approx_red}, got {volApprox_red}"
    
    # Alamo technique
    volApprox_alamo = Z1.volume_('alamo')
    true_vol_approx_alamo = 48.9897948556635612
    assert np.isclose(volApprox_alamo, true_vol_approx_alamo), f"Expected {true_vol_approx_alamo}, got {volApprox_alamo}" 