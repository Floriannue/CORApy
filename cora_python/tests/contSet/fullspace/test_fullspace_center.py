"""
test_fullspace_center - unit test function for center

Syntax:
   pytest test_fullspace_center.py

Inputs:
   None

Outputs:
   None

Example: 
   pytest test_fullspace_center.py

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       22-March-2023
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

import pytest
import numpy as np
from cora_python.contSet.fullspace import Fullspace


class TestFullspaceCenter:
    """Test class for fullspace center method"""

    def test_center_basic(self):
        """Test basic center functionality"""
        # Create fullspace object
        fs = Fullspace(2)
        
        # Call center
        c = fs.center()
        
        # Check results
        assert np.array_equal(c, np.zeros(2))

    def test_center_zero_dimension(self):
        """Test center with zero-dimensional fullspace"""
        # Create zero-dimensional fullspace
        fs = Fullspace(0)
        
        # Call center
        c = fs.center()
        
        # Check results (should be NaN for R^0)
        assert np.isnan(c)

    def test_center_high_dimension(self):
        """Test center with high-dimensional fullspace"""
        # Create high-dimensional fullspace
        fs = Fullspace(5)
        
        # Call center
        c = fs.center()
        
        # Check results
        assert np.array_equal(c, np.zeros(5))

    def test_center_one_dimension(self):
        """Test center with one-dimensional fullspace"""
        # Create one-dimensional fullspace
        fs = Fullspace(1)
        
        # Call center
        c = fs.center()
        
        # Check results
        assert np.array_equal(c, np.zeros(1))

# ------------------------------ END OF CODE ------------------------------ 