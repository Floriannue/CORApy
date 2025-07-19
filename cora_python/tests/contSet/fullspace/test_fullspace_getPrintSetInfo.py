"""
test_fullspace_getPrintSetInfo - unit test function for getPrintSetInfo

Syntax:
   pytest test_fullspace_getPrintSetInfo.py

Inputs:
   None

Outputs:
   None

Example: 
   pytest test_fullspace_getPrintSetInfo.py

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


class TestFullspaceGetPrintSetInfo:
    """Test class for fullspace getPrintSetInfo method"""

    def test_getPrintSetInfo_basic(self):
        """Test basic getPrintSetInfo functionality"""
        # Create fullspace object
        fs = Fullspace(2)
        
        # Call getPrintSetInfo
        abbrev, propertyOrder = fs.getPrintSetInfo()
        
        # Check results
        assert abbrev == 'fs'
        assert propertyOrder == ['dimension']

    def test_getPrintSetInfo_zero_dimension(self):
        """Test getPrintSetInfo with zero-dimensional fullspace"""
        # Create zero-dimensional fullspace
        fs = Fullspace(0)
        
        # Call getPrintSetInfo
        abbrev, propertyOrder = fs.getPrintSetInfo()
        
        # Check results
        assert abbrev == 'fs'
        assert propertyOrder == ['dimension']

    def test_getPrintSetInfo_high_dimension(self):
        """Test getPrintSetInfo with high-dimensional fullspace"""
        # Create high-dimensional fullspace
        fs = Fullspace(10)
        
        # Call getPrintSetInfo
        abbrev, propertyOrder = fs.getPrintSetInfo()
        
        # Check results
        assert abbrev == 'fs'
        assert propertyOrder == ['dimension']

# ------------------------------ END OF CODE ------------------------------ 