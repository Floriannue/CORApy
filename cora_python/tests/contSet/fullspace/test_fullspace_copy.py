"""
test_fullspace_copy - unit test function for copy

Syntax:
   pytest test_fullspace_copy.py

Inputs:
   None

Outputs:
   None

Example: 
   pytest test_fullspace_copy.py

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


class TestFullspaceCopy:
    """Test class for fullspace copy method"""

    def test_copy_basic(self):
        """Test basic copy functionality"""
        # Create fullspace object
        fs = Fullspace(2)
        
        # Call copy
        fs_out = fs.copy()
        
        # Check that it's a copy (same dimension)
        assert fs_out.dimension == fs.dimension
        assert isinstance(fs_out, Fullspace)

    def test_copy_zero_dimension(self):
        """Test copy with zero-dimensional fullspace"""
        # Create zero-dimensional fullspace
        fs = Fullspace(0)
        
        # Call copy
        fs_out = fs.copy()
        
        # Check that it's a copy
        assert fs_out.dimension == fs.dimension
        assert isinstance(fs_out, Fullspace)

    def test_copy_high_dimension(self):
        """Test copy with high-dimensional fullspace"""
        # Create high-dimensional fullspace
        fs = Fullspace(10)
        
        # Call copy
        fs_out = fs.copy()
        
        # Check that it's a copy
        assert fs_out.dimension == fs.dimension
        assert isinstance(fs_out, Fullspace)

    def test_copy_independence(self):
        """Test that copy creates independent object"""
        # Create fullspace object
        fs = Fullspace(2)
        
        # Call copy
        fs_out = fs.copy()
        
        # Modify original (if possible)
        # Since dimension is immutable, we can't test this directly
        # But we can verify they are separate objects
        assert fs is not fs_out

# ------------------------------ END OF CODE ------------------------------ 