"""
test_emptySet_volume - unit tests for emptySet/volume_

Syntax:
    python -m pytest cora_python/tests/contSet/emptySet/test_emptySet_volume.py

Authors: Python translation by AI Assistant  
Written: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.emptySet import EmptySet


class TestEmptySetVolume:
    """Test class for emptySet volume_ method"""
    
    def test_volume_2d(self):
        """Test volume_ method for 2D empty set"""
        O = EmptySet(2)
        vol = O.volume_()
        
        # Volume of empty set is always 0
        assert vol == 0.0
        
    def test_volume_different_dimensions(self):
        """Test volume_ method for different dimensions"""
        dimensions = [0, 1, 3, 5, 10]
        
        for n in dimensions:
            O = EmptySet(n)
            vol = O.volume_()
            
            # Volume is always 0 regardless of dimension
            assert vol == 0.0
            assert isinstance(vol, float)


if __name__ == '__main__':
    pytest.main([__file__]) 