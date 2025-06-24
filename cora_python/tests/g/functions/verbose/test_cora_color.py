"""
test_cora_color - unit test function for cora_color

Syntax:
    pytest test_cora_color.py

Inputs:
    -

Outputs:
    test results

Other modules required: none
Subfunctions: none

See also: none

Authors: AI Assistant
Written: 2025
Last update: ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.g.functions.verbose.plot.color.cora_color import cora_color


class TestCoraColor:
    """Test class for cora_color function"""
    
    def test_cora_color_basic_colors(self):
        """Test basic CORA color identifiers"""
        
        # Test initial set color (white)
        color = cora_color('CORA:initialSet')
        expected = np.array([1, 1, 1])
        assert np.allclose(color, expected)
        
        # Test final set color (light gray)
        color = cora_color('CORA:finalSet')
        expected = np.array([0.9, 0.9, 0.9])
        assert np.allclose(color, expected)
        
        # Test simulations color (black)
        color = cora_color('CORA:simulations')
        expected = np.array([0, 0, 0])
        assert np.allclose(color, expected)
        
        # Test unsafe color (red)
        color = cora_color('CORA:unsafe')
        expected = np.array([0.9451, 0.5529, 0.5686])
        assert np.allclose(color, expected)
        
        # Test safe color (green)
        color = cora_color('CORA:safe')
        expected = np.array([0.4706, 0.7725, 0.4980])
        assert np.allclose(color, expected)
        
        # Test invariant color (same as safe)
        color = cora_color('CORA:invariant')
        expected = np.array([0.4706, 0.7725, 0.4980])
        assert np.allclose(color, expected)
    
    def test_cora_color_highlights(self):
        """Test CORA highlight colors"""
        
        # Test highlight1 (orange)
        color = cora_color('CORA:highlight1')
        expected = np.array([1.0000, 0.6824, 0.2980])
        assert np.allclose(color, expected)
        
        # Test highlight2 (light green)
        color = cora_color('CORA:highlight2')
        expected = np.array([0.6235, 0.7294, 0.2118])
        assert np.allclose(color, expected)
    
    def test_cora_color_matlab_defaults(self):
        """Test MATLAB default colors"""
        
        # Test blue (color1)
        color1 = cora_color('CORA:color1')
        color_blue = cora_color('CORA:blue')
        expected = np.array([0, 0.4470, 0.7410])
        assert np.allclose(color1, expected)
        assert np.allclose(color_blue, expected)
        assert np.allclose(color1, color_blue)
        
        # Test red (color2)
        color2 = cora_color('CORA:color2')
        color_red = cora_color('CORA:red')
        expected = np.array([0.8500, 0.3250, 0.0980])
        assert np.allclose(color2, expected)
        assert np.allclose(color_red, expected)
        
        # Test yellow (color3)
        color3 = cora_color('CORA:color3')
        color_yellow = cora_color('CORA:yellow')
        expected = np.array([0.9290, 0.6940, 0.1250])
        assert np.allclose(color3, expected)
        assert np.allclose(color_yellow, expected)
        
        # Test purple (color4)
        color4 = cora_color('CORA:color4')
        color_purple = cora_color('CORA:purple')
        expected = np.array([0.4940, 0.1840, 0.5560])
        assert np.allclose(color4, expected)
        assert np.allclose(color_purple, expected)
        
        # Test green (color5)
        color5 = cora_color('CORA:color5')
        color_green = cora_color('CORA:green')
        expected = np.array([0.4660, 0.6740, 0.1880])
        assert np.allclose(color5, expected)
        assert np.allclose(color_green, expected)
        
        # Test light blue (color6)
        color6 = cora_color('CORA:color6')
        color_light_blue = cora_color('CORA:light-blue')
        expected = np.array([0.3010, 0.7450, 0.9330])
        assert np.allclose(color6, expected)
        assert np.allclose(color_light_blue, expected)
        
        # Test dark red (color7)
        color7 = cora_color('CORA:color7')
        color_dark_red = cora_color('CORA:dark-red')
        expected = np.array([0.6350, 0.0780, 0.1840])
        assert np.allclose(color7, expected)
        assert np.allclose(color_dark_red, expected)
    
    def test_cora_color_reachSet_default(self):
        """Test CORA reachSet colors with default parameters"""
        
        # Default case (num_colors=1, cidx=1)
        color = cora_color('CORA:reachSet')
        expected_main = np.array([0.2706, 0.5882, 1.0000])  # blue
        assert np.allclose(color, expected_main)
        
        # Explicit default parameters
        color = cora_color('CORA:reachSet', 1, 1)
        assert np.allclose(color, expected_main)
    
    def test_cora_color_reachSet_multiple(self):
        """Test CORA reachSet colors with multiple colors"""
        
        # Two colors: first should be light blue, last should be main blue
        color_worse = np.array([0.6902, 0.8235, 1.0000])  # light blue
        color_main = np.array([0.2706, 0.5882, 1.0000])   # blue
        
        # First color (cidx=1)
        color1 = cora_color('CORA:reachSet', 2, 1)
        assert np.allclose(color1, color_worse)
        
        # Last color (cidx=2)
        color2 = cora_color('CORA:reachSet', 2, 2)
        assert np.allclose(color2, color_main)
        
        # Three colors
        color1 = cora_color('CORA:reachSet', 3, 1)
        color2 = cora_color('CORA:reachSet', 3, 2)  # interpolated
        color3 = cora_color('CORA:reachSet', 3, 3)
        
        assert np.allclose(color1, color_worse)
        assert np.allclose(color3, color_main)
        
        # Middle color should be interpolated
        expected_middle = color_worse + (color_main - color_worse) * 0.5
        assert np.allclose(color2, expected_middle)
    
    def test_cora_color_reachSet_interpolation(self):
        """Test CORA reachSet color interpolation"""
        
        color_worse = np.array([0.6902, 0.8235, 1.0000])  # light blue
        color_main = np.array([0.2706, 0.5882, 1.0000])   # blue
        
        # Test with 5 colors
        for i in range(1, 6):
            color = cora_color('CORA:reachSet', 5, i)
            
            if i == 1:
                assert np.allclose(color, color_worse)
            elif i == 5:
                assert np.allclose(color, color_main)
            else:
                # Check that it's between the two colors
                expected = color_worse + (color_main - color_worse) * ((i - 1) / 4)
                assert np.allclose(color, expected)
    
    def test_cora_color_next(self):
        """Test CORA:next color functionality"""
        
        # Should return some default color from matplotlib
        color = cora_color('CORA:next')
        
        # Should be a valid RGB color
        assert isinstance(color, np.ndarray)
        assert color.shape == (3,)
        assert np.all(color >= 0) and np.all(color <= 1)
    
    def test_cora_color_error_cases(self):
        """Test error cases for cora_color"""
        
        # Invalid identifier
        with pytest.raises(Exception):  # CORAerror
            cora_color('INVALID:color')
        
        # Invalid reachSet arguments
        with pytest.raises(Exception):  # CORAerror for cidx > num_colors
            cora_color('CORA:reachSet', 2, 3)
        
        # Invalid argument types (should be caught by input validation)
        with pytest.raises(Exception):
            cora_color('CORA:reachSet', -1, 1)
        
        with pytest.raises(Exception):
            cora_color('CORA:reachSet', 1.5, 1)
    
    def test_cora_color_unsafeLight(self):
        """Test CORA unsafeLight color"""
        
        color = cora_color('CORA:unsafeLight')
        expected = np.array([0.9059, 0.7373, 0.7373])
        assert np.allclose(color, expected)
    
    def test_cora_color_return_type(self):
        """Test that all colors return proper numpy arrays"""
        
        # List of all valid identifiers
        identifiers = [
            'CORA:initialSet', 'CORA:finalSet', 'CORA:simulations',
            'CORA:unsafe', 'CORA:unsafeLight', 'CORA:safe', 'CORA:invariant',
            'CORA:highlight1', 'CORA:highlight2', 'CORA:next',
            'CORA:color1', 'CORA:blue', 'CORA:color2', 'CORA:red',
            'CORA:color3', 'CORA:yellow', 'CORA:color4', 'CORA:purple',
            'CORA:color5', 'CORA:green', 'CORA:color6', 'CORA:light-blue',
            'CORA:color7', 'CORA:dark-red'
        ]
        
        for identifier in identifiers:
            color = cora_color(identifier)
            
            # Check type and shape
            assert isinstance(color, np.ndarray)
            assert color.shape == (3,)
            
            # Check value range
            assert np.all(color >= 0) and np.all(color <= 1)
        
        # Test reachSet with different parameters
        color = cora_color('CORA:reachSet', 1, 1)
        assert isinstance(color, np.ndarray)
        assert color.shape == (3,)
        assert np.all(color >= 0) and np.all(color <= 1)


if __name__ == "__main__":
    pytest.main([__file__]) 