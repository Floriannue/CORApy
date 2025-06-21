"""
use_cora_colors - sets the CORA plotting colormap

This function sets matplotlib color cycles to match CORA's standardized colors.

Syntax:
    use_cora_colors(identifier)

Inputs:
    identifier - name of CORA colormap, one of:
       - 'CORA:default'             - matlab default color order
       - 'CORA:contDynamics'        - plot reachSet, initialSet, simRes 
       - 'CORA:manual'              - CORA manual: default colors
       - 'CORA:manual-result'       - CORA manual: colors for result plots
    *args - depends on identifier, see below

Outputs:
    -

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       01-March-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Any
from .cora_color import cora_color


def use_cora_colors(identifier: str, *args) -> None:
    """
    Sets the CORA plotting colormap
    
    Args:
        identifier: Name of CORA colormap
        *args: Additional arguments depending on identifier
        
    Raises:
        ValueError: If identifier is not recognized
    """
    # Runtime imports to avoid circular dependencies
    from ....matlab.validate.preprocessing.set_default_values import set_default_values
    
    # Input validation
    valid_identifiers = [
        'CORA:default', 'CORA:contDynamics', 'CORA:manual', 'CORA:manual-result'
    ]
    
    if not isinstance(identifier, str):
        raise ValueError("Identifier must be a string")
    if identifier not in valid_identifiers:
        raise ValueError(f"Invalid identifier '{identifier}'. Must be one of: {valid_identifiers}")
    
    colors = []
    
    if identifier == 'CORA:default':
        # MATLAB default colors
        colors = [
            cora_color('CORA:color1'),
            cora_color('CORA:color2'),
            cora_color('CORA:color3'),
            cora_color('CORA:color4'),
            cora_color('CORA:color5'),
            cora_color('CORA:color6'),
            cora_color('CORA:color7')
        ]
        
    elif identifier == 'CORA:contDynamics':
        # varargin - {numColors}
        #    - numColors: number of reachSet colors
        
        # Use setDefaultValues like in MATLAB
        num_colors = set_default_values([1], list(args))[0]
        
        # Generate reachSet colors
        reach_set_colors = []
        for cidx in range(1, num_colors + 1):
            reach_set_colors.append(cora_color('CORA:reachSet', num_colors, cidx))
        
        colors = reach_set_colors + [
            cora_color('CORA:initialSet'),
            cora_color('CORA:simulations')
        ]
        
    elif identifier == 'CORA:manual':
        # MATLAB default colors (same as default)
        colors = [
            cora_color('CORA:color1'),
            cora_color('CORA:color2'),
            cora_color('CORA:color3'),
            cora_color('CORA:color4'),
            cora_color('CORA:color5'),
            cora_color('CORA:color6'),
            cora_color('CORA:color7')
        ]
        
    elif identifier == 'CORA:manual-result':
        colors = [
            np.array([0.4660, 0.6740, 0.1880])  # green
        ]
    
    # Convert colors to the format matplotlib expects
    if colors:
        # Set the color cycle for current axes
        ax = plt.gca()
        color_cycle = plt.cycler('color', colors)
        ax.set_prop_cycle(color_cycle)
        
        # Reset color order index to start from beginning
        ax._get_lines.prop_cycler = iter(color_cycle)


# Alias for MATLAB compatibility
useCORAcolors = use_cora_colors 