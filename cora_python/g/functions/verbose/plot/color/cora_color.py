"""
cora_color - returns the CORA default colors by identifier

This function provides standardized colors for different CORA plotting purposes.

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       02-March-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, Tuple, Optional, Any, TYPE_CHECKING
from .default_plot_color import default_plot_color

if TYPE_CHECKING:
    from ....matlab.validate.check.input_args_check import input_args_check
    from ....matlab.validate.postprocessing.CORAerror import CORAerror


def cora_color(identifier: str, *args) -> np.ndarray:
    # Runtime imports to avoid circular dependencies
    from ....matlab.validate.check.input_args_check import input_args_check
    from ....matlab.validate.postprocessing.CORAerror import CORAerror
    from ....matlab.validate.preprocessing.set_default_values import set_default_values
    """
    Returns the CORA default colors by identifier
    
    Args:
        identifier: Name of CORA colormap, one of:
            - 'CORA:reachSet'
            - 'CORA:initialSet'
            - 'CORA:finalSet'
            - 'CORA:simulations'
            - 'CORA:unsafe'
            - 'CORA:safe'
            - 'CORA:highlight1': orange
            - 'CORA:highlight2': blue
            - 'CORA:next': next color according to colororder
            - 'CORA:color<i>': matlab default color order colors
            - 'CORA:blue','CORA:red',...: matlab default colors
        *args: Additional arguments depending on identifier
            For 'CORA:reachSet': (num_colors, cidx)
                - num_colors: number of reachSet colors
                - cidx: index of reachSet color
                
    Returns:
        RGB color triple as numpy array
        
    Raises:
        CORAerror: If identifier is not recognized or arguments are invalid
    """
    # Input validation
    admissible_colors = [
        'CORA:reachSet', 'CORA:initialSet', 'CORA:finalSet', 'CORA:simulations',
        'CORA:unsafe', 'CORA:unsafeLight', 'CORA:safe', 'CORA:invariant',
        'CORA:highlight1', 'CORA:highlight2', 'CORA:next',
        'CORA:color1', 'CORA:blue', 'CORA:color2', 'CORA:red',
        'CORA:color3', 'CORA:yellow', 'CORA:color4', 'CORA:purple',
        'CORA:color5', 'CORA:green', 'CORA:color6', 'CORA:light-blue',
        'CORA:color7', 'CORA:dark-red'
    ]
    
    input_args_check({'identifier': (identifier, 'str', admissible_colors)})
    
    # Default color (black)
    color = np.array([0, 0, 0])
    
    if identifier == 'CORA:reachSet':
        # varargin - {numColors, cidx}
        #    - numColors: number of reachSet colors
        #    - cidx: index of reachSet color
        
        # Use setDefaultValues like in MATLAB
        num_colors, cidx = set_default_values([1, 1], list(args))
        
        # Validate arguments
        input_args_check({
            'num_colors': (num_colors, 'att', 'numeric', ['integer', 'scalar', 'positive']),
            'cidx': (cidx, 'att', 'numeric', ['integer', 'scalar', 'positive'])
        })
        
        if cidx > num_colors:
            raise CORAerror('CORA:wrongValue', 'second/third', 
                          'Color index must not be larger than number of colors.')
        
        color_main = np.array([0.2706, 0.5882, 1.0000])   # blue
        color_worse = np.array([0.6902, 0.8235, 1.0000])  # light blue
        
        if cidx == num_colors:
            color = color_main
        elif cidx == 1:
            color = color_worse
        else:
            color = color_worse + (color_main - color_worse) * ((cidx - 1) / (num_colors - 1))
            
    elif identifier == 'CORA:initialSet':
        color = np.array([1, 1, 1])
        
    elif identifier == 'CORA:finalSet':
        color = np.array([1, 1, 1]) * 0.9
        
    elif identifier == 'CORA:simulations':
        color = np.array([0, 0, 0])
        
    elif identifier == 'CORA:unsafe':
        color = np.array([0.9451, 0.5529, 0.5686])  # red
        
    elif identifier == 'CORA:unsafeLight':
        color = np.array([0.9059, 0.7373, 0.7373])  # light red
        
    elif identifier in ['CORA:safe', 'CORA:invariant']:
        color = np.array([0.4706, 0.7725, 0.4980])  # green
        
    elif identifier == 'CORA:highlight1':
        color = np.array([1.0000, 0.6824, 0.2980])  # orange
        
    elif identifier == 'CORA:highlight2':
        color = np.array([0.6235, 0.7294, 0.2118])  # light green
        
    elif identifier == 'CORA:next':
        color = default_plot_color()
        
    # MATLAB default colors
    elif identifier in ['CORA:color1', 'CORA:blue']:
        color = np.array([0, 0.4470, 0.7410])       # blue
        
    elif identifier in ['CORA:color2', 'CORA:red']:
        color = np.array([0.8500, 0.3250, 0.0980])  # red
        
    elif identifier in ['CORA:color3', 'CORA:yellow']:
        color = np.array([0.9290, 0.6940, 0.1250])  # yellow
        
    elif identifier in ['CORA:color4', 'CORA:purple']:
        color = np.array([0.4940, 0.1840, 0.5560])  # purple
        
    elif identifier in ['CORA:color5', 'CORA:green']:
        color = np.array([0.4660, 0.6740, 0.1880])  # green
        
    elif identifier in ['CORA:color6', 'CORA:light-blue']:
        color = np.array([0.3010, 0.7450, 0.9330])  # light blue
        
    elif identifier in ['CORA:color7', 'CORA:dark-red']:
        color = np.array([0.6350, 0.0780, 0.1840])  # dark red
        
    else:
        raise CORAerror("CORA:wrongValue", "first", admissible_colors)

    return color