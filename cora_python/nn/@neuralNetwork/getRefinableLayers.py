"""
getRefinableLayers - get refinable layers

Description:
    Get refinable layers

Syntax:
    refinable_layers = getRefinableLayers()

Inputs:
    None

Outputs:
    refinable_layers - List of refinable layers

Example:
    refinable_layers = nn.getRefinableLayers()

Authors:       Niklas Kochdumper, Tobias Ladner
Written:       23-November-2022 (polish)
Last update:   23-November-2022 (polish)
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import List

def getRefinableLayers(self) -> List[Any]:
    """
    Get refinable layers
    
    Returns:
        refinable_layers: List of refinable layers
    """
    refinable_layers = []
    for layer in self.layers:
        if hasattr(layer, 'is_refinable') and layer.is_refinable:
            refinable_layers.append(layer)
    return refinable_layers
