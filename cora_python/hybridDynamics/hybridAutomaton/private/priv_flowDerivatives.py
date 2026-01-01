"""
priv_flowDerivatives - computes the derivatives of the flow equation for
   every location with a nonlinear flow equation

Syntax:
    priv_flowDerivatives(HA,options)

Inputs:
    HA - hybridAutomaton object
    options - reachability settings

Outputs:
    -

Authors:       Niklas Kochdumper
Written:       20-May-2020
Last update:   15-October-2024 (MW, rename function)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, Dict
from cora_python.contDynamics.contDynamics.derivatives import derivatives


def priv_flowDerivatives(HA: Any, options: Dict[str, Any]) -> None:
    """
    Computes the derivatives of the flow equation for every location with
    a nonlinear flow equation
    
    Args:
        HA: hybridAutomaton object
        options: reachability settings
    """
    
    # loop over all locations
    for i in range(len(HA.location)):
        
        # read out location and corresponding flow equation
        loc = HA.location[i]
        sys = loc.contDynamics
        
        # derivatives computation only required for nonlinear systems
        if (hasattr(sys, '__class__') and 
            (sys.__class__.__name__ == 'nonlinearSys' or
             sys.__class__.__name__ == 'nonlinDASys' or
             sys.__class__.__name__ == 'nonlinParamSys')):
            
            # compute derivatives (generates files in models/auxiliary)
            derivatives(sys, options)

