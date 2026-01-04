"""
display - Displays the properties of a hybridAutomaton object on the
   command window

Syntax:
    display(HA)

Inputs:
    HA - hybridAutomaton object

Outputs:
    -

Example: 
    % invariant
    inv = polytope([-1,0],0);
 
    % transition
    guard = polytope([0,1],0,[-1,0],0);
    reset = linearReset([1,0;0,-0.75]);
    trans(1) = transition(guard,reset,1);
 
    % flow equation
    dynamics = linearSys([0,1;0,0],[0;0],[0;-9.81]);
 
    % define location
    loc(1) = location('S1',inv,trans,dynamics);
 
    % instantiate hybrid automaton (and display)
    HA = hybridAutomaton('HA',loc)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       18-June-2022
Last update:   15-October-2024 (MW, add name property)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, Optional
import numpy as np
from cora_python.g.functions.verbose.display.dispEmptySet import dispEmptySet


def display(HA: Any, var_name: str = None) -> None:
    """
    Display a hybrid automaton object
    
    Args:
        HA: HybridAutomaton object
        var_name: Optional variable name (for display)
    """
    # MATLAB: fprintf(newline);
    print()
    
    # MATLAB: disp([inputname(1), ' =']);
    if var_name is None:
        var_name = 'HA'
    print(f"{var_name} =")
    
    # MATLAB: fprintf(newline);
    print()
    
    # MATLAB: if isemptyobject(HA)
    if hasattr(HA, 'isemptyobject') and HA.isemptyobject():
        # MATLAB: dispEmptyObj(HA,inputname(1));
        # Simplified: just display empty message
        if hasattr(HA, 'name'):
            print(f"  {HA.name} =")
        print("  (empty hybridAutomaton)")
    # MATLAB: elseif length(HA) > 1
    # For now, handle single objects (not arrays)
    # TODO: Handle arrays if needed
    else:
        # MATLAB: display name
        # fprintf('Hybrid automaton: ''%s''\n', HA.name);
        if hasattr(HA, 'name'):
            print(f"Hybrid automaton: '{HA.name}'")
        
        # MATLAB: number of locations
        # numLoc = length(HA.location);
        if hasattr(HA, 'location') and HA.location is not None:
            num_loc = len(HA.location)
            
            # MATLAB: loop over locations
            # for i=1:numLoc
            for i in range(num_loc):
                # MATLAB: number of location
                # disp("Location " + i + " of " + numLoc + ":");
                print(f"Location {i+1} of {num_loc}:")
                # MATLAB: display(HA.location(i));
                loc = HA.location[i]
                if hasattr(loc, 'display'):
                    loc.display(call_from_hybrid_display=True)
                else:
                    print(f"  {loc}")
    
    # MATLAB: fprintf(newline);
    print()

