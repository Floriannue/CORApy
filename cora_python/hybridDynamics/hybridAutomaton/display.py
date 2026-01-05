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


def display_(HA: Any, var_name: str = None) -> str:
    """
    Display a hybrid automaton object (internal function that returns string)
    
    Args:
        HA: HybridAutomaton object
        var_name: Optional variable name (for display)
    
    Returns:
        String representation for display
    """
    output_lines = []
    
    # MATLAB: fprintf(newline);
    output_lines.append("")
    
    # MATLAB: disp([inputname(1), ' =']);
    if var_name is None:
        var_name = 'HA'
    output_lines.append(f"{var_name} =")
    
    # MATLAB: fprintf(newline);
    output_lines.append("")
    
    # MATLAB: if isemptyobject(HA)
    if hasattr(HA, 'isemptyobject') and HA.isemptyobject():
        # MATLAB: dispEmptyObj(HA,inputname(1));
        # Simplified: just display empty message
        if hasattr(HA, 'name'):
            output_lines.append(f"  {HA.name} =")
        output_lines.append("  (empty hybridAutomaton)")
    # MATLAB: elseif length(HA) > 1
    # For now, handle single objects (not arrays)
    # TODO: Handle arrays if needed
    else:
        # MATLAB: display name
        # fprintf('Hybrid automaton: ''%s''\n', HA.name);
        if hasattr(HA, 'name'):
            output_lines.append(f"Hybrid automaton: '{HA.name}'")
        
        # MATLAB: number of locations
        # numLoc = length(HA.location);
        if hasattr(HA, 'location') and HA.location is not None:
            num_loc = len(HA.location)
            
            # MATLAB: loop over locations
            # for i=1:numLoc
            for i in range(num_loc):
                # MATLAB: number of location
                # disp("Location " + i + " of " + numLoc + ":");
                output_lines.append(f"Location {i+1} of {num_loc}:")
                # MATLAB: display(HA.location(i));
                loc = HA.location[i]
                if hasattr(loc, 'display_'):
                    # Use display_ to get string representation
                    loc_str = loc.display_(call_from_hybrid_display=True)
                    # Indent each line
                    for line in loc_str.split('\n'):
                        if line.strip():
                            output_lines.append(f"  {line}")
                else:
                    output_lines.append(f"  {loc}")
    
    # MATLAB: fprintf(newline);
    output_lines.append("")
    
    return "\n".join(output_lines)


def display(HA: Any, var_name: str = None) -> None:
    """
    Display a hybrid automaton object (prints to stdout)
    
    Args:
        HA: HybridAutomaton object
        var_name: Optional variable name (for display)
    """
    print(display_(HA, var_name), end='')
