"""
display - Displays a location object on the command window

Syntax:
    display(loc)

Inputs:
    loc - location object

Outputs:
    ---

Example:
    inv = interval([-2;-1],[1;2]);
    trans = transition(polytope([],[],[2 0],1),linearReset.eye(2),2);
    sys = linearSys([1 -3; -2 1],[0;1]);
    loc = location(inv,trans,sys)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff, Mark Wetzlinger
Written:       06-November-2007
Last update:   18-June-2022 (MW, update displayed information)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, Optional
import numpy as np


def display_(loc: Any, var_name: str = None, call_from_hybrid_display: bool = False) -> str:
    """
    Display a location object (internal function that returns string)
    
    Args:
        loc: Location object
        var_name: Optional variable name (for display)
        call_from_hybrid_display: Whether called from hybridAutomaton.display
    
    Returns:
        String representation for display
    """
    output_lines = []
    # MATLAB: check if called from display hybridAutomaton
    # st = dbstack("-completenames");
    # For Python, we use the call_from_hybrid_display parameter
    
    if not call_from_hybrid_display:
        # MATLAB: fprintf(newline);
        output_lines.append("")
        # MATLAB: disp([inputname(1), ' =']);
        if var_name is None:
            var_name = 'loc'
        output_lines.append(f"{var_name} =")
        # MATLAB: fprintf(newline);
        output_lines.append("")
    
    # MATLAB: array of location objects
    # if length(loc) > 1
    # For now, handle single objects (not arrays)
    # TODO: Handle arrays if needed
    
    # MATLAB: display name
    # if strcmp(loc.name,'location')
    if hasattr(loc, 'name'):
        if loc.name == 'location':
            output_lines.append(f"  Name: '{loc.name}' (default)")
        else:
            output_lines.append(f"  Name: '{loc.name}'")
    
    # MATLAB: invariant
    # if isnumeric(loc.invariant) && isempty(loc.invariant)
    inv_is_numeric_empty = isinstance(loc.invariant, (int, float, np.number)) or \
                          (isinstance(loc.invariant, np.ndarray) and loc.invariant.size == 0)
    
    if inv_is_numeric_empty:
        output_lines.append("  Invariant: []")
    else:
        # MATLAB: disp("  Invariant: " + class(loc.invariant) + ...
        #         " (dimension: " + dim(loc.invariant) + ")");
        inv_class = type(loc.invariant).__name__
        inv_dim = 0
        if hasattr(loc.invariant, 'dim'):
            inv_dim = loc.invariant.dim()
        output_lines.append(f"  Invariant: {inv_class} (dimension: {inv_dim})")
    
    # MATLAB: transitions
    # if isempty(loc.transition)
    if not hasattr(loc, 'transition') or len(loc.transition) == 0:
        output_lines.append("  Number of transitions: 0")
    else:
        num_trans = len(loc.transition)
        trans_str = f"  Number of transitions: {num_trans} ("
        
        # MATLAB: if isscalar(loc.transition(1).target)
        first_target = loc.transition[0].target
        is_scalar_target = isinstance(first_target, (int, float, np.integer, np.floating)) or \
                          (isinstance(first_target, np.ndarray) and first_target.size == 1)
        
        if is_scalar_target:
            # MATLAB: targetLoc = arrayfun(@(x) x.target,loc.transition,'UniformOutput',true);
            target_locs = [trans.target if isinstance(trans.target, (int, float)) else trans.target.item() 
                          for trans in loc.transition]
            
            # MATLAB: different grammar...
            if num_trans == 1:
                add_string = "target location: "
            else:
                add_string = "target locations: "
            
            # MATLAB: loop over targets of transition and synchronization labels
            temp = []
            for i, trans in enumerate(loc.transition):
                sync_label = trans.syncLabel if hasattr(trans, 'syncLabel') else ''
                if not sync_label:
                    temp.append(str(target_locs[i]))
                else:
                    temp.append(f"{target_locs[i]} ('{sync_label}')")
        else:
            # MATLAB: location from location product of parallel hybrid automaton
            temp = []
            for trans in loc.transition:
                sync_label = trans.syncLabel if hasattr(trans, 'syncLabel') else ''
                target_arr = np.array(trans.target).flatten()
                target_str = ",".join(str(t) for t in target_arr)
                if not sync_label:
                    temp.append(f"[{target_str}]")
                else:
                    temp.append(f"[{target_str}] ('{sync_label}')")
            add_string = "target locations: "
        
        # MATLAB: extend first entry by additional string
        temp[0] = add_string + temp[0]
        # MATLAB: extend last entry by parenthesis
        temp[-1] = temp[-1] + ")"
        
        # MATLAB: display transition
        # dispUpToLength(temp,100,transStr);
        # Simplified: just join and display
        full_str = trans_str + ", ".join(temp)
        output_lines.append(full_str)
    
    # MATLAB: dynamics
    # disp("  Dynamics: " + class(loc.contDynamics) + ...
    #     " (state dim.: " + loc.contDynamics.nrOfDims + ...
    #     ", input dim.: " + loc.contDynamics.nrOfInputs + ...
    #     ", output dim.: " + loc.contDynamics.nrOfOutputs + ")");
    if hasattr(loc, 'contDynamics') and loc.contDynamics is not None:
        dyn_class = type(loc.contDynamics).__name__
        nr_of_dims = loc.contDynamics.nr_of_dims if hasattr(loc.contDynamics, 'nr_of_dims') else \
                     (loc.contDynamics.nrOfDims if hasattr(loc.contDynamics, 'nrOfDims') else 0)
        nr_of_inputs = loc.contDynamics.nr_of_inputs if hasattr(loc.contDynamics, 'nr_of_inputs') else \
                       (loc.contDynamics.nrOfInputs if hasattr(loc.contDynamics, 'nrOfInputs') else 0)
        nr_of_outputs = loc.contDynamics.nr_of_outputs if hasattr(loc.contDynamics, 'nr_of_outputs') else \
                        (loc.contDynamics.nrOfOutputs if hasattr(loc.contDynamics, 'nrOfOutputs') else 0)
        output_lines.append(f"  Dynamics: {dyn_class} (state dim.: {nr_of_dims}, input dim.: {nr_of_inputs}, output dim.: {nr_of_outputs})")
    
    if not call_from_hybrid_display:
        # MATLAB: fprintf(newline);
        output_lines.append("")
    
    return "\n".join(output_lines)


def display(loc: Any, var_name: str = None, call_from_hybrid_display: bool = False) -> None:
    """
    Display a location object (prints to stdout)
    
    Args:
        loc: Location object
        var_name: Optional variable name (for display)
        call_from_hybrid_display: Whether called from hybridAutomaton.display
    """
    print(display_(loc, var_name, call_from_hybrid_display), end='')
