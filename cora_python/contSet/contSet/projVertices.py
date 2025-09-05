import numpy as np
from cora_python.g.functions.matlab.validate.preprocessing import setDefaultValues
from cora_python.g.functions.matlab.validate.check import compareMatrices, withinTol

def _aux_check_actual_vertex(V_list, idx):
    """
    Auxiliary function to check if a vertex is an actual corner or if it's
    on the line segment between its neighboring vertices.
    """
    num_vertices = len(V_list)
    if num_vertices <= 2:
        return V_list, True

    v_current = V_list[idx]
    v_prev = V_list[idx - 1]  # Python's negative indexing handles the wrap-around
    v_next = V_list[(idx + 1) % num_vertices]

    # Check for collinearity
    pts_start_mid_end = np.column_stack((v_current - v_next, v_prev - v_current))
    if np.linalg.matrix_rank(pts_start_mid_end, 1e-12) < 2:
        V_list.pop(idx)
        return V_list, False # Vertex was removed
    
    return V_list, True # Vertex was kept


def projVertices(S, *varargin):
    """
    projVertices - computes the vertices of a 2D projection of a set based on
    support function evaluation of the projected set; if no more support
    vectors can be found, the algorithm terminates
    this function also supports degenerate sets (lines, points)

    Syntax:
        V = projVertices(S)
        V = projVertices(S, dims)

    Inputs:
        S - contSet object
        dims - dimensions for projection

    Outputs:
        V - list of vertices in the projected space

    Example:
        # To be implemented

    Other m-files required: none
    Subfunctions: none
    MAT-files required: none

    See also: none

    Authors:
        Mark Wetzlinger
    Written:
        21-December-2022
    Last update:
        29-April-2024
    Last revision:
        ---
    """
    from cora_python.contSet.contSet.contSet import ContSet
    # Check if subclass has overridden this method
    if type(S).projVertices is not ContSet.projVertices:
        return type(S).projVertices(S, *varargin)

    # --- Primary Method Body ---
    # setDefaultValues returns only the parsed defaults list in Python
    defaults = setDefaultValues([[1, 2]], varargin)
    dims = defaults[0]
    dims_0_indexed = [d - 1 for d in dims]

    if not (isinstance(dims, list) and len(dims) == 2 and all(isinstance(d, int) and d > 0 for d in dims)):
        raise ValueError("dims must be a list of two positive integers.")

    if not hasattr(S, 'supportFunc_'):
        raise ValueError("Input set must support the 'supportFunc_' method.")

    S_proj = S.project(dims_0_indexed)

    if S_proj.is_empty():
        return np.array([])

    other_options = {}
    # Placeholder for polyZonotope/conPolyZono specific options
    # if isinstance(S_proj, (polyZonotope, conPolyZono)):
    #     other_options = {'interval': 8, 'tol': 1e-3}

    # Initial vertices
    V_init = np.zeros((2, 3))
    _, v0, _ = S_proj.supportFunc_(np.array([1, 0]), 'upper', **other_options)
    V_init[:, 0] = v0.flatten()
    
    angle_120 = 120 * np.pi / 180
    dir_120 = np.array([np.cos(angle_120), np.sin(angle_120)])
    _, v1, _ = S_proj.supportFunc_(dir_120, 'upper', **other_options)
    V_init[:, 1] = v1.flatten()
    
    angle_240 = 240 * np.pi / 180
    dir_240 = np.array([np.cos(angle_240), np.sin(angle_240)])
    _, v2, _ = S_proj.supportFunc_(dir_240, 'upper', **other_options)
    V_init[:, 2] = v2.flatten()
    
    # Use a list of vectors for easier insertion
    V_list = [V_init[:, 0]]
    idx_map = {0: 0} # Maps original (0,1,2) to current list index
    
    if not withinTol(V_init[:, 1], V_list[0], 1e-12).all():
        idx_map[1] = len(V_list)
        V_list.append(V_init[:, 1])
    if not any(withinTol(V_init[:, 2], v, 1e-12).all() for v in V_list):
        idx_map[2] = len(V_list)
        V_list.append(V_init[:, 2])

    if len(V_list) <= 1:
        return np.array(V_list).T

    num_v = len(V_list)
    sections = [[i, (i + 1) % num_v] for i in range(num_v)]

    while sections:
        section = sections.pop(0)
        idx1, idx2 = section
        
        v1 = V_list[idx1]
        v2 = V_list[idx2]
        
        v = v2 - v1
        norm_v = np.linalg.norm(v)

        if norm_v < 1e-9:
            continue

        direction = np.array([v[1], -v[0]]) / norm_v
        
        _, v_new, _ = S_proj.supportFunc_(direction, 'upper', **other_options)
        v_new = v_new.flatten()
        
        # Check if v_new is already in the list
        is_duplicate = any(withinTol(v_new, v_existing, 1e-6).all() for v_existing in V_list)
        
        # Check for collinearity
        pts_start_mid_end = np.column_stack((v_new - v1, v2 - v_new))
        is_collinear = np.linalg.matrix_rank(pts_start_mid_end, 1e-6) < 2
        
        if is_duplicate or is_collinear:
            pass # Section completed
        else:
            # Insert new vertex and update sections
            # All indices in `sections` >= idx1 + 1 must be incremented
            new_idx = idx1 + 1
            for i, sec in enumerate(sections):
                sections[i] = [s + 1 if s >= new_idx else s for s in sec]
            
            # Insert vertex
            V_list.insert(new_idx, v_new)
            
            # The original section [idx1, idx2] is now invalid.
            # idx2 has been shifted to idx2 + 1.
            # We add two new sections.
            sections.insert(0, [new_idx, idx2 + 1])
            sections.insert(0, [idx1, new_idx])
            
            # Also update the indices we track for the final check
            for k in idx_map:
                if idx_map[k] >= new_idx:
                    idx_map[k] +=1

    V = np.array(V_list).T

    # Check if first three original vertices are actual vertices
    # Need to check indices in their potentially new positions
    # Sort keys to check in order, important if one is removed
    keys_to_check = sorted(idx_map.keys())
    
    offset = 0
    for key in keys_to_check:
        current_idx = idx_map[key] - offset
        V_list, was_kept = _aux_check_actual_vertex(V_list, current_idx)
        if not was_kept:
            # Shift subsequent indices in map
            offset += 1
            # Decrement other indices in map that were after the removed one
            for k_inner in idx_map:
                if idx_map[k_inner] > idx_map[key]:
                    idx_map[k_inner] -= 1

    return np.array(V_list).T 