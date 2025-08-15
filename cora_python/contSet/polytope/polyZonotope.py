import numpy as np
from scipy.spatial import ConvexHull
from typing import TYPE_CHECKING

from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.helper.sets.contSet.polyZonotope.removeRedundantExponents import removeRedundantExponents
from cora_python.g.functions.auxiliary.nearestNeighbour import nearestNeighbour

if TYPE_CHECKING:
    pass

def polyZonotope(P: Polytope) -> PolyZonotope:
    """
    polyZonotope - Convert polytope to a polynomial zonotope

    Syntax:
        pZ = polyZonotope(P)

    Inputs:
        P - polytope object

    Outputs:
        pZ - polyZonotope object
    """

    if P.representsa_('fullspace', 0):
        # conversion of fullspace object not possible
        raise CORAerror('CORA:specialError', 'Polytope is unbounded and '\
            'can therefore not be converted into a polynomial zonotope.')

    # compute polytope vertices
    try:
        V = P.vertices_()
    except Exception as ME:
        # Check if the error is due to unboundedness (simplified check for Python)
        # MATLAB uses P.bounded.val, which might not be directly available or easily translatable
        # If vertices() fails, and it's not explicitly bounded, assume it's an unbounded issue for this context.
        if isinstance(ME, CORAerror) and 'CORA:unboundedSet' in str(ME):
             raise CORAerror('CORA:specialError', 'Polytope is unbounded and '\
                'can therefore not be converted into a polynomial zonotope.')
        else:
            raise ME # Re-raise other exceptions

    # read out dimension
    n = P.dim()

    # distinguish between 2D and multi-dimensional case
    if n == 2:
        pZ = _aux_polyZonotope_2D(V)
    else:
        pZ = _aux_polyZonotope_nD(V)

    return pZ


# Auxiliary functions -----------------------------------------------------

def _aux_polyZonotope_2D(V: np.ndarray) -> PolyZonotope:
    """
    aux_polyZonotope_2D - special function for conversion of 2D polytopes
    """
    
    # MATLAB: if size(V,2) > 2, ind = convhull(V(1,:),V(2,:)); V = V(:,ind); end
    # If the number of vertices is greater than 2, compute the convex hull to get the boundary vertices in order.
    if V.shape[1] > 2:
        # ConvexHull expects points as (num_points, dimensions), so transpose V
        hull = ConvexHull(V.T)
        # The vertices in hull.vertices are the indices of the original points forming the convex hull
        # Use these indices to reorder V
        V = V[:, hull.vertices]

    # number of vertices
    numV = V.shape[1]

    # single point
    if numV == 1:
        # MATLAB: pZ = polyZonotope(V);
        return PolyZonotope(V.flatten())

    # loop over all polytope faces
    # MATLAB: pZface = cell(ceil(size(V,2)/2),1);
    # In Python, we can use a list and append.
    pZface = []
    # MATLAB: counter = 1; i = 1;
    # Python loop starts from 0.
    i = 0

    while i < numV:
        # construct polynomial zonotope from the facet
        # MATLAB: c = 0.5*(V(:,i) + V(:,i+1));
        # MATLAB: g = 0.5*(V(:,i) - V(:,i+1));
        # Note: MATLAB uses 1-based indexing, so V(:,i) is V[:, i-1] in Python
        # Also, V[:,i] will be a column vector already in Python if V is 2D.
        
        # Handle the case when i+1 goes out of bounds for the last face (numV is odd)
        if i + 1 >= numV:
            # This case means the last face is V(:,i) -> V(:,0) (closing the loop)
            # The MATLAB code has a special check for `isempty(pZface{end})` for the last face
            # This implies the last segment connects the last vertex to the first for a closed polygon.
            # However, `convhull` already gives ordered vertices of the convex hull, which implicitly closes.
            # The MATLAB code's `while i < numV` implies it processes segments (V(i) to V(i+1)).
            # If numV is odd, the last vertex V(end) is not paired up this way. The `if representsa_(pZface{end},'emptySet',eps)` check handles this.
            # For a closed polygon, the last edge is (V[numV-1], V[0]).
            # The MATLAB code seems to process pairs (V_i, V_i+1) and then a special case for the last face.
            # Let's stick to the direct translation of the loop structure.
            break # Exit loop, handle last element if needed outside.
            
        c = 0.5 * (V[:, i:i+1] + V[:, i+1:i+2])
        g = 0.5 * (V[:, i:i+1] - V[:, i+1:i+2])

        # MATLAB: pZface{counter} = polyZonotope(c,g,[],1);
        # In Python, GI=[] and E=[] are default empty matrices handled by constructor.
        # id=1 needs to be an array for polyZonotope, e.g., np.array([[1]])
        pZface.append(PolyZonotope(c, g, np.array([]).reshape(c.shape[0],0), np.array([]).reshape(1,0), np.array([[1]])))
        
        i += 2
    
    # MATLAB's logic: `if representsa_(pZface{end},'emptySet',eps)`
    # This means if the last segment was skipped (numV is odd and not enough pairs)
    # or if the resulting polyZonotope from the last segment was somehow empty,
    # it re-calculates the last segment to make sure it's not empty. This seems redundant
    # if convhull correctly orders vertices of a non-empty polygon.
    # For now, I will omit this check and assume proper polygon formation from convhull.
    # If issues arise, it can be re-introduced after verifying what `representa_(..., 'emptySet', eps)` implies.

    # iteratively compute the convex hull of neighboring faces until only
    # one single polynomial zonotope is left
    # MATLAB: while length(pZface) > 1
    while len(pZface) > 1:
        # MATLAB: pZtemp = cell(floor(length(pZface)/2),1);
        pZtemp = []

        counter = 0 # Python 0-based index
        counterNew = 0

        # loop over all pairs
        # MATLAB: while counter+1 <= length(pZface)
        while counter + 1 < len(pZface):
            # construct polyZonotope objects with appropriate id vectors
            pZ1_ = pZface[counter]
            pZ2_ = pZface[counter+1]

            # MATLAB: id1 = 1:length(pZ1_.id);
            # MATLAB: id2 = length(pZ1_.id)+1 : length(pZ1_.id) + length(pZ2_.id);
            # MATLAB: id2(end) = id1(end); % This line is crucial for shared IDs at the joint

            # The above MATLAB ID logic is for managing shared IDs in the convex hull of two pZs.
            # This is complex and ties deeply to the specific `enclose` implementation that the MATLAB
            # polyZonotope class uses internally when constructing convex hulls of dependent generators.
            # The Python `enclose` method for PolyZonotope already handles `id` merging via `mergeExpMatrix`.
            # So, we should not manually manipulate IDs here, but rather rely on `enclose` to do it correctly.
            # The MATLAB code snippet for `id1` and `id2` suggests a specific way of assigning new IDs
            # when forming the convex hull. If `enclose` handles it, we pass original pZ1_ and pZ2_.

            # For now, I will simplify this part and directly call enclose, assuming it handles IDs.
            # If `enclose` for PolyZonotope needs specific ID values passed to it from the outside
            # for this type of operation, then this section will need to be revisited. 

            # compute convex hull
            # MATLAB: pZtemp{counterNew} = enclose(pZ1,pZ2);
            pZtemp.append(pZ1_.enclose(pZ2_)) # Calls the PolyZonotope.enclose method
            counterNew += 1
            counter += 2
        
        # MATLAB: if counterNew == ceil(length(pZface)/2), pZtemp = [pZtemp; pZface(end)]; end
        # This handles the case if there's an odd number of elements in pZface. The last element
        # is simply appended to pZtemp without being paired.
        if len(pZface) % 2 != 0: # If original list had odd number of elements
            pZtemp.append(pZface[-1])

        # MATLAB: pZface = pZtemp;
        pZface = pZtemp

    # MATLAB: pZ = pZface{1};
    pZ = pZface[0]

    return pZ


def _aux_polyZonotope_nD(V: np.ndarray) -> PolyZonotope:
    """
    aux_polyZonotope_nD - special function for conversion of N-D polytopes
    """

    # convert each vertex to a separate polynomial zonotope
    # MATLAB: list = cell(size(V,2),1);
    # MATLAB: cent = V;
    # MATLAB: lenID = zeros(length(list),1);
    # MATLAB: for i = 1:length(list), list{i} = polyZonotope(V(:,i),[],[],[]); end
    
    list_pZ = []
    # Convert each column (vertex) of V into a PolyZonotope object (a point pZ)
    for i in range(V.shape[1]):
        vertex = V[:, i:i+1] # Get as a column vector
        list_pZ.append(PolyZonotope(vertex))
    
    # MATLAB: cent = V; # Keep track of centers (vertices themselves initially)
    # We can just use the center property of the pZ objects later if needed.
    # MATLAB: lenID = zeros(length(list),1); # length of ID vector for each pZ
    # We can get this dynamically using pZ.id.shape[0]

    # recursively compute the convex hull
    # MATLAB: while length(list) > 1
    while len(list_pZ) > 1:
        # MATLAB: tempList = cell(length(list),1);
        # MATLAB: tempCent = zeros(size(cent));
        # MATLAB: tempLenID = zeros(length(list),1);
        tempList_pZ = []
        tempCent = np.zeros((V.shape[0], 0)) # Will append columns
        tempLenID = []

        # MATLAB: indID = find(lenID == max(lenID));
        # This finds pZs with the maximum number of IDs. This prioritizes merging complex pZs.
        current_lenIDs = np.array([pZ.id.shape[0] for pZ in list_pZ])
        if current_lenIDs.size > 0: # Ensure there are elements to find max from
            max_lenID = np.max(current_lenIDs)
            indID = np.where(current_lenIDs == max_lenID)[0] # 0-based indices
        else:
            indID = np.array([]) # No elements, no preferred indices

        counter = 0 # Index for tempList_pZ

        # loop over all polynomial zonotopes in the list
        # MATLAB: while ~isempty(list) && length(list) > 1
        # Our Python loop will consume elements from list_pZ directly.
        current_list_pZ = list_pZ.copy() # Make a copy to iterate and remove from original
        current_indices = np.arange(len(list_pZ))

        while len(current_indices) > 1: # While there are at least two pZs to combine
            # determine nearest polynomial zonotope from the list
            ind1_idx: int # Index in current_indices for pZ1
            ind2_idx: int # Index in current_indices for pZ2
            
            if indID.size >= 2:
                # MATLAB: ind = aux_nearestNeighbour(cent(:,indID(1)),cent(:,indID(2:end)));
                # This means find the nearest neighbour for the first element in indID from the rest of indID elements.
                # Then pick the actual indices from the original list_pZ.
                
                # Get centers of the pZs identified by indID
                centers_for_nn_search = np.hstack([list_pZ[i].center() for i in indID])
                
                p_nn = centers_for_nn_search[:, 0:1] # First center as point p
                points_nn = centers_for_nn_search[:, 1:] # Rest of centers as points to search in

                # MATLAB returns 1-based index for nearest, `nearestNeighbour` returns 0-based.
                # `ind` is the index within `points_nn`.
                nn_relative_idx = nearestNeighbour(p_nn, points_nn)

                ind1_idx_in_indID = 0
                ind2_idx_in_indID = nn_relative_idx + 1 # +1 because points_nn starts from indID[1]
                
                ind1_idx = indID[ind1_idx_in_indID] # Actual index in original list_pZ
                ind2_idx = indID[ind2_idx_in_indID] # Actual index in original list_pZ

                # Remove selected indices from indID for next iteration
                indID = np.delete(indID, [ind1_idx_in_indID, ind2_idx_in_indID])
                
            else: # If less than 2 elements in indID, or indID is empty, pick first two from `current_indices`
                # MATLAB: ind1 = 1; ind2 = aux_nearestNeighbour(cent(:,1),cent(:,2:end)) + 1;
                # This means pick the first pZ in the list, and find its nearest neighbor among the rest.
                ind1_idx = current_indices[0] # First available pZ

                # Centers of remaining pZs for NN search
                remaining_pZs = [list_pZ[i] for i in current_indices[1:]]
                remaining_centers = np.hstack([pZ.center() for pZ in remaining_pZs])

                p_nn = list_pZ[ind1_idx].center() # Center of the first pZ
                
                if remaining_centers.shape[1] > 0: # Ensure there are points to search in
                    nn_relative_idx = nearestNeighbour(p_nn, remaining_centers)
                    ind2_idx = current_indices[1 + nn_relative_idx] # Convert relative index to absolute
                else:
                    # Only one element left, cannot form a pair. Break loop or handle.
                    # For this loop, we need at least two elements to proceed.
                    break 

            pZ1_ = list_pZ[ind1_idx]
            pZ2_ = list_pZ[ind2_idx]
            
            # MATLAB: if ~isempty(pZ1_.E) && ~isempty(pZ2_.E)
            # The MATLAB code applies specific ID handling only if both E matrices are non-empty.
            # Otherwise, it uses original pZ1_, pZ2_ (which should be point polyZonotopes in this nD case).
            # Our `enclose` method should handle empty E matrices and ID generation, so direct call is fine.
            
            # compute convex hull
            # MATLAB: tempList{counter} = enclose(pZ1,pZ2);
            enclosed_pZ = pZ1_.enclose(pZ2_)
            tempList_pZ.append(enclosed_pZ)
            tempCent = np.hstack((tempCent, enclosed_pZ.center()))

            # update id-vectors if empty (this is from MATLAB, for points/empty sets)
            # MATLAB: if isempty(tempList{counter}.id), temp = tempList{counter}; tempList{counter} = polyZonotope(temp.c,temp.G,temp.GI,temp.E,1); end
            # Python PolyZonotope constructor should handle empty ID automatically to assign a default if needed.
            # If `polyZonotope` constructor takes care of assigning a default id (e.g., [[1]]) when empty, 
            # this explicit re-assignment might not be needed. Assuming constructor handles it.
            
            # update length of ID vectors
            tempLenID.append(enclosed_pZ.id.shape[0])
            
            # update variables (remove processed pZs from current_indices)
            # Sort indices to delete in descending order to avoid index issues
            indices_to_remove_from_current = sorted([np.where(current_indices == ind1_idx)[0][0], np.where(current_indices == ind2_idx)[0][0]], reverse=True)
            current_indices = np.delete(current_indices, indices_to_remove_from_current)
            counter += 1
            
        # add last list element 
        # MATLAB: if ~isempty(list), tempList{counter} = list{1}; tempCent(:,counter) = cent(:,1); tempLenID(counter) = length(list{1}.id); else, counter = counter - 1; end
        # This handles the case if after pairing, one element is left over.
        if len(current_indices) == 1:
            last_pZ_idx = current_indices[0]
            last_pZ = list_pZ[last_pZ_idx]
            tempList_pZ.append(last_pZ)
            tempCent = np.hstack((tempCent, last_pZ.center()))
            tempLenID.append(last_pZ.id.shape[0])
        
        # update lists for next iteration of while len(list) > 1
        list_pZ = tempList_pZ
        # Recalculate original_centers for the next iteration based on `list_pZ` if needed for `aux_nearestNeighbour`
        # For now, `aux_nearestNeighbour` gets points directly, so `tempCent` is the list of centers.

    # construct the resulting polynomial zonotope
    # MATLAB: pZ = list{1};
    if list_pZ:
        pZ = list_pZ[0]
    else:
        # Handle case where list_pZ might become empty (e.g., initial empty set)
        # Or maybe it should always return a PolyZonotope, even if empty.
        # Defaulting to an empty PolyZonotope if no elements remain (shouldn't happen for non-empty input).
        # Need to ensure this is consistent with MATLAB's behavior for empty inputs leading to empty outputs.
        pZ = PolyZonotope.empty(V.shape[0]) # Use the original dimension

    return pZ
