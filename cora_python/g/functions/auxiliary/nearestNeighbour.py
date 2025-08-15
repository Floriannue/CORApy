import numpy as np

def nearestNeighbour(p: np.ndarray, points: np.ndarray) -> int:
    """
    aux_nearestNeighbour - determines the nearest neighbour of a point

    Syntax:
        ind = aux_nearestNeighbour(p,points)

    Inputs:
        p - column vector
        points - matrix of column vectors

    Outputs:
        ind - index of the nearest point

    """

    # calculate the squared Euclidean distance
    # MATLAB: dist = sum((points-p*ones(1,size(points,2))).^2,1);
    # In numpy, broadcasting handles this for us.
    # p has shape (n, 1), points has shape (n, m)
    # (points - p) will broadcast p to (n, m)
    # (points - p)**2 squares element-wise
    # sum(..., axis=0) sums along the columns, resulting in (1, m) array of squared distances
    dist = np.sum((points - p)**2, axis=0)

    # get index of the nearest point
    # MATLAB: [~,ind] = min(dist);
    # In numpy, np.argmin returns the index of the minimum value.
    ind = np.argmin(dist)

    # MATLAB returns 1-based index, Python uses 0-based.
    # The MATLAB code also does `ind = ind(1)` if multiple minimums exist.
    # np.argmin already returns the first occurrence's index.
    return int(ind)
