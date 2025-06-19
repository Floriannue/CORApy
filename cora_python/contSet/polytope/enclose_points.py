# This file is part of the CORA project.
# Copyright (c) 2024, System Control and Robotics Group, TU Graz.
# All rights reserved.

import numpy as np
from scipy.spatial import ConvexHull
from typing import Union
from cora_python.contSet.polytope.polytope import Polytope

def enclose_points(points: np.ndarray) -> Polytope:
    """
    enclosePoints - encloses a point cloud with a polytope

    Syntax:
       P_out = enclose_points(points)

    Inputs:
       points - matrix storing point cloud (dimension: [n,p] for p points)

    Outputs:
       P_out - polytope object

    Example: 
       # mu = [2 3];
       # sigma = [1 1.5; 1.5 3];
       # points = mvnrnd(mu,sigma,100)';
       # P = Polytope.enclose_points(points);
       # figure; hold on
       # plot(points(1,:),points(2,:),'.k');
       # plot(P,[1,2],'r');

    Other m-files required: none
    Subfunctions: none
    MAT-files required: none

    See also: zonotope/enclosePoints, interval/enclosePoints

    Authors:       Niklas Kochdumper, Victor Gassmann
    Written:       05-May-2020
    Last update:   17-March-2021 (now also works for degenerate point cloud)
    Last revision: ---
    """

    n, N = points.shape
    Q = np.eye(n)

    # check whether points are lower-dimensional
    rank_points = np.linalg.matrix_rank(points)
    if rank_points < n:
        # In MATLAB, qr decomposition often returns a Q that can be used for basis transformation.
        # In NumPy, `np.linalg.qr` returns `Q` and `R` where `Q` is unitary/orthogonal.
        # We need an orthogonal basis for the column space of `points`.
        # One way is to use SVD or QR. Let's stick to QR but ensure correct basis.
        Q_qr, _ = np.linalg.qr(points)
        # Ensure Q is an orthogonal basis for the original space
        # We need to pick the first `rank_points` columns of Q_qr that form an orthonormal basis
        # for the column space of points.
        Q = Q_qr
        points = Q.T @ points
        # remove zeros
        points = points[:rank_points, :]
        r = rank_points # Update r to the actual rank
    else:
        r = n # If full rank, r is the dimension

    # compute convex hull
    if points.shape[0] > 1: # if dimension > 1
        # scipy.spatial.ConvexHull expects points as (num_points, num_dimensions)
        # So, transpose points before passing it to ConvexHull
        hull = ConvexHull(points.T)
        ind = hull.vertices
    else: # 1D case
        # For 1D, the convex hull are simply the min and max points.
        # Find indices of min and max points
        ii_max = np.argmax(points)
        ii_min = np.argmin(points)
        ind = np.unique([ii_max, ii_min]) # Use unique to handle cases where min and max are the same point

    # Select points based on convex hull indices
    points = points[:, ind]

    # add zeros again and backtransform
    # This part needs to be careful: if points were lower-dimensional, they were reduced.
    # Now we need to pad them back to original dimension `n` before `Q` multiplication.
    if r < n:
        padding_zeros = np.zeros((n - r, points.shape[1]))
        points = np.vstack((points, padding_zeros))
    
    points = Q @ points # Backtransform using original Q

    # construct polytope object
    P_out = Polytope(points)

    return P_out 