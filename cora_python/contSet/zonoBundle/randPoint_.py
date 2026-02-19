"""
randPoint_ - generates a random point within a zonotope bundle

Syntax:
    p = randPoint_(zB)
    p = randPoint_(zB,N)
    p = randPoint_(zB,N,type)
    p = randPoint_(zB,'all','extreme')

Inputs:
    zB - zonoBundle object
    N - number of random points
    type - type of the random point ('standard' or 'extreme')

Outputs:
    p - random point in R^n

Example: 
    Z1 = zonotope([0 1 2 0;0 1 0 2]);
    Z2 = zonotope([3 -0.5 3 0;-1 0.5 0 3]);
    zB = zonoBundle({Z1,Z2});
 
    points = randPoint(zB,100);
   
    figure; hold on;
    plot(zB,[1,2],'r');
    plot(points(1,:),points(2,:),'.k','MarkerSize',10);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/randPoint, conZonotope/randPoint_

Authors:       Matthias Althoff
Written:       18-August-2016 
Last update:   19-August-2022 (MW, integrate standardized pre-processing)
Last revision: 27-March-2023 (MW, rename randPoint_)
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Union
import numpy as np

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def randPoint_(zB: 'ZonoBundle', N: Union[int, str] = 1, type_: str = 'standard') -> np.ndarray:
    """
    Generate random points within a zonotope bundle.
    """
    # return all extreme points
    if isinstance(N, str) and N == 'all':
        return zB.vertices_()

    # generate random points
    if type_ == 'standard':
        Vmat = zB.vertices_()
        nr_of_vertices = Vmat.shape[1]
        if nr_of_vertices == 0:
            return Vmat

        # random convex combination
        alpha = np.random.rand(nr_of_vertices, int(N))
        alpha_norm = alpha / np.sum(alpha, axis=0, keepdims=True)

        return Vmat @ alpha_norm

    elif type_ == 'extreme':
        N_int = int(N) if not isinstance(N, str) else 1
        p = np.zeros((zB.dim(), N_int))

        # center polytope at origin
        c = zB.center()
        temp = zB + (-c)

        for i in range(N_int):
            # select random direction
            d = np.random.rand(c.shape[0], 1) - 0.5 * np.ones((c.shape[0], 1))
            d = d / np.linalg.norm(d)

            # compute farthest point in this direction
            _, x = temp.supportFunc_(d, 'upper')
            p[:, i] = (x + c).reshape(-1)

        return p

    else:
        raise CORAerror('CORA:noSpecificAlg', type_, zB)

