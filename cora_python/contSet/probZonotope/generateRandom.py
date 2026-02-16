"""
generateRandom - generates a random probabilistic zonotope

Python API (positional and/or keyword; no MATLAB-style string name-value pairs):
    pZ = ProbZonotope.generateRandom()
    pZ = ProbZonotope.generateRandom(n)                    # dimension
    pZ = ProbZonotope.generateRandom(n, nrGens)           # dimension, nr_generators
    pZ = ProbZonotope.generateRandom(n, nrGens, nrProb)   # dimension, nr_generators, nr_prob_generators
    pZ = ProbZonotope.generateRandom(dimension=n)
    pZ = ProbZonotope.generateRandom(dimension=n, nr_generators=nrGens)

Positional order: (dimension, nr_generators, nr_prob_generators). Keywords override.

Keyword arguments (all optional):
    - dimension (int): ambient dimension
    - center (ndarray): center vector (overrides dimension)
    - nr_generators (int): number of deterministic generators
    - nr_prob_generators (int): number of probabilistic generators

Returns:
    ProbZonotope: random probabilistic zonotope

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: zonotope.generateRandom

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       27-September-2019 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.preprocessing.NVpairsPlainSetDefaultValues import NVpairsPlainSetDefaultValues
from cora_python.contSet.probZonotope.probZonotope import ProbZonotope
from cora_python.contSet.zonotope.zonotope import Zonotope

if TYPE_CHECKING:
    pass


def generateRandom(*args, **kwargs) -> ProbZonotope:
    """
    Generates a random probabilistic zonotope.

    Supports positional and/or keyword args (no MATLAB-style string names).
    Positional order: (dimension, nr_generators, nr_prob_generators).
    Keywords override positionals.

    Returns:
        pZ: random probZonotope object
    """
    # Positional order: (dimension, nr_generators, nr_prob_generators). Keywords override.
    pos_names = ('dimension', 'nr_generators', 'nr_prob_generators')
    merged = {}
    for i, name in enumerate(pos_names):
        if i < len(args):
            merged[name] = args[i]
    merged.update(kwargs)
    kwargs = merged

    # Default values (Python-style lowercase parameter names)
    listOfNameValuePairs = [
        'dimension', 2,
        'nr_generators', 5,
        'nr_prob_generators', 2
    ]

    # Parse input arguments (center optional, not in default list)
    [n, nrGens, nrProbGens] = NVpairsPlainSetDefaultValues(
        listOfNameValuePairs, list(kwargs.keys()), list(kwargs.values()))
    c = kwargs.get('center')
    if c is not None:
        c = np.asarray(c).reshape(-1, 1)
        n = c.shape[0]
    
    # Check input arguments
    inputArgsCheck([
        [n, 'att', 'numeric', ['scalar', 'positive', 'integer']],
        [nrGens, 'att', 'numeric', ['scalar', 'nonnegative', 'integer']],
        [nrProbGens, 'att', 'numeric', ['scalar', 'nonnegative', 'integer']]
    ])
    
    # Generate random zonotope
    Z = Zonotope.generateRandom(dimension=n, center=c, nr_generators=nrGens)

    # Convert to numeric zonotope matrix Z = [c, G] as in MATLAB
    if Z.G.size > 0:
        Z_mat = np.hstack([Z.c, Z.G])
    else:
        Z_mat = Z.c
    
    # Generate random probabilistic generators
    g = -1 + 2 * np.random.rand(n, nrProbGens)
    
    # Default gamma value
    gamma = 2

    # Constructor takes positional arguments (Z, g, gamma)
    return ProbZonotope(Z_mat, g, gamma) 