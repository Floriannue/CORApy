"""
empty - instantiates an empty ellipsoid
%
% Syntax:
%    E = empty(n)
%
% Inputs:
%    n - dimension
%
% Outputs:
%    E - empty ellipsoid
%
% Example:
%    E = ellipsoid.empty(2)
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: none

% Authors:       Mark Wetzlinger (MATLAB)
%                Python translation by AI Assistant
% Written:       09-January-2024 (MATLAB)
% Last update:   15-January-2024 (TL, parse input, MATLAB)
% Python translation: 2025
"""

import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


def empty(n: int) -> 'Ellipsoid':
    """
    empty - instantiates an empty ellipsoid

    Syntax:
        E = empty(n)

    Inputs:
        n - dimension of the ellipsoid

    Outputs:
        E - generated empty ellipsoid object

    Example:
        E = ellipsoid.empty(2);
        % true
        isemptyobject(E)

    Other m-files required: none
    Subfunctions: none
    MAT-files required: none

    See also: none

    Authors:       Victor Gassmann, Matthias Althoff
    Written:       13-March-2019
    Last update:   16-October-2019
    Last revision: 16-June-2023 (MW, restructure using auxiliary functions)
    Automatic python translation: Florian Nüssel BA 2025
    """
    # An empty ellipsoid is represented by an empty (0x0) shape matrix.
    # The center vector q should also be empty to be consistent with MATLAB's `isempty`.
    Q_empty = np.array([[]]).reshape(0, 0)
    q_initial = np.zeros((n, 0))  # q should be an n x 0 zero vector to be empty in MATLAB sense
    return Ellipsoid(Q_empty, q_initial) 