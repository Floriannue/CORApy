import numpy as np
from typing import Union, Optional, Tuple, Any

from cora_python.contSet.contSet.contSet import ContSet
from cora_python.g.functions.matlab.validate.check import inputArgsCheck
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check import assertNarginConstructor


class EmptySet(ContSet):
    """
    emptySet - object constructor for empty sets

    Description:
       This class represents empty sets.

    Syntax:
       obj = EmptySet(n)

    Inputs:
       n - dimension

    Outputs:
       obj - generated EmptySet object

    Example:
       n = 2;
       O = EmptySet(n);
       # plot(O); # Plotting functionality to be implemented later

    Other m-files required: none
    Subfunctions: none
    MAT-files required: none

    See also: none

    Authors:       Manuel Schaller
    Written:       09-November-2015
    Last update:   15-January-2024 (TL, constructor simplified)
                   03-March-2025 (TL, parse input)
    Last revision: ---
    """

    def __init__(self, n=0):
        # input check
        inputArgsCheck([[n, 'att', ['numeric'], {'scalar', 'nonnegative'}]])

        # generate empty set
        self.dimension = n

    def isemptyobject(self) -> bool:
        # isemptyobject - checks if an emptySet object is empty

        # Syntax:
        #    res = isemptyobject(obj)

        # Inputs:
        #    obj - emptySet object

        # Outputs:
        #    res - true if object is empty

        # Example:
        #    obj = EmptySet(2);
        #    res = isemptyobject(obj);

        # Authors:       Mark Wetzlinger
        # Written:       29-March-2019
        # Last update:   ---
        # Last revision: ---

        return True

    def ismember(self, point: np.ndarray) -> bool:
        # ismember - checks if a point is a member of an empty set

        # Syntax:
        #    res = ismember(obj, point)

        # Inputs:
        #    obj - emptySet object
        #    point - point to be checked

        # Outputs:
        #    res - true if point is a member of the set

        # Example:
        #    obj = EmptySet(2);
        #    res = ismember(obj, np.array([[1], [2]]));

        # Authors:       Mark Wetzlinger
        # Written:       29-March-2019
        # Last update:   ---
        # Last revision: ---

        inputArgsCheck([
            [self, 'att', ['EmptySet'], {'scalar'}] ,
            [point, 'att', ['numeric'], {'column', 'real', 'finite'}] ,
        ])

        return False

    def mtimes(self, factor: Union[float, np.ndarray]) -> 'EmptySet':
        # mtimes - computes the product of an emptySet object and a scalar or matrix

        # Syntax:
        #    res = mtimes(obj, factor)

        # Inputs:
        #    obj - emptySet object
        #    factor - scalar or matrix

        # Outputs:
        #    res - emptySet object

        # Example:
        #    obj = EmptySet(2);
        #    res = mtimes(obj, 2);

        # Authors:       Mark Wetzlinger
        # Written:       29-March-2019
        # Last update:   ---
        # Last revision: ---

        return self

    def plus(self, summand: Union[float, np.ndarray]) -> 'EmptySet':
        # plus - computes the sum of an emptySet object and a scalar or matrix

        # Syntax:
        #    res = plus(obj, summand)

        # Inputs:
        #    obj - emptySet object
        #    summand - scalar or matrix

        # Outputs:
        #    res - emptySet object

        # Example:
        #    obj = EmptySet(2);
        #    res = plus(obj, 2);

        # Authors:       Mark Wetzlinger
        # Written:       29-March-2019
        # Last update:   ---
        # Last revision: ---

        return self

    def minus(self, subtrahend: Union[float, np.ndarray]) -> 'EmptySet':
        # minus - computes the difference of an emptySet object and a scalar or matrix

        # Syntax:
        #    res = minus(obj, subtrahend)

        # Inputs:
        #    obj - emptySet object
        #    subtrahend - scalar or matrix

        # Outputs:
        #    res - emptySet object

        # Example:
        #    obj = EmptySet(2);
        #    res = minus(obj, 2);

        # Authors:       Mark Wetzlinger
        # Written:       29-March-2019
        # Last update:   ---
        # Last revision: ---

        return self

    def or_(self, obj: 'EmptySet') -> 'EmptySet':
        # or_ - computes the union of two emptySet objects

        # Syntax:
        #    res = or_(obj1, obj2)

        # Inputs:
        #    obj1 - emptySet object
        #    obj2 - emptySet object

        # Outputs:
        #    res - emptySet object

        # Example:
        #    obj1 = EmptySet(2);
        #    obj2 = EmptySet(2);
        #    res = or_(obj1, obj2);

        # Authors:       Mark Wetzlinger
        # Written:       29-March-2019
        # Last update:   ---
        # Last revision: ---

        return self

    def and_(self, obj: 'EmptySet') -> 'EmptySet':
        # and_ - computes the intersection of two emptySet objects

        # Syntax:
        #    res = and_(obj1, obj2)

        # Inputs:
        #    obj1 - emptySet object
        #    obj2 - emptySet object

        # Outputs:
        #    res - emptySet object

        # Example:
        #    obj1 = EmptySet(2);
        #    obj2 = EmptySet(2);
        #    res = and_(obj1, obj2);

        # Authors:       Mark Wetzlinger
        # Written:       29-March-2019
        # Last update:   ---
        # Last revision: ---

        return self

    def sup(self, direction: np.ndarray) -> float:
        # sup - computes the supremum of an emptySet object in a given direction

        # Syntax:
        #    res = sup(obj, direction)

        # Inputs:
        #    obj - emptySet object
        #    direction - direction vector

        # Outputs:
        #    res - supremum in the given direction

        # Example:
        #    obj = EmptySet(2);
        #    res = sup(obj, np.array([[1], [0]]));

        # Authors:       Mark Wetzlinger
        # Written:       29-March-2019
        # Last update:   ---
        # Last revision: ---

        raise CORAerror("CORA:emptySet", "The supremum of an empty set is undefined.")

    def inf(self, direction: np.ndarray) -> float:
        # inf - computes the infimum of an emptySet object in a given direction

        # Syntax:
        #    res = inf(obj, direction)

        # Inputs:
        #    obj - emptySet object
        #    direction - direction vector

        # Outputs:
        #    res - infimum in the given direction

        # Example:
        #    obj = EmptySet(2);
        #    res = inf(obj, np.array([[1], [0]]));

        # Authors:       Mark Wetzlinger
        # Written:       29-March-2019
        # Last update:   ---
        # Last revision: ---

        raise CORAerror("CORA:emptySet", "The infimum of an empty set is undefined.")

    def dim(self) -> int:
        # dim - returns the dimension of an emptySet object

        # Syntax:
        #    res = dim(obj)

        # Inputs:
        #    obj - emptySet object

        # Outputs:
        #    res - dimension of the empty set

        # Example:
        #    obj = EmptySet(2);
        #    res = dim(obj);

        # Authors:       Mark Wetzlinger
        # Written:       29-March-2019
        # Last update:   ---
        # Last revision: ---

        return self.dimension

    def isempty(self) -> bool:
        # isempty - checks if an emptySet object is empty

        # Syntax:
        #    res = isempty(obj)

        # Inputs:
        #    obj - emptySet object

        # Outputs:
        #    res - true if object is empty

        # Example:
        #    obj = EmptySet(2);
        #    res = isempty(obj);

        # Authors:       Mark Wetzlinger
        # Written:       29-March-2019
        # Last update:   ---
        # Last revision: ---

        return True

    def center(self) -> np.ndarray:
        # center - returns the center of an emptySet object

        # Syntax:
        #    res = center(obj)

        # Inputs:
        #    obj - emptySet object

        # Outputs:
        #    res - center of the empty set

        # Example:
        #    obj = EmptySet(2);
        #    res = center(obj);

        # Authors:       Mark Wetzlinger
        # Written:       29-March-2019
        # Last update:   ---
        # Last revision: ---

        raise CORAerror("CORA:emptySet", "The center of an empty set is undefined.") 