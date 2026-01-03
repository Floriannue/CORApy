"""Test _aux_power_ with zero constraint"""
from cora_python.specification.syntaxTree import _aux_power_
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

# Test: constraint [0, 0] for x^2 with previous value [1, 3]
try:
    res = _aux_power_(Interval([0], [0]), 2, Interval([1], [3]))
    print('Result:', res.inf, res.sup)
except CORAerror as e:
    print('Error:', e.identifier)

# Test: constraint [-3, -1] for x^2 with previous value [1, 9]
try:
    res = _aux_power_(Interval([-3], [-1]), 2, Interval([1], [9]))
    print('Result:', res.inf, res.sup)
except CORAerror as e:
    print('Error:', e.identifier)



