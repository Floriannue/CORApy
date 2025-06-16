import sys
sys.path.insert(0, 'cora_python')
from cora_python.contSet.interval.interval import Interval

I = Interval([-2, -1], [3, 4])

try:
    result = I.contains([0, 0])
    print(f'Success: {result} (type: {type(result)})')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc() 