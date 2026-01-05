"""Quick verification script to check display pattern consistency"""
import sys
import io
from cora_python.contSet.interval import Interval
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.zonotope import Zonotope
from cora_python.hybridDynamics.transition import Transition
from cora_python.contDynamics.linearSys import LinearSys
import numpy as np

test_classes = [
    ('Interval', Interval([1], [2])),
    ('Polytope', Polytope(np.array([[1,0],[-1,0]]), np.array([[1],[0]]))),
    ('Zonotope', Zonotope(np.array([[0],[0]]), np.array([[1,0],[0,1]]))),
    ('Transition', Transition()),
    ('LinearSys', LinearSys(np.array([[0,1],[0,0]]))),
]

print('Testing display pattern consistency:')
print('=' * 70)
all_pass = True

for name, obj in test_classes:
    try:
        display_str = obj.display_()
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        obj.display()
        printed = buffer.getvalue()
        sys.stdout = old_stdout
        str_result = str(obj)
        match = (display_str == printed == str_result)
        status = 'PASS' if match else 'FAIL'
        print(f'{name:15} {status} - display_==display==__str__')
        if not match:
            all_pass = False
            print(f'  display_ length: {len(display_str)}')
            print(f'  display() length: {len(printed)}')
            print(f'  __str__ length: {len(str_result)}')
    except Exception as e:
        print(f'{name:15} ERROR - {e}')
        all_pass = False

print('=' * 70)
print(f'Overall: {"ALL PASS" if all_pass else "SOME FAILED"}')

