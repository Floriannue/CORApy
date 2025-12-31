"""Test contract union logic with splits"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import contract, _aux_bestSplit

def f(x):
    return x[0]**2 + x[1]**2 - 4

dom = Interval([1, 1], [3, 3])

print("=== Simulating splitting logic ===")
list_intervals = [dom]

for i in range(2):  # splits = 2
    print(f"\n--- Split iteration {i} ---")
    domSplit = _aux_bestSplit(f, list_intervals[0] if len(list_intervals) > 0 else dom)
    print(f"domSplit: {domSplit}")
    
    list_ = [None] * (2 * len(list_intervals))
    counter = 0
    
    for j in range(len(list_intervals)):
        print(f"\n  Processing interval {j}: {list_intervals[j]}")
        
        for k in range(len(domSplit)):
            print(f"    Split {k}: {domSplit[k]}")
            temp = f(domSplit[k])
            p = np.zeros(len(temp) if hasattr(temp, '__len__') else 1)
            
            if not temp.contains_(p):
                print(f"      Skipped: f(domSplit[{k}]) does not contain 0")
                continue
            
            print(f"      Contracting split {k}...")
            domTemp = contract(f, domSplit[k], 'interval', 2, None)
            print(f"      Result: {domTemp}")
            
            if domTemp is not None and not domTemp.representsa_('emptySet', np.finfo(float).eps):
                list_[counter] = domTemp
                counter += 1
                print(f"      Added to list at index {counter-1}")
            else:
                print(f"      Not added: domTemp is None or empty")
    
    list_intervals = list_[:counter]
    print(f"\n  list_intervals after iteration {i}: {list_intervals}")
    print(f"  counter: {counter}")

print(f"\n=== Final union ===")
if len(list_intervals) > 0:
    res = list_intervals[0]
    print(f"  Starting with: {res}")
    for i in range(1, len(list_intervals)):
        print(f"  Unioning with: {list_intervals[i]}")
        res = res | list_intervals[i]
        print(f"  Result: {res}")
    print(f"\nFinal result: {res}")
else:
    print("  No intervals to union, returning None")

