"""Test 'all' contractor with 2 splits - trace through iterations"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import contract, _aux_bestSplit

def f(x):
    return x[0]**2 + x[1]**2 - 4

dom = Interval([1, 1], [3, 3])

print("=== Simulating 'all' contractor with splits=2 ===")
print(f"Initial domain: {dom}")

# Simulate splitting logic
list_intervals = [dom]

for i in range(2):  # splits = 2
    print(f"\n--- Split iteration {i+1} ---")
    print(f"list_intervals: {list_intervals}")
    
    list_ = [None] * (2 * len(list_intervals))
    counter = 0
    
    for j in range(len(list_intervals)):
        print(f"\n  Processing interval {j}: {list_intervals[j]}")
        domSplit = _aux_bestSplit(f, list_intervals[j])
        print(f"  Split result: {domSplit}")
        
        for k in range(len(domSplit)):
            print(f"\n    Processing split {k}: {domSplit[k]}")
            
            # Check if domain is empty
            temp = f(domSplit[k])
            p = np.zeros(len(temp) if hasattr(temp, '__len__') else 1)
            
            if not temp.contains_(p):
                print(f"    SKIPPED: f(split[{k}]) does not contain 0")
                continue
            
            # Contract the splitted domain (recursive call)
            print(f"    Contracting with 'all' algorithm...")
            domTemp = contract(f, domSplit[k], 'all', 2, None)
            print(f"    Contract result: {domTemp}")
            
            if domTemp is None:
                print(f"    SKIPPED: domTemp is None")
            elif domTemp.representsa_('emptySet', np.finfo(float).eps):
                print(f"    SKIPPED: domTemp is empty set")
            else:
                print(f"    ADDED to list at index {counter}")
                list_[counter] = domTemp
                counter += 1
    
    list_intervals = list_[:counter]
    print(f"\n  list_intervals after iteration {i+1}: {list_intervals}")
    print(f"  counter: {counter}")
    
    if counter == 0:
        print(f"  WARNING: No intervals survived iteration {i+1}!")
        break

print(f"\n=== Final result ===")
if len(list_intervals) > 0:
    res = list_intervals[0]
    for i in range(1, len(list_intervals)):
        res = res | list_intervals[i]
    print(f"Result: {res}")
else:
    print("Result: None")

